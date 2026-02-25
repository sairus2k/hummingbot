"""
Liquidity Mining V2 Controller

A V2 market-making controller optimized for liquidity mining reward programs.
Designed to maximize mining rewards by keeping tight two-sided quotes while
actively managing inventory to prevent one-sided depletion.

Key improvements over the legacy liquidity_mining strategy:
  1. Asymmetric spread adjustment — shifts spreads based on inventory imbalance
     (tighter on the side we want filled, wider on the other)
  2. Volatility-adaptive spreads using NATR — scales with real market conditions
  3. Campaign max_spread awareness — keeps orders within the reward-eligible range
  4. Dynamic budget rebalancing — continuously rebalances rather than one-shot at startup
  5. Multiple order levels per side for better fill probability
  6. Inventory-based reference price shift — nudges mid price to attract rebalancing fills
"""
from decimal import Decimal
from typing import List, Optional

import pandas_ta as ta  # noqa: F401
from pydantic import Field, field_validator

from hummingbot.core.data_type.common import PriceType, TradeType
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy_v2.controllers.market_making_controller_base import (
    MarketMakingControllerBase,
    MarketMakingControllerConfigBase,
)
from hummingbot.strategy_v2.executors.position_executor.data_types import PositionExecutorConfig
from hummingbot.strategy_v2.models.executor_actions import ExecutorAction, StopExecutorAction


class LiquidityMiningV2Config(MarketMakingControllerConfigBase):
    """
    Configuration for the Liquidity Mining V2 controller.

    Spread parameters are expressed as multiples of current NATR (normalized ATR).
    For example, buy_spreads=[0.5, 1.0] means the first buy level is placed at
    0.5 * NATR below the reference price, and the second at 1.0 * NATR.
    """
    controller_name: str = "liquidity_mining_v2"
    candles_config: List[CandlesConfig] = []

    # --- Candle data source ---
    candles_connector: Optional[str] = Field(
        default=None,
        json_schema_extra={
            "prompt": "Enter the candles connector (leave blank to use the trading connector): ",
            "prompt_on_new": False,
        },
    )
    candles_trading_pair: Optional[str] = Field(
        default=None,
        json_schema_extra={
            "prompt": "Enter the candles trading pair (leave blank to use the trading pair): ",
            "prompt_on_new": False,
        },
    )
    interval: str = Field(
        default="5m",
        json_schema_extra={
            "prompt": "Enter the candle interval (e.g., 1m, 5m, 15m): ",
            "prompt_on_new": True,
        },
    )

    # --- Volatility indicator ---
    natr_length: int = Field(
        default=14,
        json_schema_extra={
            "prompt": "Enter the NATR length (e.g., 14): ",
            "prompt_on_new": False,
        },
    )

    # --- Inventory management ---
    target_base_pct: Decimal = Field(
        default=Decimal("0.5"),
        json_schema_extra={
            "prompt": "Target base asset ratio (0.5 = 50/50 split between base and quote): ",
            "prompt_on_new": True,
            "is_updatable": True,
        },
    )
    inventory_skew_strength: Decimal = Field(
        default=Decimal("0.5"),
        json_schema_extra={
            "prompt": "Inventory skew strength (0 = none, 1 = aggressive rebalancing): ",
            "prompt_on_new": True,
            "is_updatable": True,
        },
    )

    # --- Campaign constraint ---
    max_spread: Optional[Decimal] = Field(
        default=None,
        json_schema_extra={
            "prompt": "Maximum spread for reward eligibility (e.g., 0.02 for 2%): ",
            "prompt_on_new": True,
            "is_updatable": True,
        },
    )

    @field_validator("candles_connector", mode="before")
    @classmethod
    def set_candles_connector(cls, v):
        if v is None or v == "":
            return None
        return v

    @field_validator("candles_trading_pair", mode="before")
    @classmethod
    def set_candles_trading_pair(cls, v):
        if v is None or v == "":
            return None
        return v

    @field_validator("max_spread", mode="before")
    @classmethod
    def parse_max_spread(cls, v):
        if v is None or v == "":
            return None
        return Decimal(str(v))


class LiquidityMiningV2Controller(MarketMakingControllerBase):
    """
    V2 controller for liquidity mining that combines:
      - NATR-based dynamic spreads (scale with volatility)
      - Asymmetric inventory skew (shift spreads AND reference price)
      - Campaign max_spread clamping
      - Multiple order levels per side
    """

    def __init__(self, config: LiquidityMiningV2Config, *args, **kwargs):
        self.config = config
        self._resolved_candles_connector = config.candles_connector or config.connector_name
        self._resolved_candles_pair = config.candles_trading_pair or config.trading_pair
        self._max_records = max(config.natr_length, 50) + 100

        if not config.candles_config:
            config.candles_config = [
                CandlesConfig(
                    connector=self._resolved_candles_connector,
                    trading_pair=self._resolved_candles_pair,
                    interval=config.interval,
                    max_records=self._max_records,
                )
            ]
        super().__init__(config, *args, **kwargs)

    # ------------------------------------------------------------------
    # Core data update — called every tick before determine_executor_actions
    # ------------------------------------------------------------------
    async def update_processed_data(self):
        """
        Compute:
          1. natr → spread_multiplier (volatility-scaled)
          2. inventory_ratio → reference price shift + spread asymmetry
          3. effective spreads clamped to max_spread
        """
        # --- 1. Get candle-derived volatility ---
        df = self.market_data_provider.get_candles_df(
            connector_name=self._resolved_candles_connector,
            trading_pair=self._resolved_candles_pair,
            interval=self.config.interval,
            max_records=self._max_records,
        )

        natr = Decimal("0.005")  # fallback: 0.5%
        if df is not None and len(df) > self.config.natr_length:
            natr_series = ta.natr(df["high"], df["low"], df["close"], length=self.config.natr_length)
            if natr_series is not None and len(natr_series) > 0:
                last_natr = natr_series.iloc[-1]
                if last_natr is not None and last_natr > 0:
                    natr = Decimal(str(last_natr)) / Decimal("100")

        # --- 2. Compute inventory ratio ---
        mid_price = self.market_data_provider.get_price_by_type(
            self.config.connector_name, self.config.trading_pair, PriceType.MidPrice
        )
        mid_price = Decimal(str(mid_price))
        inv_ratio = self._compute_inventory_ratio(mid_price)  # 0..1 (0 = all quote, 1 = all base)
        target = self.config.target_base_pct
        skew_strength = self.config.inventory_skew_strength

        # deviation: -1 (max short of base) to +1 (max excess of base)
        deviation = (inv_ratio - target) / max(target, Decimal("1") - target) if target > 0 else Decimal("0")
        deviation = max(Decimal("-1"), min(Decimal("1"), deviation))

        # --- 3. Reference price shift ---
        # When holding too much base (deviation > 0): shift reference price DOWN
        # to make sells tighter (more likely to fill) → reduces base.
        # When holding too little base (deviation < 0): shift reference price UP
        # to make buys tighter → accumulates base.
        max_price_shift = natr * Decimal("0.5") * skew_strength
        price_shift = -deviation * max_price_shift
        reference_price = mid_price * (Decimal("1") + price_shift)

        # --- 4. Spread asymmetry ---
        # Tighten the side we want to get filled, widen the side we want to avoid
        # buy_spread_factor < 1 when we want to buy more aggressively (deviation < 0)
        # sell_spread_factor < 1 when we want to sell more aggressively (deviation > 0)
        buy_spread_factor = Decimal("1") + deviation * skew_strength
        sell_spread_factor = Decimal("1") - deviation * skew_strength
        # Clamp to [0.2, 1.8] to avoid extreme or negative spreads
        buy_spread_factor = max(Decimal("0.2"), min(Decimal("1.8"), buy_spread_factor))
        sell_spread_factor = max(Decimal("0.2"), min(Decimal("1.8"), sell_spread_factor))

        # --- 5. Clamp spread_multiplier so final spreads respect max_spread ---
        spread_multiplier = natr
        if self.config.max_spread is not None and self.config.max_spread > 0:
            max_buy_spread_cfg = max(self.config.buy_spreads) if self.config.buy_spreads else 1.0
            max_sell_spread_cfg = max(self.config.sell_spreads) if self.config.sell_spreads else 1.0
            # Worst case effective spread on either side
            worst_factor = max(
                Decimal(str(max_buy_spread_cfg)) * buy_spread_factor,
                Decimal(str(max_sell_spread_cfg)) * sell_spread_factor,
            )
            if worst_factor > 0 and spread_multiplier * worst_factor > self.config.max_spread:
                spread_multiplier = self.config.max_spread / worst_factor

        self.processed_data = {
            "reference_price": reference_price,
            "spread_multiplier": spread_multiplier,
            "buy_spread_factor": buy_spread_factor,
            "sell_spread_factor": sell_spread_factor,
            "natr": natr,
            "inventory_ratio": inv_ratio,
            "deviation": deviation,
        }

    # ------------------------------------------------------------------
    # Override get_price_and_amount to apply asymmetric spread factors
    # ------------------------------------------------------------------
    def get_price_and_amount(self, level_id: str):
        level = self.get_level_from_level_id(level_id)
        trade_type = self.get_trade_type_from_level_id(level_id)
        spreads, amounts_quote = self.config.get_spreads_and_amounts_in_quote(trade_type)

        reference_price = Decimal(self.processed_data["reference_price"])
        spread_multiplier = Decimal(self.processed_data["spread_multiplier"])

        if trade_type == TradeType.BUY:
            side_factor = Decimal(self.processed_data["buy_spread_factor"])
        else:
            side_factor = Decimal(self.processed_data["sell_spread_factor"])

        spread_in_pct = Decimal(str(spreads[int(level)])) * spread_multiplier * side_factor

        # Clamp to max_spread if configured
        if self.config.max_spread is not None and self.config.max_spread > 0:
            spread_in_pct = min(spread_in_pct, self.config.max_spread)

        side_multiplier = Decimal("-1") if trade_type == TradeType.BUY else Decimal("1")
        order_price = reference_price * (Decimal("1") + side_multiplier * spread_in_pct)

        amount_base = Decimal(str(amounts_quote[int(level)])) / order_price
        return order_price, amount_base

    # ------------------------------------------------------------------
    # Executor config — simple PositionExecutor per level
    # ------------------------------------------------------------------
    def get_executor_config(self, level_id: str, price: Decimal, amount: Decimal):
        trade_type = self.get_trade_type_from_level_id(level_id)
        return PositionExecutorConfig(
            timestamp=self.market_data_provider.time(),
            level_id=level_id,
            connector_name=self.config.connector_name,
            trading_pair=self.config.trading_pair,
            entry_price=price,
            amount=amount,
            triple_barrier_config=self.config.triple_barrier_config,
            leverage=self.config.leverage,
            side=trade_type,
        )

    # ------------------------------------------------------------------
    # Refresh executors more aggressively when inventory is skewed
    # ------------------------------------------------------------------
    def executors_to_refresh(self) -> List[ExecutorAction]:
        refresh_time = self.config.executor_refresh_time

        # If inventory is significantly off-target, refresh faster so spreads re-center
        deviation = abs(self.processed_data.get("deviation", Decimal("0")))
        if deviation > Decimal("0.3"):
            refresh_time = max(30, int(refresh_time * 0.5))

        executors_to_refresh = self.filter_executors(
            executors=self.executors_info,
            filter_func=lambda x: (
                not x.is_trading
                and x.is_active
                and self.market_data_provider.time() - x.timestamp > refresh_time
            ),
        )
        return [
            StopExecutorAction(controller_id=self.config.id, executor_id=executor.id)
            for executor in executors_to_refresh
        ]

    # ------------------------------------------------------------------
    # Status display
    # ------------------------------------------------------------------
    def to_format_status(self) -> List[str]:
        lines = []
        if not self.processed_data:
            return ["Warming up..."]

        natr = self.processed_data.get("natr", Decimal("0"))
        inv_ratio = self.processed_data.get("inventory_ratio", Decimal("0"))
        deviation = self.processed_data.get("deviation", Decimal("0"))
        ref_price = self.processed_data.get("reference_price", Decimal("0"))
        buy_factor = self.processed_data.get("buy_spread_factor", Decimal("1"))
        sell_factor = self.processed_data.get("sell_spread_factor", Decimal("1"))

        lines.append(f"  NATR: {natr:.4%} | Inventory ratio: {inv_ratio:.2%} "
                     f"(target: {self.config.target_base_pct:.0%})")
        lines.append(f"  Deviation: {deviation:+.3f} | Ref price: {ref_price:.6f}")
        lines.append(f"  Spread factors — buy: {buy_factor:.3f}x  sell: {sell_factor:.3f}x")
        if self.config.max_spread is not None:
            lines.append(f"  Max spread (campaign): {self.config.max_spread:.2%}")

        active = self.filter_executors(self.executors_info, lambda x: x.is_active)
        trading = self.filter_executors(self.executors_info, lambda x: x.is_trading)
        lines.append(f"  Active executors: {len(active)} | Trading: {len(trading)}")

        return lines

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _compute_inventory_ratio(self, mid_price: Decimal) -> Decimal:
        """
        Compute the ratio of base asset value to total portfolio value.
        Returns a Decimal in [0, 1].  0 = all quote, 1 = all base.
        """
        base, quote = self.config.trading_pair.split("-")

        # Available balances from the connector
        connector = self.market_data_provider.get_connector(self.config.connector_name)
        base_bal = connector.get_available_balance(base)
        quote_bal = connector.get_available_balance(quote)

        # Include base locked in active sell executors and quote locked in active buy executors
        for executor in self.executors_info:
            if not executor.is_active:
                continue
            level_id = executor.custom_info.get("level_id", "")
            config = executor.config
            amount = getattr(config, "amount", None)
            entry_price = getattr(config, "entry_price", None)
            if amount is None or entry_price is None:
                continue
            if level_id.startswith("sell"):
                base_bal += amount
            elif level_id.startswith("buy"):
                quote_bal += amount * entry_price

        base_value = base_bal * mid_price
        total_value = base_value + quote_bal
        if total_value <= 0:
            return Decimal("0.5")
        return base_value / total_value
