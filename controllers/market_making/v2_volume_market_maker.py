from decimal import Decimal
from typing import List, Optional

import pandas_ta as ta  # noqa: F401
from pydantic import Field, field_validator
from pydantic_core.core_schema import ValidationInfo

from hummingbot.core.data_type.common import PriceType, TradeType
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy_v2.controllers.market_making_controller_base import (
    MarketMakingControllerBase,
    MarketMakingControllerConfigBase,
)
from hummingbot.strategy_v2.executors.position_executor.data_types import PositionExecutorConfig
from hummingbot.strategy_v2.models.executor_actions import ExecutorAction, StopExecutorAction


class VolumeMarketMakerConfig(MarketMakingControllerConfigBase):
    """
    Volume Market Maker: Maximizes trading volume while preserving capital.

    Uses tight spreads with NATR-based volatility adjustment and inventory
    skewing to generate maximum volume with minimal directional risk.

    Key mechanisms:
    - Tight base spreads for frequent fills (high volume)
    - NATR volatility guard: widens spreads in volatile markets (capital preservation)
    - Inventory skewing: shifts mid-price against accumulated inventory to rebalance
    - Max inventory cap: stops adding to the overweight side
    - Tight triple barriers: quick take-profit, stop-loss, time limit
    """
    controller_name: str = "v2_volume_market_maker"
    controller_type: str = "market_making"

    # Tight spreads for maximum fill rate (3 levels per side)
    buy_spreads: List[float] = Field(
        default="0.0005,0.001,0.002",
        json_schema_extra={
            "prompt": "Enter comma-separated buy spreads (e.g., '0.0005,0.001,0.002' for 0.05%,0.1%,0.2%): ",
            "prompt_on_new": True, "is_updatable": True}
    )
    sell_spreads: List[float] = Field(
        default="0.0005,0.001,0.002",
        json_schema_extra={
            "prompt": "Enter comma-separated sell spreads (e.g., '0.0005,0.001,0.002' for 0.05%,0.1%,0.2%): ",
            "prompt_on_new": True, "is_updatable": True}
    )

    # Fast refresh to keep orders fresh and avoid stale quotes
    executor_refresh_time: int = Field(
        default=60,
        json_schema_extra={
            "prompt": "Enter order refresh time in seconds (e.g., 60): ",
            "prompt_on_new": True, "is_updatable": True}
    )
    cooldown_time: int = Field(
        default=5,
        json_schema_extra={
            "prompt": "Enter cooldown time in seconds after a fill (e.g., 5): ",
            "prompt_on_new": True, "is_updatable": True}
    )

    # Tight risk management for capital preservation
    stop_loss: Optional[Decimal] = Field(
        default=Decimal("0.01"),
        json_schema_extra={
            "prompt": "Enter stop loss (e.g., 0.01 for 1%): ",
            "prompt_on_new": True, "is_updatable": True}
    )
    take_profit: Optional[Decimal] = Field(
        default=Decimal("0.005"),
        json_schema_extra={
            "prompt": "Enter take profit (e.g., 0.005 for 0.5%): ",
            "prompt_on_new": True, "is_updatable": True}
    )
    time_limit: Optional[int] = Field(
        default=900,
        json_schema_extra={
            "prompt": "Enter time limit in seconds (e.g., 900 for 15 min): ",
            "prompt_on_new": True, "is_updatable": True}
    )

    # --- Volatility settings ---
    candles_connector: Optional[str] = Field(
        default=None,
        json_schema_extra={
            "prompt": "Enter candles connector (leave empty to use same as trading connector): ",
            "prompt_on_new": True}
    )
    candles_trading_pair: Optional[str] = Field(
        default=None,
        json_schema_extra={
            "prompt": "Enter candles trading pair (leave empty to use same as trading pair): ",
            "prompt_on_new": True}
    )
    interval: str = Field(
        default="3m",
        json_schema_extra={
            "prompt": "Enter candle interval (e.g., 1m, 3m, 5m): ",
            "prompt_on_new": True}
    )
    natr_length: int = Field(
        default=14,
        json_schema_extra={
            "prompt": "Enter NATR length for volatility calculation (e.g., 14): ",
            "prompt_on_new": True}
    )
    volatility_target: float = Field(
        default=0.002,
        json_schema_extra={
            "prompt": "Enter target NATR level (e.g., 0.002 for 0.2%). Spreads widen above this: ",
            "prompt_on_new": True, "is_updatable": True}
    )
    min_spread_factor: float = Field(
        default=1.0,
        json_schema_extra={
            "prompt": "Enter minimum spread multiplier floor (e.g., 1.0): ",
            "prompt_on_new": True, "is_updatable": True}
    )
    max_spread_factor: float = Field(
        default=5.0,
        json_schema_extra={
            "prompt": "Enter maximum spread multiplier cap (e.g., 5.0): ",
            "prompt_on_new": True, "is_updatable": True}
    )

    # --- Inventory management ---
    inventory_skew_factor: float = Field(
        default=0.5,
        json_schema_extra={
            "prompt": "Enter inventory skew factor (0=no skew, 1=aggressive, e.g., 0.5): ",
            "prompt_on_new": True, "is_updatable": True}
    )
    max_inventory_pct: float = Field(
        default=0.3,
        json_schema_extra={
            "prompt": "Enter max inventory as fraction of capital (e.g., 0.3 for 30%): ",
            "prompt_on_new": True, "is_updatable": True}
    )

    @field_validator("candles_connector", mode="before")
    @classmethod
    def set_candles_connector(cls, v, validation_info: ValidationInfo):
        if v is None or v == "":
            return validation_info.data.get("connector_name")
        return v

    @field_validator("candles_trading_pair", mode="before")
    @classmethod
    def set_candles_trading_pair(cls, v, validation_info: ValidationInfo):
        if v is None or v == "":
            return validation_info.data.get("trading_pair")
        return v


class VolumeMarketMaker(MarketMakingControllerBase):
    """
    Volume Market Maker controller.

    Maximizes trading volume while preserving capital through:
    1. NATR-based dynamic spread adjustment (widens in volatility)
    2. Inventory skewing (shifts price against accumulated position)
    3. Max inventory limits (stops adding to overweight side)
    4. Tight triple barriers (quick TP/SL/time exits)
    """

    def __init__(self, config: VolumeMarketMakerConfig, *args, **kwargs):
        self.config = config
        self.max_records = config.natr_length + 50
        if len(self.config.candles_config) == 0:
            self.config.candles_config = [CandlesConfig(
                connector=config.candles_connector,
                trading_pair=config.candles_trading_pair,
                interval=config.interval,
                max_records=self.max_records
            )]
        self._current_natr = Decimal("0")
        self._inventory_ratio = Decimal("0")
        self._spread_multiplier = Decimal("1")
        super().__init__(config, *args, **kwargs)

    async def update_processed_data(self):
        """
        Calculate dynamic reference price and spread multiplier.

        1. NATR -> spread_multiplier: spreads widen when volatility > target
        2. Inventory ratio -> reference_price shift: price moves against inventory
        """
        mid_price = self.market_data_provider.get_price_by_type(
            self.config.connector_name,
            self.config.trading_pair,
            PriceType.MidPrice
        )

        # --- Volatility-based spread adjustment ---
        spread_multiplier = Decimal(str(self.config.min_spread_factor))
        try:
            candles = self.market_data_provider.get_candles_df(
                connector_name=self.config.candles_connector,
                trading_pair=self.config.candles_trading_pair,
                interval=self.config.interval,
                max_records=self.max_records
            )
            if len(candles) >= self.config.natr_length:
                natr = ta.natr(
                    candles["high"], candles["low"], candles["close"],
                    length=self.config.natr_length
                )
                current_natr = natr.iloc[-1] / 100  # Convert percentage to decimal
                self._current_natr = Decimal(str(current_natr))

                # Spread multiplier: ratio of current volatility to target
                # When vol > target: spreads widen (protection)
                # When vol < target: spreads stay at floor (max volume)
                if self.config.volatility_target > 0:
                    raw_multiplier = float(current_natr) / self.config.volatility_target
                    spread_multiplier = Decimal(str(max(
                        self.config.min_spread_factor,
                        min(self.config.max_spread_factor, raw_multiplier)
                    )))
        except Exception:
            pass  # Use default if candles not ready

        self._spread_multiplier = spread_multiplier

        # --- Inventory skewing ---
        reference_price = Decimal(str(mid_price))
        inventory_ratio = self._calculate_inventory_ratio()
        self._inventory_ratio = inventory_ratio

        if self.config.inventory_skew_factor > 0 and self._current_natr > 0:
            # Shift price against inventory direction to encourage mean reversion:
            # Long inventory -> shift price down -> sells become more attractive
            # Short inventory -> shift price up -> buys become more attractive
            price_shift = (
                -inventory_ratio
                * Decimal(str(self.config.inventory_skew_factor))
                * self._current_natr
            )
            reference_price = reference_price * (1 + price_shift)

        self.processed_data = {
            "reference_price": reference_price,
            "spread_multiplier": spread_multiplier,
        }

    def _calculate_inventory_ratio(self) -> Decimal:
        """
        Calculate net inventory as a fraction of total capital.
        Returns value in [-1, 1]. Positive = net long, negative = net short.
        """
        if self.config.total_amount_quote <= 0:
            return Decimal("0")

        # Use positions_held for aggregate position (most accurate)
        if self.positions_held:
            net_value = Decimal("0")
            for position in self.positions_held:
                if (position.trading_pair == self.config.trading_pair and
                        position.connector_name == self.config.connector_name):
                    if position.side == TradeType.BUY:
                        net_value += position.amount * position.breakeven_price
                    else:
                        net_value -= position.amount * position.breakeven_price
            ratio = net_value / self.config.total_amount_quote
            return max(Decimal("-1"), min(Decimal("1"), ratio))

        # Fallback: calculate from active trading executors
        net_position_quote = Decimal("0")
        for executor in self.executors_info:
            if executor.is_trading and executor.custom_info:
                side = executor.custom_info.get("side")
                if side == TradeType.BUY or side == "BUY":
                    net_position_quote += executor.filled_amount_quote
                elif side == TradeType.SELL or side == "SELL":
                    net_position_quote -= executor.filled_amount_quote
        ratio = net_position_quote / self.config.total_amount_quote
        return max(Decimal("-1"), min(Decimal("1"), ratio))

    def get_executor_config(self, level_id: str, price: Decimal, amount: Decimal):
        """
        Create a PositionExecutor with tight triple barrier for quick exit.
        Returns None if adding to this side would exceed max inventory.
        """
        trade_type = self.get_trade_type_from_level_id(level_id)

        # Inventory cap: don't add to the overweight side
        if self.config.max_inventory_pct > 0:
            if (trade_type == TradeType.BUY and
                    float(self._inventory_ratio) >= self.config.max_inventory_pct):
                return None
            if (trade_type == TradeType.SELL and
                    float(self._inventory_ratio) <= -self.config.max_inventory_pct):
                return None

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

    def executors_to_early_stop(self) -> List[ExecutorAction]:
        """
        Emergency stop: if inventory is way past the max limit,
        close trading executors on the heavy side to reduce exposure.
        """
        actions = []
        emergency_threshold = self.config.max_inventory_pct * 1.5
        if emergency_threshold <= 0:
            return actions

        if abs(float(self._inventory_ratio)) > emergency_threshold:
            for executor in self.executors_info:
                if executor.is_trading and executor.custom_info:
                    side = executor.custom_info.get("side")
                    # If net long, close buy-side executors
                    if (self._inventory_ratio > 0 and
                            (side == TradeType.BUY or side == "BUY")):
                        actions.append(StopExecutorAction(
                            controller_id=self.config.id,
                            executor_id=executor.id
                        ))
                    # If net short, close sell-side executors
                    elif (self._inventory_ratio < 0 and
                          (side == TradeType.SELL or side == "SELL")):
                        actions.append(StopExecutorAction(
                            controller_id=self.config.id,
                            executor_id=executor.id
                        ))
        return actions

    def to_format_status(self) -> List[str]:
        lines = []
        lines.append(f"  NATR: {float(self._current_natr) * 100:.4f}%")
        lines.append(f"  Spread Multiplier: {float(self._spread_multiplier):.2f}x")
        lines.append(f"  Inventory Ratio: {float(self._inventory_ratio) * 100:.2f}%")

        active_buys = 0
        active_sells = 0
        trading = 0
        for e in self.executors_info:
            if e.is_active and e.custom_info:
                side = e.custom_info.get("side")
                if side == TradeType.BUY or side == "BUY":
                    active_buys += 1
                elif side == TradeType.SELL or side == "SELL":
                    active_sells += 1
            if e.is_trading:
                trading += 1

        lines.append(f"  Active Orders: {active_buys} buys / {active_sells} sells")
        lines.append(f"  Trading: {trading}")

        if self.performance_report:
            lines.append(
                f"  PnL: {self.performance_report.global_pnl_quote:.4f} quote "
                f"({self.performance_report.global_pnl_pct:.2f}%)"
            )
            lines.append(f"  Volume Traded: {self.performance_report.volume_traded:.2f} quote")

        return lines
