import time
from decimal import Decimal
from typing import Dict, List, Optional, Set

import pandas as pd
from pydantic import Field

from hummingbot.client.ui.interface_utils import format_df_for_printout
from hummingbot.core.data_type.common import PriceType, TradeType
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy_v2.controllers.controller_base import (
    ControllerBase,
    ControllerConfigBase,
)
from hummingbot.strategy_v2.executors.data_types import ConnectorPair
from hummingbot.strategy_v2.executors.dca_executor.dca_executor import (
    DCAExecutorConfig,
    DCAMode,
)
from hummingbot.strategy_v2.models.executor_actions import (
    CreateExecutorAction,
    ExecutorAction,
)


class FunnyPatternConfig(ControllerConfigBase):
    """
    Simple candle-pattern controller that opens a single-leg DCA executor
    whenever the Funny-Pattern signal occurs.
    """

    controller_name: str = "funny_pattern"

    # -------------------- market / candle feed --------------------
    connector: str = Field(
        default="binance_perpetual",
        json_schema_extra={"prompt": "Connector to trade on: ", "prompt_on_new": True},
    )
    trading_pair: str = Field(
        default="BTC-USDT",
        json_schema_extra={
            "prompt": "Trading pair (exchange format, e.g. BTC-USDT): ",
            "prompt_on_new": True,
        },
    )
    candle_interval: str = Field(
        default="1m",
        json_schema_extra={
            "prompt": "Candle interval (e.g. 1m, 5m): ",
            "prompt_on_new": False,
        },
    )

    candles_config: List[CandlesConfig] = []

    # -------------------- risk & trade params --------------------
    order_amount: Decimal = Field(
        default=Decimal("0.01"),
        json_schema_extra={"prompt": "Order amount (quote): ", "prompt_on_new": True},
    )
    target_profitability: Decimal = Field(
        default=Decimal("0.005"),
        json_schema_extra={
            "prompt": "Take-profit % (as decimal, e.g. 0.005 = 0.5%): ",
            "prompt_on_new": True,
        },
    )
    stop_loss: Decimal = Field(
        default=Decimal("0.01"),
        json_schema_extra={
            "prompt": "Stop-loss % (as decimal, e.g. 0.01 = 1%): ",
            "prompt_on_new": True,
        },
    )
    time_limit: int = Field(
        default=1800,
        json_schema_extra={
            "prompt": "Maximum position lifetime in seconds (0 = disabled): ",
            "prompt_on_new": False,
        },
    )
    leverage: int = Field(
        default=1,
        json_schema_extra={
            "prompt": "Leverage (1 for spot): ",
            "prompt_on_new": False,
        },
    )

    # -----------------------------------------------------------------

    def __init__(self, **data):
        super().__init__(**data)
        # create default candle subscription automatically
        if not self.candles_config:
            self.candles_config = [
                CandlesConfig(
                    connector=self.connector,
                    trading_pair=self.trading_pair,
                    interval=self.candle_interval,
                    max_records=200,
                )
            ]

    # strategy-runner helper – adds market to HB
    def update_markets(self, markets: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
        markets.setdefault(self.connector, set()).add(self.trading_pair)
        return markets


class FunnyPattern(ControllerBase):
    """
    Detects Funny-Pattern on the specified candle feed and spawns a single-leg
    DCA executor (maker order at market price) in the indicated direction.
    """

    SIGNAL_KEY = "signal"  # key used inside processed_data dict

    def __init__(self, config: FunnyPatternConfig, *args, **kwargs):
        self.config: FunnyPatternConfig = config
        super().__init__(config, *args, **kwargs)

    # -----------------------------------------------------------------
    # Candle pattern logic
    # -----------------------------------------------------------------
    @staticmethod
    def _analyze_candles(prev: pd.Series, curr: pd.Series) -> Optional[str]:
        """
        Python translation of the provided TypeScript analyzeCandles().
        Returns 'long', 'short', or None.
        """
        is_prev_green = prev["close"] > prev["open"]
        is_curr_green = curr["close"] > curr["open"]

        if is_prev_green == is_curr_green:
            return None

        body_height = abs(curr["open"] - curr["close"])

        # potential LONG pattern
        if (
            prev["low"] > curr["low"]
            and prev["open"] < curr["close"]
            and is_curr_green
        ):
            wick_height = abs(curr["high"] - curr["close"])
            return "long" if wick_height < body_height / 3 else None

        # potential SHORT pattern
        if (
            prev["high"] < curr["high"]
            and prev["open"] > curr["close"]
            and not is_curr_green
        ):
            wick_height = abs(curr["low"] - curr["close"])
            return "short" if wick_height < body_height / 3 else None

        return None

    # -----------------------------------------------------------------
    # Controller life-cycle
    # -----------------------------------------------------------------
    async def update_processed_data(self):
        """
        Every tick we store the newest funny-pattern signal (if any)
        in self.processed_data[SIGNAL_KEY]
        """
        df = self.market_data_provider.get_candles_df(
            self.config.connector, self.config.trading_pair, self.config.candle_interval
        )

        if df is None or len(df) < 2:
            self.processed_data[self.SIGNAL_KEY] = None
            return

        # use the two most recent CLOSED candles
        prev_c, curr_c = df.iloc[-2], df.iloc[-1]
        signal = self._analyze_candles(prev_c, curr_c)
        self.processed_data[self.SIGNAL_KEY] = signal

    # -----------------------------------------------------------------
    # Trade execution logic
    # -----------------------------------------------------------------
    def _build_executor_config(self, side: TradeType, price: Decimal) -> DCAExecutorConfig:
        """
        Builds a single-level DCAExecutorConfig that represents one trade.
        """
        return DCAExecutorConfig(
            controller_id=self.config.id,
            timestamp=time.time(),
            connector_name=self.config.connector,
            trading_pair=self.config.trading_pair,
            mode=DCAMode.MAKER,
            leverage=self.config.leverage,
            side=side,
            amounts_quote=[self.config.order_amount],
            prices=[price],
            take_profit=self.config.target_profitability,
            stop_loss=self.config.stop_loss,
            time_limit=self.config.time_limit,
        )

    def determine_executor_actions(self) -> List[ExecutorAction]:
        """
        Creates an executor once a new signal is produced **and** there is no
        active executor on the same controller.
        """
        executor_actions: List[ExecutorAction] = []

        signal: Optional[str] = self.processed_data.get(self.SIGNAL_KEY)
        if signal is None:
            return executor_actions

        # already in position?
        active_execs = self.filter_executors(
            self.executors_info,
            lambda e: e.is_active and e.controller_id == self.config.id,
        )
        if active_execs:
            return executor_actions  # ignore new signals while in position

        trade_type = TradeType.BUY if signal == "long" else TradeType.SELL
        price = self.market_data_provider.get_price_by_type(
            self.config.connector, self.config.trading_pair, PriceType.MidPrice
        )
        if price is None:
            return executor_actions

        dca_config = self._build_executor_config(trade_type, Decimal(str(price)))
        executor_actions.append(
            CreateExecutorAction(executor_config=dca_config, controller_id=self.config.id)
        )
        self.logger().info(f"Funny-Pattern detected ({signal}). Opening new position.")
        return executor_actions

    # -----------------------------------------------------------------
    # CLI / status display
    # -----------------------------------------------------------------
    def to_format_status(self) -> List[str]:
        lines: List[str] = []

        sig = self.processed_data.get(self.SIGNAL_KEY)
        lines.append(f"Latest funny-pattern signal : {sig}")

        if self.executors_info:
            df = pd.DataFrame(e.custom_info for e in self.executors_info)
            lines.append(format_df_for_printout(df, table_format="psql"))
        else:
            lines.append("No active executors.")

        return lines