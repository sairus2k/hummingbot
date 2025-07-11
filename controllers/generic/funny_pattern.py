import time
from decimal import Decimal
from typing import Dict, List, Set

import pandas as pd
from pydantic import Field

from hummingbot.client.ui.interface_utils import format_df_for_printout
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig

from hummingbot.strategy_v2.controllers.directional_trading_controller_base import (
    DirectionalTradingControllerBase,
    DirectionalTradingControllerConfigBase,
)

class FunnyPatternConfig(DirectionalTradingControllerConfigBase):
    """
    Simple candle-pattern controller that opens a single-leg DCA executor
    whenever the Funny-Pattern signal occurs.
    """

    id: str = Field(default_factory=lambda: "funny_pattern_" + str(int(time.time())))
    controller_type: str = "generic"
    controller_name: str = "funny_pattern"

    # -------------------- market / candle feed --------------------
    connector_name: str = Field(
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
    take_profit: Decimal = Field(
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
                    connector=self.connector_name,
                    trading_pair=self.trading_pair,
                    interval=self.candle_interval,
                    max_records=200,
                )
            ]

    # strategy-runner helper – adds market to HB
    def update_markets(self, markets: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
        markets.setdefault(self.connector_name, set()).add(self.trading_pair)
        return markets


class FunnyPattern(DirectionalTradingControllerBase):
    """
    Detects Funny-Pattern on the specified candle feed and spawns a single-leg
    DCA executor (maker order at market price) in the indicated direction.
    """

    def __init__(self, config: FunnyPatternConfig, *args, **kwargs):
        self.config = config
        self.max_records = 100
        if len(self.config.candles_config) == 0:
            self.config.candles_config = [CandlesConfig(
                connector=config.candles_connector,
                trading_pair=config.candles_trading_pair,
                interval=config.interval,
                max_records=self.max_records
            )]
        super().__init__(config, *args, **kwargs)

    # -----------------------------------------------------------------
    # Candle pattern logic
    # -----------------------------------------------------------------
    @staticmethod
    def _analyze_candles(prev: pd.Series, curr: pd.Series) -> int:
        """
        Python translation of the provided TypeScript analyzeCandles().
        Returns 'long', 'short', or None.
        """
        is_prev_green = prev["close"] > prev["open"]
        is_curr_green = curr["close"] > curr["open"]

        if is_prev_green == is_curr_green:
            return 0

        body_height = abs(curr["open"] - curr["close"])

        # potential LONG pattern
        if (
            prev["low"] > curr["low"]
            and prev["open"] < curr["close"]
            and is_curr_green
        ):
            wick_height = abs(curr["high"] - curr["close"])
            return 1 if wick_height < body_height / 3 else 0 # long

        # potential SHORT pattern
        if (
            prev["high"] < curr["high"]
            and prev["open"] > curr["close"]
            and not is_curr_green
        ):
            wick_height = abs(curr["low"] - curr["close"])
            return -1 if wick_height < body_height / 3 else 0 # short

        return 0

    # -----------------------------------------------------------------
    # Controller life-cycle
    # -----------------------------------------------------------------
    async def update_processed_data(self):
        df = self.market_data_provider.get_candles_df(
            connector_name=self.config.connector_name,
            trading_pair=self.config.trading_pair,
            interval=self.config.candle_interval,
            max_records=self.max_records,
        )

        if df is None or len(df) < 2:
            self.processed_data["features"] = pd.DataFrame()
            self.processed_data["signal"] = 0
            return

        signals = [0]
        for i in range(1, len(df)):
            prev_c = df.iloc[i - 1]
            curr_c = df.iloc[i]
            sig = self._analyze_candles(prev_c, curr_c)
            print(f"Signal={sig}")
            signals.append(sig if sig is not None else 0)
        df["signal"] = signals
        self.processed_data["features"] = df
        # Optionally, set the latest signal
        self.processed_data["signal"] = signals[-1]

    # -----------------------------------------------------------------
    # CLI / status display
    # -----------------------------------------------------------------
    def to_format_status(self) -> List[str]:
        lines: List[str] = []

        sig = self.processed_data["signal"]
        lines.append(f"Latest funny-pattern signal : {sig}")

        if self.executors_info:
            df = pd.DataFrame(e.custom_info for e in self.executors_info)
            lines.append(format_df_for_printout(df, table_format="psql"))
        else:
            lines.append("No active executors.")

        return lines