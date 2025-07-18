import time
from typing import Dict, List, Set

import pandas as pd
from pydantic import Field

from hummingbot.client.ui.interface_utils import format_df_for_printout
from hummingbot.core.data_type.common import TradeType
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
    candles_config: List[CandlesConfig] = []

    # -------------------- market / candle feed --------------------
    connector_name: str = Field(
        default=None,
        json_schema_extra={"prompt": "Connector to trade on: ", "prompt_on_new": True})
    trading_pair: str = Field(
        default=None,
        json_schema_extra={
            "prompt": "Trading pair (exchange format, e.g. BTC-USDT): ",
            "prompt_on_new": True})
    candles_connector: str = Field(
        default=None,
        json_schema_extra={
            "prompt": "Enter the connector for the candles data, leave empty to use the same exchange as the connector: ",
            "prompt_on_new": True})
    candles_trading_pair: str = Field(
        default=None,
        json_schema_extra={
            "prompt": "Enter the trading pair for the candles data, leave empty to use the same trading pair as the connector: ",
            "prompt_on_new": True})
    interval: str = Field(
        default="3m",
        json_schema_extra={
            "prompt": "Enter the candle interval (e.g., 1m, 5m, 1h, 1d): ",
            "prompt_on_new": True})

    # -----------------------------------------------------------------

    def __init__(self, **data):
        super().__init__(**data)
        # create default candle subscription automatically
        if not self.candles_config:
            self.candles_config = [
                CandlesConfig(
                    connector=self.connector_name,
                    trading_pair=self.trading_pair,
                    interval=self.interval,
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
            interval=self.config.interval,
            max_records=self.max_records,
        )

        if df is None or len(df) < 3:  # Need at least 3 candles (2 for analysis plus 1 current)
            self.processed_data["features"] = pd.DataFrame()
            self.processed_data["signal"] = 0
            return

        # Exclude the last candle as it's likely still forming
        analysis_df = df.iloc[:-1].copy()

        # Get the current candle timestamp and interval to determine if a new candle has closed
        current_time = self.market_data_provider.time() * 1000  # Convert to milliseconds
        last_closed_candle_timestamp = analysis_df.iloc[-1]["timestamp"]

        # Check if we've already analyzed this set of candles
        last_analyzed_timestamp = self.processed_data.get("last_analyzed_timestamp", 0)

        # Only analyze if a new candle has closed
        if last_closed_candle_timestamp > last_analyzed_timestamp:
            self.logger().info(f"Analyzing new closed candle at {last_closed_candle_timestamp}")

            signals = [0]
            for i in range(1, len(analysis_df)):
                prev_c = analysis_df.iloc[i - 1]
                curr_c = analysis_df.iloc[i]
                sig = self._analyze_candles(prev_c, curr_c)
                signals.append(sig if sig is not None else 0)

            analysis_df["signal"] = signals
            self.processed_data["features"] = analysis_df
            self.processed_data["signal"] = signals[-1]
            self.logger().info(f"Signal analysis complete. Signals: {signals}")
            self.processed_data["last_analyzed_timestamp"] = last_closed_candle_timestamp

            # Log when the pattern is detected
            if signals[-1] != 0:
                direction = "LONG" if signals[-1] > 0 else "SHORT"
                self.logger().info(f"Funny Pattern detected: {direction} signal at timestamp {last_closed_candle_timestamp}")
        else:
            self.logger().debug(f"No new closed candle to analyze. Last analyzed: {last_analyzed_timestamp}, Last closed: {last_closed_candle_timestamp}")

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

    def can_create_executor(self, signal: int) -> bool:
        """
        Check if an executor can be created based on the signal, the quantity of active executors and the cooldown time.
        Enhanced to respect cooldown_time for all executors, not just active ones.
        """
        active_executors_by_signal_side = self.filter_executors(
            executors=self.executors_info,
            filter_func=lambda x: x.is_active and (x.side == TradeType.BUY if signal > 0 else TradeType.SELL))

        # Check active executors limit
        active_executors_condition = len(active_executors_by_signal_side) < self.config.max_executors_per_side

        # Check cooldown against ALL executors of the same side (not just active ones)
        all_executors_by_signal_side = self.filter_executors(
            executors=self.executors_info,
            filter_func=lambda x: x.side == TradeType.BUY if signal > 0 else TradeType.SELL)

        max_timestamp = max([executor.timestamp for executor in all_executors_by_signal_side], default=0)
        cooldown_condition = self.market_data_provider.time() - max_timestamp > self.config.cooldown_time

        return active_executors_condition and cooldown_condition
