"""
Volume Market Maker - V2 Strategy Script

Goal: Maximize trading volume while preserving capital (not losing the deposit).

How it works:
  1. Places buy and sell limit orders at multiple tight spread levels around the mid-price.
  2. Uses NATR (Normalized Average True Range) to dynamically widen spreads during volatile
     markets, protecting against adverse selection.
  3. Skews the reference price against accumulated inventory to naturally rebalance and avoid
     directional exposure.
  4. Caps maximum inventory to prevent excessive one-sided risk.
  5. Uses tight triple barriers (take-profit, stop-loss, time-limit) for quick position exits.

Configuration:
  Edit the VolumeMMConfig fields below or create a YAML config file and load via
  v2_with_controllers.py with controller_name "v2_volume_market_maker".

Usage:
  1. Set your connector_name and trading_pair
  2. Adjust total_amount_quote to your desired capital allocation
  3. Tune spreads, volatility_target, and inventory settings to your market
  4. Start the script in Hummingbot
"""
import os
from decimal import Decimal
from typing import Dict, List, Optional, Set

from pydantic import Field

from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy.strategy_v2_base import StrategyV2Base, StrategyV2ConfigBase
from hummingbot.strategy_v2.models.base import RunnableStatus
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, StopExecutorAction

from controllers.market_making.v2_volume_market_maker import VolumeMarketMakerConfig


class VolumeMMConfig(StrategyV2ConfigBase):
    script_file_name: str = os.path.basename(__file__)
    candles_config: List[CandlesConfig] = []
    markets: Dict[str, Set[str]] = {}

    # --- Main settings ---
    connector_name: str = Field(
        default="binance",
        json_schema_extra={"prompt": "Enter the connector name: ", "prompt_on_new": True}
    )
    trading_pair: str = Field(
        default="ETH-USDT",
        json_schema_extra={"prompt": "Enter the trading pair: ", "prompt_on_new": True}
    )
    total_amount_quote: Decimal = Field(
        default=Decimal("1000"),
        json_schema_extra={"prompt": "Enter total capital in quote asset: ", "prompt_on_new": True}
    )
    leverage: int = Field(
        default=1,
        json_schema_extra={"prompt": "Enter leverage (1 for spot): ", "prompt_on_new": True}
    )

    # --- Spread settings ---
    buy_spreads: str = Field(
        default="0.0005,0.001,0.002",
        json_schema_extra={
            "prompt": "Enter buy spreads (e.g., '0.0005,0.001,0.002'): ",
            "prompt_on_new": True}
    )
    sell_spreads: str = Field(
        default="0.0005,0.001,0.002",
        json_schema_extra={
            "prompt": "Enter sell spreads (e.g., '0.0005,0.001,0.002'): ",
            "prompt_on_new": True}
    )

    # --- Timing ---
    executor_refresh_time: int = Field(
        default=60,
        json_schema_extra={"prompt": "Enter order refresh time in seconds: ", "prompt_on_new": True}
    )
    cooldown_time: int = Field(
        default=5,
        json_schema_extra={"prompt": "Enter cooldown after fill in seconds: ", "prompt_on_new": True}
    )

    # --- Risk management ---
    stop_loss: Decimal = Field(
        default=Decimal("0.01"),
        json_schema_extra={"prompt": "Enter stop loss (e.g., 0.01 for 1%): ", "prompt_on_new": True}
    )
    take_profit: Decimal = Field(
        default=Decimal("0.005"),
        json_schema_extra={"prompt": "Enter take profit (e.g., 0.005 for 0.5%): ", "prompt_on_new": True}
    )
    time_limit: int = Field(
        default=900,
        json_schema_extra={"prompt": "Enter position time limit in seconds: ", "prompt_on_new": True}
    )

    # --- Volatility ---
    interval: str = Field(default="1m")
    natr_length: int = Field(default=14)
    volatility_target: float = Field(
        default=0.002,
        json_schema_extra={
            "prompt": "Enter target NATR (e.g., 0.002). Spreads widen above this: ",
            "prompt_on_new": True}
    )

    # --- Inventory management ---
    inventory_skew_factor: float = Field(
        default=0.5,
        json_schema_extra={"prompt": "Enter inventory skew factor (0-1): ", "prompt_on_new": True}
    )
    max_inventory_pct: float = Field(
        default=0.3,
        json_schema_extra={"prompt": "Enter max inventory as fraction of capital (e.g., 0.3): ", "prompt_on_new": True}
    )

    # --- Drawdown protection ---
    max_drawdown_quote: Optional[Decimal] = Field(
        default=Decimal("50"),
        json_schema_extra={"prompt": "Enter max drawdown in quote to stop trading (e.g., 50): ", "prompt_on_new": True}
    )


class VolumeMarketMakerScript(StrategyV2Base):
    """
    Volume Market Maker script.

    Creates a VolumeMarketMaker controller with the configured parameters
    and manages its lifecycle including drawdown protection.
    """
    CONTROLLER_ID = "volume_mm"

    @classmethod
    def init_markets(cls, config: VolumeMMConfig):
        cls.markets = {config.connector_name: {config.trading_pair}}

    def __init__(self, connectors: Dict[str, ConnectorBase], config: VolumeMMConfig):
        super().__init__(connectors, config)
        self.config = config
        self._max_pnl = Decimal("0")

        # Build controller config from script config
        controller_config = VolumeMarketMakerConfig(
            id=self.CONTROLLER_ID,
            connector_name=config.connector_name,
            trading_pair=config.trading_pair,
            total_amount_quote=config.total_amount_quote,
            leverage=config.leverage,
            buy_spreads=config.buy_spreads,
            sell_spreads=config.sell_spreads,
            executor_refresh_time=config.executor_refresh_time,
            cooldown_time=config.cooldown_time,
            stop_loss=config.stop_loss,
            take_profit=config.take_profit,
            time_limit=config.time_limit,
            interval=config.interval,
            natr_length=config.natr_length,
            volatility_target=config.volatility_target,
            inventory_skew_factor=config.inventory_skew_factor,
            max_inventory_pct=config.max_inventory_pct,
        )
        self.add_controller(controller_config)

    def on_tick(self):
        super().on_tick()
        if not self._is_stop_triggered:
            self._check_drawdown()

    def _check_drawdown(self):
        """Stop strategy if drawdown exceeds max allowed."""
        if not self.config.max_drawdown_quote:
            return

        report = self.get_performance_report(self.CONTROLLER_ID)
        if report is None:
            return

        current_pnl = report.global_pnl_quote
        if current_pnl > self._max_pnl:
            self._max_pnl = current_pnl

        drawdown = self._max_pnl - current_pnl
        if drawdown > self.config.max_drawdown_quote:
            self.logger().warning(
                f"Max drawdown reached ({drawdown:.2f} > {self.config.max_drawdown_quote}). "
                "Stopping strategy."
            )
            controller = self.controllers.get(self.CONTROLLER_ID)
            if controller and controller.status == RunnableStatus.RUNNING:
                controller.stop()
                # Stop non-trading executors immediately
                executors = self.get_executors_by_controller(self.CONTROLLER_ID)
                for executor in executors:
                    if executor.is_active and not executor.is_trading:
                        self.executor_orchestrator.execute_actions(
                            [StopExecutorAction(
                                controller_id=self.CONTROLLER_ID,
                                executor_id=executor.id
                            )]
                        )
            self._is_stop_triggered = True

    def create_actions_proposal(self) -> List[CreateExecutorAction]:
        return []

    def stop_actions_proposal(self) -> List[StopExecutorAction]:
        return []

    def apply_initial_setting(self):
        for controller_id, controller in self.controllers.items():
            config_dict = controller.config.model_dump()
            if "connector_name" in config_dict and self.is_perpetual(config_dict["connector_name"]):
                if "position_mode" in config_dict:
                    self.connectors[config_dict["connector_name"]].set_position_mode(
                        config_dict["position_mode"])
                if "leverage" in config_dict and "trading_pair" in config_dict:
                    self.connectors[config_dict["connector_name"]].set_leverage(
                        leverage=config_dict["leverage"],
                        trading_pair=config_dict["trading_pair"])
