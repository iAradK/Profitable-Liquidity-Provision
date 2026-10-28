"""What every experiment receives: the config and the loaded market data (plus panels shared between experiments)."""

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from experiments.config import ExperimentConfig, Horizon
from experiments.data import MarketData
from experiments.outcomes import Panel, build_panel, sample_periods


@dataclass
class Context:
    config: ExperimentConfig
    market: MarketData

    def rng(self, salt: int = 0) -> np.random.Generator:
        return np.random.default_rng(self.config.seed + salt)

    @cached_property
    def monthly_panel(self) -> Panel:
        """Full-range outcomes over back-to-back calendar months of the sample."""
        return self.panel(Horizon(months=1))

    def panel(self, horizon: Horizon) -> Panel:
        starts, ends = sample_periods(self.config.sample, self.market.data_end_ts, horizon)
        return build_panel(self.market, starts, ends, horizon)

    def block_periods(self, horizon: Horizon) -> int:
        """Bootstrap block length in periods of `horizon`, from the configured block length in months."""
        return horizon.periods_for_months(self.config.stats.block_months)
