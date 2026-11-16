"""Load the [experiments] settings (and the [universe] pools) from config.toml into typed, frozen dataclasses."""

import re
import tomllib
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.toml"

EXPERIMENT_IDS = ("E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9", "E10", "E11", "E12", "E13", "E14")
CREDITING_RULES = ("fixed_volume", "proportional")
MONTH_DAYS = 30.4375


class ConfigError(Exception):
    pass


@dataclass(frozen=True)
class Horizon:
    """A holding period of whole calendar months or of days."""
    months: int = 0
    days: int = 0

    @property
    def approx_days(self) -> float:
        return self.months * MONTH_DAYS if self.months else float(self.days)

    @property
    def periods_per_year(self) -> float:
        return 365.25 / self.approx_days

    def periods_for_months(self, months: int) -> int:
        """Number of consecutive periods covering `months` months (0 stays 0 = expanding)."""
        if months == 0:
            return 0
        return max(1, round(months * MONTH_DAYS / self.approx_days))

    def __str__(self) -> str:
        return f"{self.months}m" if self.months else f"{self.days}d"


def parse_horizon(text: str) -> Horizon:
    match = re.fullmatch(r"(\d+)([md])", text.strip())
    if not match or int(match.group(1)) == 0:
        raise ConfigError(f"horizon must look like '1m' or '14d', got {text!r}")
    n = int(match.group(1))
    return Horizon(months=n) if match.group(2) == "m" else Horizon(days=n)


@dataclass(frozen=True)
class Pool:
    address: str
    label: str
    category: str


@dataclass(frozen=True)
class EventWindow:
    name: str
    start_ts: int
    end_ts: int


@dataclass(frozen=True)
class DataConfig:
    data_dir: Path
    gas_file: Path | None
    swaps_file: Path | None
    replay_swaps_files: tuple[Path, ...]  # Swap logs of the months replayed swap by swap (E12)
    eth_token: str


@dataclass(frozen=True)
class SampleConfig:
    start_ts: int
    end_ts: int
    first_decision_ts: int
    regime_split_ts: int
    event_windows: tuple[EventWindow, ...]


@dataclass(frozen=True)
class EligibilityConfig:
    min_history_months: int
    min_tvl_usd: float


@dataclass(frozen=True)
class CostConfig:
    capital_usd: float
    entry_swap_fraction: float
    include_gas: bool


@dataclass(frozen=True)
class FeeCrediting:
    """How a hypothetical position of position_usd is credited with the pool's fees (Section III-A)."""
    rule: str            # "fixed_volume" (L_act / (L_act + L) of L * dG) or "proportional" (L * dG)
    position_usd: float


@dataclass(frozen=True)
class FeeConfig:
    main: FeeCrediting                          # used by every experiment
    sensitivity_rules: tuple[str, ...]          # E12 grid: rules x position sizes
    sensitivity_position_usd: tuple[float, ...]
    exclude_periods: tuple[str, ...]            # "YYYY-MM" months E5, E8 and E12 also report without
    replay_implied_ratio: float                 # thin-liquidity months: implied fees / fees paid above this ...
    replay_min_fee_yield: float                 # ... and a fee yield above this are computed swap by swap


@dataclass(frozen=True)
class StatsConfig:
    bootstrap_samples: int
    block_months: int
    block_days: int
    newey_west_lags: int
    confidence: float
    min_pools_for_ic: int


@dataclass(frozen=True)
class SelectionConfig:
    tail_quantile: float
    horizon: Horizon
    calibration_months: int
    top_k: tuple[int, ...]
    strategies: tuple[str, ...]
    random_draws: int
    risk_aversion: tuple[float, ...]  # lambda grid of fee - sigma^2/8 - lambda * tail (strategies lvr_tail_<lambda>)


@dataclass(frozen=True)
class SensitivityConfig:
    tail_quantiles: tuple[float, ...]
    horizons: tuple[Horizon, ...]
    calibration_months: tuple[int, ...]
    top_k: tuple[int, ...]


@dataclass(frozen=True)
class PersistenceConfig:
    window_days: int
    step_days: int
    phi_max: float
    variance_ratio_lags: tuple[int, ...]
    test_alpha: float


@dataclass(frozen=True)
class CalibrationConfig:
    min_train_months: int
    rolling_months: int


@dataclass(frozen=True)
class HorizonMatchingConfig:
    fixed_months: tuple[int, ...]
    max_months: int
    window_days: int
    top_k: int


@dataclass(frozen=True)
class ConcentratedConfig:
    half_widths: tuple[float, ...]
    sigma_multiples: tuple[float, ...]
    validation_pools: tuple[str, ...]
    validation_start_ts: int
    validation_end_ts: int
    plot_risk_aversion: tuple[float, ...]  # lambdas shown in the IC-by-range figure (all are in the table)


@dataclass(frozen=True)
class PortfolioConfig:
    scenarios: int  # simulated months per decision for the joint IL tail of a portfolio (E13)


@dataclass(frozen=True)
class RhoScoreConfig:
    min_training_months: int  # past monthly regressions needed before the out-of-sample score is used (E14)


@dataclass(frozen=True)
class FeeValidationConfig:
    min_interval_hours: float
    tolerance: float


@dataclass(frozen=True)
class ReplicationConfig:
    data_dir: Path
    gas_file: Path | None
    eth_token: str


@dataclass(frozen=True)
class ExperimentConfig:
    run: tuple[str, ...]
    output_dir: Path
    force_refresh: bool
    max_workers: int
    seed: int
    pools: tuple[Pool, ...]
    data: DataConfig
    sample: SampleConfig
    eligibility: EligibilityConfig
    costs: CostConfig
    fees: FeeConfig
    stats: StatsConfig
    selection: SelectionConfig
    sensitivity: SensitivityConfig
    persistence: PersistenceConfig
    calibration: CalibrationConfig
    horizon_matching: HorizonMatchingConfig
    concentrated: ConcentratedConfig
    portfolio: PortfolioConfig
    rho_score: RhoScoreConfig
    fee_validation: FeeValidationConfig
    replication: ReplicationConfig

    @property
    def cache_dir(self) -> Path:
        return self.output_dir / "cache"


def timestamp(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp())


def _optional_path(text: str) -> Path | None:
    return ROOT / text if text else None


def _parse(raw: dict) -> ExperimentConfig:
    exp = raw["experiments"]
    data, sample, elig, costs, stats = (exp[k] for k in ("data", "sample", "eligibility", "costs", "statistics"))
    fees = exp["fees"]
    sel, sens, pers, cal = (exp[k] for k in ("selection", "sensitivity", "persistence", "calibration"))
    hm, conc, fv, rep = (exp[k] for k in ("horizon_matching", "concentrated", "fee_validation", "replication"))
    return ExperimentConfig(
        run=tuple(exp["run"]),
        output_dir=ROOT / exp["output_dir"],
        force_refresh=exp["force_refresh"],
        max_workers=exp["max_workers"],
        seed=exp["seed"],
        pools=tuple(Pool(p["address"].lower(), p["label"], p["category"]) for p in raw["universe"]["pools"]),
        data=DataConfig(ROOT / data["data_dir"], _optional_path(data["gas_file"]),
                        _optional_path(data["swaps_file"]),
                        tuple(ROOT / f for f in data["replay_swaps_files"]), data["eth_token"].lower()),
        sample=SampleConfig(
            start_ts=timestamp(sample["start"]),
            end_ts=timestamp(sample["end"]),
            first_decision_ts=timestamp(sample["first_decision"]),
            regime_split_ts=timestamp(sample["regime_split"]),
            event_windows=tuple(EventWindow(w["name"], timestamp(w["start"]), timestamp(w["end"]))
                                for w in sample["event_windows"]),
        ),
        eligibility=EligibilityConfig(elig["min_history_months"], float(elig["min_tvl_usd"])),
        costs=CostConfig(float(costs["capital_usd"]), costs["entry_swap_fraction"], costs["include_gas"]),
        fees=FeeConfig(FeeCrediting(fees["crediting"], float(fees["position_usd"])),
                       tuple(fees["sensitivity_rules"]),
                       tuple(float(x) for x in fees["sensitivity_position_usd"]),
                       tuple(fees["exclude_periods"]), float(fees["replay_implied_ratio"]),
                       float(fees["replay_min_fee_yield"])),
        stats=StatsConfig(stats["bootstrap_samples"], stats["block_months"], stats["block_days"],
                          stats["newey_west_lags"], stats["confidence"], stats["min_pools_for_ic"]),
        selection=SelectionConfig(sel["tail_quantile"], parse_horizon(sel["horizon"]), sel["calibration_months"],
                                  tuple(sel["top_k"]), tuple(sel["strategies"]), sel["random_draws"],
                                  tuple(float(lam) for lam in sel["risk_aversion"])),
        sensitivity=SensitivityConfig(tuple(sens["tail_quantiles"]), tuple(parse_horizon(h) for h in sens["horizons"]),
                                      tuple(sens["calibration_months"]), tuple(sens["top_k"])),
        persistence=PersistenceConfig(pers["window_days"], pers["step_days"], pers["phi_max"],
                                      tuple(pers["variance_ratio_lags"]), pers["test_alpha"]),
        calibration=CalibrationConfig(cal["min_train_months"], cal["rolling_months"]),
        horizon_matching=HorizonMatchingConfig(tuple(hm["fixed_months"]), hm["max_months"], hm["window_days"],
                                               hm["top_k"]),
        concentrated=ConcentratedConfig(
            half_widths=tuple(conc["half_widths"]),
            sigma_multiples=tuple(conc["sigma_multiples"]),
            validation_pools=tuple(a.lower() for a in conc["validation_pools"]),
            validation_start_ts=timestamp(conc["validation_start"]),
            validation_end_ts=timestamp(conc["validation_end"]),
            plot_risk_aversion=tuple(float(lam) for lam in conc["plot_risk_aversion"]),
        ),
        portfolio=PortfolioConfig(exp["portfolio"]["scenarios"]),
        rho_score=RhoScoreConfig(exp["rho_score"]["min_training_months"]),
        fee_validation=FeeValidationConfig(fv["min_interval_hours"], fv["tolerance"]),
        replication=ReplicationConfig(ROOT / rep["data_dir"], _optional_path(rep["gas_file"]),
                                      rep["eth_token"].lower()),
    )


def _validate(config: ExperimentConfig) -> None:
    unknown = [e for e in config.run if e not in EXPERIMENT_IDS]
    if unknown:
        raise ConfigError(f"unknown experiments in [experiments].run: {unknown}; valid: {EXPERIMENT_IDS}")
    s = config.sample
    if not s.start_ts < s.first_decision_ts < s.end_ts:
        raise ConfigError("[experiments.sample] needs start < first_decision < end")
    if "net_tail" not in config.selection.strategies:
        raise ConfigError("[experiments.selection].strategies must include 'net_tail' (Net90)")
    rules = (config.fees.main.rule, *config.fees.sensitivity_rules)
    if any(rule not in CREDITING_RULES for rule in rules):
        raise ConfigError(f"[experiments.fees] crediting rules must be in {CREDITING_RULES}, got {rules}")
    if min(config.fees.main.position_usd, *config.fees.sensitivity_position_usd) <= 0:
        raise ConfigError("[experiments.fees] position sizes must be > 0")
    if config.rho_score.min_training_months < 1:
        raise ConfigError("[experiments.rho_score] min_training_months must be >= 1")
    if config.portfolio.scenarios < 100:
        raise ConfigError("[experiments.portfolio] scenarios must be >= 100")
    if any(lam < 0 for lam in config.selection.risk_aversion):
        raise ConfigError("[experiments.selection] risk_aversion values must be >= 0")
    quantiles = (config.selection.tail_quantile, *config.sensitivity.tail_quantiles)
    if not all(0 < q < 0.5 for q in quantiles):
        raise ConfigError("tail quantiles must be in (0, 0.5)")
    if any(w <= 0 or w >= 1 for w in config.concentrated.half_widths):
        raise ConfigError("[experiments.concentrated] half_widths must be in (0, 1)")
    if config.max_workers < 1 or config.stats.bootstrap_samples < 100:
        raise ConfigError("max_workers must be >= 1 and bootstrap_samples >= 100")
    if not config.data.data_dir.is_dir():
        raise ConfigError(f"data_dir not found: {config.data.data_dir}")
    for path in (config.data.gas_file, config.data.swaps_file, *config.data.replay_swaps_files):
        if path is not None and not path.exists():
            raise ConfigError(f"file not found: {path} (set it to \"\" if the data is not available)")


def load_config(path: Path = CONFIG_PATH) -> ExperimentConfig:
    try:
        with path.open("rb") as f:
            raw = tomllib.load(f)
        config = _parse(raw)
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {exc.filename}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path.name} is not valid TOML: {exc}") from exc
    except KeyError as exc:
        raise ConfigError(f"{path.name} is missing the setting {exc}") from exc
    _validate(config)
    return config
