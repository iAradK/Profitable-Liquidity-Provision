"""Rolling out-of-sample pool selection: eligibility, scores from past data only, top-k portfolios with
entry/exit costs, HODL benchmarks, random portfolios and the information coefficient."""

from dataclasses import dataclass

import numpy as np

from experiments.config import CostConfig, ExperimentConfig, Horizon
from experiments.data import MarketData
from experiments.outcomes import Panel
from experiments.scores import History, ScoreFn
from experiments.stats import bootstrap_mean, newey_west, quiet, spearman

BENCHMARKS = ("hodl_5050", "hodl_eth")


@dataclass(frozen=True)
class SelectionRules:
    calibration_periods: int   # past periods a score may use; 0 = all of them (expanding window)
    min_history_periods: int
    min_tvl_usd: float
    first_decision_ts: int


def selection_rules(config: ExperimentConfig, horizon: Horizon, calibration_months: int) -> SelectionRules:
    return SelectionRules(
        calibration_periods=horizon.periods_for_months(calibration_months),
        min_history_periods=horizon.periods_for_months(config.eligibility.min_history_months),
        min_tvl_usd=config.eligibility.min_tvl_usd,
        first_decision_ts=config.sample.first_decision_ts,
    )


def decision_periods(panel: Panel, rules: SelectionRules) -> np.ndarray:
    return np.flatnonzero(panel.starts >= rules.first_decision_ts)


def past_periods(panel: Panel, t: int, calibration_periods: int) -> np.ndarray:
    """Periods that ended by the start of period t (the information available at the decision)."""
    past = np.flatnonzero(panel.ends <= panel.starts[t])
    return past[-calibration_periods:] if calibration_periods else past


def eligibility(panel: Panel, rules: SelectionRules) -> np.ndarray:
    """(n_pools, n_periods): enough history and enough TVL at the decision time.

    The pool must also have data for period t itself; this only drops cells with missing data,
    it never looks at the outcome.
    """
    eligible = np.zeros((panel.n_pools, panel.n_periods), dtype=bool)
    valid = panel.valid
    for t in decision_periods(panel, rules):
        n_history = valid[:, panel.ends <= panel.starts[t]].sum(axis=1)
        eligible[:, t] = (n_history >= rules.min_history_periods) & (panel.tvl[:, t] >= rules.min_tvl_usd) & valid[:, t]
    return eligible


def history_at(panel: Panel, t: int, calibration_periods: int) -> History:
    cols = past_periods(panel, t, calibration_periods)
    return History(panel.il[:, cols], panel.fee[:, cols], panel.variance[:, cols])


def score_matrix(panel: Panel, rules: SelectionRules, eligible: np.ndarray, score_fn: ScoreFn) -> np.ndarray:
    """Score of every eligible pool at every decision period; NaN elsewhere."""
    scores = np.full((panel.n_pools, panel.n_periods), np.nan)
    for t in decision_periods(panel, rules):
        if eligible[:, t].any():
            values = score_fn(history_at(panel, t, rules.calibration_periods))
            scores[:, t] = np.where(eligible[:, t], values, np.nan)
    return scores


@dataclass(frozen=True)
class Backtest:
    panel: Panel
    rules: SelectionRules
    eligible: np.ndarray
    scores: dict[str, np.ndarray]

    @property
    def decisions(self) -> np.ndarray:
        return decision_periods(self.panel, self.rules)


def prepare_backtest(panel: Panel, rules: SelectionRules, score_fns: dict[str, ScoreFn]) -> Backtest:
    eligible = eligibility(panel, rules)
    return Backtest(panel, rules, eligible, {name: score_matrix(panel, rules, eligible, fn)
                                             for name, fn in score_fns.items()})


# ---------------------------------------------------------------- costs and portfolio returns

@dataclass(frozen=True)
class CostSchedule:
    entry_gas_usd: np.ndarray  # per period start: gas of opening one position
    exit_gas_usd: np.ndarray
    capital_usd: float
    entry_swap_fraction: float


def cost_schedule(market: MarketData, starts: np.ndarray, costs: CostConfig) -> CostSchedule:
    if not costs.include_gas:
        zeros = np.zeros(len(starts))
        return CostSchedule(zeros, zeros, costs.capital_usd, costs.entry_swap_fraction)
    eth_usd = market.eth_usd.at(starts)
    entry = np.array([market.gas.cost_eth(int(ts), "entry") for ts in starts]) * eth_usd
    exit_ = np.array([market.gas.cost_eth(int(ts), "exit") for ts in starts]) * eth_usd
    return CostSchedule(entry, exit_, costs.capital_usd, costs.entry_swap_fraction)


@dataclass(frozen=True)
class StrategyResult:
    name: str
    decisions: np.ndarray       # period indices
    excess: np.ndarray          # return relative to holding the deposited tokens, after costs
    usd: np.ndarray             # USD return, after costs
    cost: np.ndarray            # costs as a share of capital
    holdings: tuple[np.ndarray, ...]


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    candidates = np.flatnonzero(np.isfinite(scores))
    return candidates[np.argsort(-scores[candidates], kind="stable")][:k]


def hold(name: str, panel: Panel, decisions: np.ndarray, holdings: tuple[np.ndarray, ...],
         schedule: CostSchedule) -> StrategyResult:
    """Equal weights over the held pools each period. Opening a position costs entry gas plus swapping
    `entry_swap_fraction` of it at the pool fee; closing one costs exit gas. Unchanged positions cost nothing."""
    fee_rate = np.array([p.fee_rate for p in panel.pools])
    excess, usd, cost = (np.full(len(decisions), np.nan) for _ in range(3))
    previous = np.array([], dtype=int)
    for j, (t, held) in enumerate(zip(decisions, holdings)):
        if len(held) == 0:
            previous = held
            continue
        new, exited = np.setdiff1d(held, previous), np.setdiff1d(previous, held)
        gas = len(new) * schedule.entry_gas_usd[t] + len(exited) * schedule.exit_gas_usd[t]
        cost[j] = gas / schedule.capital_usd + schedule.entry_swap_fraction * fee_rate[new].sum() / len(held)
        net = panel.net[held, t]
        excess[j] = net.mean() - cost[j]
        usd[j] = np.mean((1.0 + net) * (1.0 + panel.hodl_usd[held, t]) - 1.0) - cost[j]
        previous = held
    return StrategyResult(name, decisions, excess, usd, cost, holdings)


def benchmark(name: str, backtest: Backtest, eth_return: np.ndarray) -> StrategyResult:
    """HODL 50/50 of every eligible pool's tokens, or ETH only; no LP position, no costs."""
    decisions = backtest.decisions
    if name == "hodl_5050":
        usd = np.array([quiet(np.nanmean, backtest.panel.hodl_usd[backtest.eligible[:, t], t]) for t in decisions])
        excess = np.where(np.isnan(usd), np.nan, 0.0)
    elif name == "hodl_eth":
        usd, excess = eth_return[decisions], np.full(len(decisions), np.nan)
    else:
        raise ValueError(f"unknown benchmark {name!r}; known: {BENCHMARKS}")
    return StrategyResult(name, decisions, excess, usd, np.zeros(len(decisions)), ())


def run_strategies(backtest: Backtest, names: tuple[str, ...], k: int, schedule: CostSchedule,
                   eth_return: np.ndarray) -> dict[str, StrategyResult]:
    """Score-based strategies hold the top k pools; 'equal_weight' holds every eligible pool."""
    decisions = backtest.decisions
    results = {}
    for name in names:
        if name in BENCHMARKS:
            results[name] = benchmark(name, backtest, eth_return)
        elif name == "equal_weight":
            holdings = tuple(np.flatnonzero(backtest.eligible[:, t]) for t in decisions)
            results[name] = hold(name, backtest.panel, decisions, holdings, schedule)
        elif name in backtest.scores:
            holdings = tuple(top_k(backtest.scores[name][:, t], k) for t in decisions)
            results[name] = hold(name, backtest.panel, decisions, holdings, schedule)
        else:
            raise ValueError(f"unknown strategy {name!r}; scores: {sorted(backtest.scores)}")
    return results


def random_portfolios(backtest: Backtest, k: int, schedule: CostSchedule, n_draws: int,
                      rng: np.random.Generator) -> np.ndarray:
    """Mean excess return of `n_draws` strategies that hold k random eligible pools each period."""
    decisions = backtest.decisions
    pools_by_decision = [np.flatnonzero(backtest.eligible[:, t]) for t in decisions]
    means = np.empty(n_draws)
    for d in range(n_draws):
        holdings = tuple(rng.choice(pools, size=min(k, len(pools)), replace=False) for pools in pools_by_decision)
        means[d] = quiet(np.nanmean, hold("random", backtest.panel, decisions, holdings, schedule).excess)
    return means


def eth_returns(market: MarketData, panel: Panel) -> np.ndarray:
    return market.eth_usd.at(panel.ends) / market.eth_usd.at(panel.starts) - 1.0


# ---------------------------------------------------------------- information coefficient

def information_coefficients(scores: np.ndarray, outcome: np.ndarray, eligible: np.ndarray,
                             min_pools: int) -> np.ndarray:
    """Per period: Spearman correlation across eligible pools between the score and the realized outcome."""
    ic = np.full(scores.shape[1], np.nan)
    for t in range(scores.shape[1]):
        mask = eligible[:, t] & np.isfinite(scores[:, t]) & np.isfinite(outcome[:, t])
        if mask.sum() >= min_pools:
            ic[t] = spearman(scores[mask, t], outcome[mask, t])
    return ic


def family_information_coefficients(scores: np.ndarray, outcome: np.ndarray, eligible: np.ndarray,
                                    families: list[str], min_pools: int) -> np.ndarray:
    """Same, after averaging pools of one token pair (different fee tiers are not independent)."""
    labels = np.array(families)
    unique = sorted(set(families))
    family_scores = np.full((len(unique), scores.shape[1]), np.nan)
    family_outcome = np.full_like(family_scores, np.nan)
    for f, family in enumerate(unique):
        rows = labels == family
        mask = eligible[rows]
        family_scores[f] = quiet(np.nanmean, np.where(mask, scores[rows], np.nan), axis=0)
        family_outcome[f] = quiet(np.nanmean, np.where(mask, outcome[rows], np.nan), axis=0)
    return information_coefficients(family_scores, family_outcome, np.isfinite(family_scores), min_pools)


def summarize_ic(ic: np.ndarray, lags: int, block: int, n_samples: int, rng: np.random.Generator,
                 confidence: float) -> dict[str, float]:
    mean, t_stat = newey_west(ic, lags)
    boot = bootstrap_mean(ic, block, n_samples, rng, confidence)
    values = ic[np.isfinite(ic)]
    return {"ic_mean": mean, "ic_nw_t": t_stat, "ic_lo": boot["lo"], "ic_hi": boot["hi"], "n_periods": len(values),
            "share_positive": float(np.mean(values > 0)) if len(values) else np.nan}
