# Risk-Aware Pool Screening for Uniswap v3 Liquidity Provision

Data pipeline and experiments for the paper **"Mitigating DeFi Impermanent Loss through Risk-Aware Pool Screening"** (submitted to IEEE ICBC 2027).

The paper asks how a liquidity provider (LP) should choose among AMM pools when fee income competes with impermanent loss (IL). It scores each pool with

$$S_\lambda = \overline{y^{\text{fee}}} - \overline{\sigma^2}/8 + \lambda \cdot \mathrm{IL}_q,$$

namely the mean past fee yield, minus the expected divergence loss (the LVR rate), plus a weighted model-based tail of the IL. It then evaluates the pools this score selects out of sample, on 27 Uniswap v3 pools on Ethereum from May 2021 to October 2026, including gas and swap costs.

This repository contains:
- the downloader for the on-chain data;
- the 15 experiments (E0–E14) behind every number in the paper;
- the script that turns their results into the paper's tables, figures and number macros.

---

## Repository layout

```
config.toml                 every setting: download, experiments and paper assets (no secrets)
secrets.example.toml        template for secrets.toml (The Graph API key; secrets.toml is git-ignored)
requirements.txt            Python dependencies

download_subgraph.py        step 1: download pool data from the Uniswap v3 subgraph
subgraph/                   downloader package (client, schema, meta, hourly, daily, positions, prices, report)
bigquery_queries_followup.sql  BigQuery queries 4-7: month boundaries, gap checks, swap logs of thin-liquidity months

run_experiments.py          step 2: run the experiments E0-E14
experiments/                one module per experiment (e00_...e14_) plus shared code:
  config.py                 typed configuration loaded from config.toml
  data.py, outcomes.py      loading the hourly grids and computing position outcomes (IL, fees, variance)
  swaps.py                  swap-by-swap fee replay of thin-liquidity pool-months
  scores.py, selection.py   pool scores (S_lambda, Net90, baselines) and the rolling top-k backtest
  stats.py, correlation.py  bootstrap, coverage tests, IC, token correlation
  plotting.py, output.py    shared figure style, atomic table writes, resumable caching

make_paper_assets.py        step 3: tables, number macros and figures for the paper
paper_figures.py            the paper's figures drawn from saved results

pool_data/                  downloaded data (not in git, about 2 GB; see "Data")
results_v2/                 experiment outputs of the paper run (not in git)
```

## Installation

Python 3.11 or newer is required (`tomllib`). The paper run used Python 3.12, pandas 3.0 and numpy 2.5.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data

### Sources
| Source | What | Used for |
|---|---|---|
| Uniswap v3 subgraph on The Graph (`download_subgraph.py`) | hourly pool data (price, in-range liquidity, TVL, fee-growth accumulators), daily pool data, pool metadata, real full-range LP positions with their snapshots, daily token USD prices | every experiment |
| Google BigQuery, `bigquery-public-data.crypto_ethereum` | daily median gas of LP entry and exit transactions (Query 1); Uniswap v3 `Swap` logs (Queries 3, 6 and 7) | transaction costs; swap-level fee validation (E8) and the swap-by-swap replay of thin-liquidity months |

### Layout of `pool_data/`
```
pool_data/
  meta/          pools.csv/json (tokens, decimals, fee tier, creation), schema.json
  hourly/        <pool address>.csv/json   poolHourDatas, one row per hour (about 1.3 GB)
  daily/         <pool address>.csv/json   poolDayDatas (daily volume and fees)
  positions/     <pool address>.csv/json   real full-range positions and their snapshots (fee validation, E0)
  token_prices/  <token address>.csv/json  tokenDayDatas (daily USD price)
  bigquery/      BigQuery exports of Queries 1, 3, 6 and 7 (about 0.6 GB)
  reports/       sanity report per pool
```
`feeGrowthGlobal*`, `liquidity`, `sqrtPrice` and similar fields are 256-bit integers and are stored as strings. Read them with `dtype=str` and convert with `int(...)`.

### Getting the data
**Option A: download the prepared data.** The full `pool_data/` folder will be published as a separate archive; the link will be added here.

**Option B: rebuild it from the public sources.**
1. Copy `secrets.example.toml` to `secrets.toml` and add a The Graph Studio API key.
2. Set the date range and the step in `config.toml` (`[download]`, `[run]`), then run:
   ```bash
   python download_subgraph.py
   ```
   Every step resumes from the saved data, so a rerun only fetches what is missing or new.
3. Export from BigQuery the daily median gas of Uniswap v3 LP entry and exit transactions (Query 1) and the Uniswap v3 `Swap` logs (Query 3). Also run Queries 6 and 7 of `bigquery_queries_followup.sql`. Save the results as CSV in `pool_data/bigquery/`, and set their file names in `[experiments.data]` of `config.toml`.
   - The query of the thin-liquidity months (Query 7) is written by `run_experiments.py` to `<output_dir>/replay_query.sql` whenever some months are not yet covered.
   - Always keep the timestamp filters: the tables are partitioned on them, and without them a query scans several TB.

## Pools

The 27 Uniswap v3 pools on Ethereum mainnet (from 21 pairs), as listed in `[universe]` of `config.toml`. Creation dates are in UTC.

| # | Pool (pair, fee tier) | Category | Token 0 | Token 1 | Created | Address |
|---|---|---|---|---|---|---|
| 1 | WETH/USDT 0.30% | Stable–volatile | WETH | USDT | 2021-05-05 | [`0x4e68ccd3e89f51c3074ca5072bbac773960dfa36`](https://etherscan.io/address/0x4e68ccd3e89f51c3074ca5072bbac773960dfa36) |
| 2 | WETH/USDT 0.05% | Stable–volatile | WETH | USDT | 2021-05-05 | [`0x11b815efb8f581194ae79006d24e0d814b7697f6`](https://etherscan.io/address/0x11b815efb8f581194ae79006d24e0d814b7697f6) |
| 3 | WBTC/USDT 0.30% | Stable–volatile | WBTC | USDT | 2021-05-05 | [`0x9db9e0e53058c89e5b94e29621a205198648425b`](https://etherscan.io/address/0x9db9e0e53058c89e5b94e29621a205198648425b) |
| 4 | USDC/WETH 0.05% | Stable–volatile | USDC | WETH | 2021-05-05 | [`0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640`](https://etherscan.io/address/0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640) |
| 5 | USDC/WETH 0.30% | Stable–volatile | USDC | WETH | 2021-05-04 | [`0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8`](https://etherscan.io/address/0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8) |
| 6 | WBTC/USDC 0.30% | Stable–volatile | WBTC | USDC | 2021-05-05 | [`0x99ac8ca7087fa4a2a1fb6357269965a2014abc35`](https://etherscan.io/address/0x99ac8ca7087fa4a2a1fb6357269965a2014abc35) |
| 7 | USDC/WETH 1.00% | Stable–volatile | USDC | WETH | 2021-05-04 | [`0x7bea39867e4169dbe237d55c8242a8f2fcdcc387`](https://etherscan.io/address/0x7bea39867e4169dbe237d55c8242a8f2fcdcc387) |
| 8 | WBTC/USDC 0.05% | Stable–volatile | WBTC | USDC | 2021-06-03 | [`0x9a772018fbd77fcd2d25657e5c547baff3fd7d16`](https://etherscan.io/address/0x9a772018fbd77fcd2d25657e5c547baff3fd7d16) |
| 9 | DAI/WETH 0.30% | Stable–volatile | DAI | WETH | 2021-05-04 | [`0xc2e9f25be6257c210d7adf0d4cd6e3e881ba25f8`](https://etherscan.io/address/0xc2e9f25be6257c210d7adf0d4cd6e3e881ba25f8) |
| 10 | USDC/USDT 0.01% | Stablecoin pair | USDC | USDT | 2021-11-13 | [`0x3416cf6c708da44db2624d63ea0aaef7113527c6`](https://etherscan.io/address/0x3416cf6c708da44db2624d63ea0aaef7113527c6) |
| 11 | DAI/USDC 0.01% | Stablecoin pair | DAI | USDC | 2021-11-13 | [`0x5777d92f208679db4b9778590fa3cab3ac9e2168`](https://etherscan.io/address/0x5777d92f208679db4b9778590fa3cab3ac9e2168) |
| 12 | FRAX/USDC 0.05% | Stablecoin pair | FRAX | USDC | 2021-05-06 | [`0xc63b0708e2f7e69cb8a1df0e1389a98c35a76d52`](https://etherscan.io/address/0xc63b0708e2f7e69cb8a1df0e1389a98c35a76d52) |
| 13 | WBTC/WETH 0.05% | Correlated | WBTC | WETH | 2021-05-05 | [`0x4585fe77225b41b697c938b018e2ac67ac5a20c0`](https://etherscan.io/address/0x4585fe77225b41b697c938b018e2ac67ac5a20c0) |
| 14 | WBTC/WETH 0.30% | Correlated | WBTC | WETH | 2021-05-04 | [`0xcbcdf9626bc03e24f779434178a73a0b4bad62ed`](https://etherscan.io/address/0xcbcdf9626bc03e24f779434178a73a0b4bad62ed) |
| 15 | wstETH/WETH 0.01% | Correlated | wstETH | WETH | 2022-08-21 | [`0x109830a1aaad605bbf02a9dfa7b0b92ec2fb7daa`](https://etherscan.io/address/0x109830a1aaad605bbf02a9dfa7b0b92ec2fb7daa) |
| 16 | rETH/WETH 0.05% | Correlated | rETH | WETH | 2022-02-06 | [`0xa4e0faa58465a2d369aa21b3e42d43374c6f9613`](https://etherscan.io/address/0xa4e0faa58465a2d369aa21b3e42d43374c6f9613) |
| 17 | cbETH/WETH 0.05% | Correlated | cbETH | WETH | 2022-08-24 | [`0x840deeef2f115cf50da625f7368c24af6fe74410`](https://etherscan.io/address/0x840deeef2f115cf50da625f7368c24af6fe74410) |
| 18 | AAVE/WETH 0.30% | Governance | AAVE | WETH | 2021-05-05 | [`0x5ab53ee1d50eef2c1dd3d5402789cd27bb52c1bb`](https://etherscan.io/address/0x5ab53ee1d50eef2c1dd3d5402789cd27bb52c1bb) |
| 19 | LINK/WETH 0.30% | Governance | LINK | WETH | 2021-05-05 | [`0xa6cc3c2531fdaa6ae1a3ca84c2855806728693e8`](https://etherscan.io/address/0xa6cc3c2531fdaa6ae1a3ca84c2855806728693e8) |
| 20 | UNI/WETH 0.30% | Governance | UNI | WETH | 2021-05-04 | [`0x1d42064fc4beb5f8aaf85f4617ae8b3b5b8bd801`](https://etherscan.io/address/0x1d42064fc4beb5f8aaf85f4617ae8b3b5b8bd801) |
| 21 | MKR/WETH 0.30% | Governance | MKR | WETH | 2021-05-05 | [`0xe8c6c9227491c0a8156a0106a0204d881bb7e531`](https://etherscan.io/address/0xe8c6c9227491c0a8156a0106a0204d881bb7e531) |
| 22 | LDO/WETH 0.30% | Governance | LDO | WETH | 2021-05-26 | [`0xa3f558aebaecaf0e11ca4b2199cc5ed341edfd74`](https://etherscan.io/address/0xa3f558aebaecaf0e11ca4b2199cc5ed341edfd74) |
| 23 | CRV/WETH 0.30% | Governance | WETH | CRV | 2021-05-05 | [`0x919fa96e88d67499339577fa202345436bcdaf79`](https://etherscan.io/address/0x919fa96e88d67499339577fa202345436bcdaf79) |
| 24 | APE/WETH 0.30% | High-volatility / declining | APE | WETH | 2022-03-17 | [`0xac4b3dacb91461209ae9d41ec517c2b9cb1b7daf`](https://etherscan.io/address/0xac4b3dacb91461209ae9d41ec517c2b9cb1b7daf) |
| 25 | SHIB/WETH 0.30% | High-volatility / declining | SHIB | WETH | 2021-05-05 | [`0x2f62f2b4c5fcd7570a709dec05d68ea19c82a9ec`](https://etherscan.io/address/0x2f62f2b4c5fcd7570a709dec05d68ea19c82a9ec) |
| 26 | PEPE/WETH 0.30% | High-volatility / declining | PEPE | WETH | 2023-04-19 | [`0x11950d141ecb863f01007add7d1a342041227b58`](https://etherscan.io/address/0x11950d141ecb863f01007add7d1a342041227b58) |
| 27 | CRV/WETH 1.00% | High-volatility / declining | WETH | CRV | 2021-05-05 | [`0x4c83a7f819a5c37d64b4c5a2f8238ea082fa1f4e`](https://etherscan.io/address/0x4c83a7f819a5c37d64b4c5a2f8238ea082fa1f4e) |

A pair is *correlated* if both tokens follow the price of the same asset (WBTC/WETH, and liquid-staking tokens against WETH).

## Running the experiments

Choose the experiments and their settings in `[experiments]` of `config.toml`, then run:
```bash
python run_experiments.py
```
- Each experiment writes to `<output_dir>/<id>/` (`results_v2/` for the paper run) and marks itself finished there. The next run skips finished experiments, and long experiments also resume from their saved per-pool or per-cell results.
- Hourly data is parsed once and cached in `<output_dir>/cache/`, and rebuilt automatically when a CSV changes. Set `force_refresh = true` to rerun everything.
- A failing experiment is logged in `<output_dir>/experiments.log`, and the others still run.
- Random choices use the fixed `seed` of `config.toml`, so the results are reproducible.

| ID | Experiment | Main outputs |
|---|---|---|
| E0 | Fee accounting validation: hourly fee-growth fees against real full-range positions | `fee_validation_summary.csv` |
| E1 | IL and net-yield profile per pool, with moving-block bootstrap intervals | per-pool profiles |
| E2 | LVR decomposition: how much of the IL the Gaussian (LVR) rate and the token correlation explain | `summary_by_pool.csv`, `summary_by_category.csv` |
| E3 | Persistence of the log price ratio: rolling AR(1), ADF/KPSS, variance-ratio tests | `rolling_ar1_summary.csv` |
| E4 | Calibration of the IL tail forecasts: Wilson intervals, Kupiec and Christoffersen tests, standardized moves | `exceedance_tests.csv`, `standardized_moves.csv` |
| E5 | Rolling out-of-sample pool selection against the baselines (main experiment) | `risk_return_frontier.csv`, `returns.csv`, `ic_summary.csv`, `tail_tests.csv` |
| E6 | Sensitivity grid: tail level, horizon, calibration window and k | `sensitivity_grid.csv` |
| E7 | Holding horizon matched to the half-life against fixed horizons | horizon comparison |
| E8 | Concentrated ranges (±1% to ±20%, ±1σ to ±3σ), with swap-level fee validation | `top_k_by_range.csv`, `picks_by_category.csv`, `swap_validation_summary.csv` |
| E9 | Fee tiers against the turnover rule | fee-tier comparison |
| E10 | Comparison with the RV and FARV measures | `ic_summary.csv`, `performance.csv` |
| E11 | Replication on another network (needs `pool_data_arbitrum/`; skipped otherwise) | — |
| E12 | Fee crediting rule (fixed-volume or proportional) and position size (10K–1M USD) | crediting grid |
| E13 | Correlation-aware portfolios: choosing the k pools jointly | portfolio comparison |
| E14 | Out-of-sample value of the token correlation in a pool score | IC with and without the correlation |

Key modelling settings in `config.toml`:
- `[experiments.fees].crediting`: how a hypothetical position shares the pool's fees. `fixed_volume` (main, conservative) gives `L · ΔG · L_act / (L_act + L)`, and `proportional` gives `L · ΔG`.
- `[experiments.fees].replay_*`: thresholds that flag thin-liquidity pool-months, whose fees are recomputed swap by swap from the BigQuery swap logs.
- `[experiments.selection].risk_aversion`: the λ grid, fixed before the evaluation.
- `[experiments.costs]`: capital, entry swap and gas costs.

## Paper assets

```bash
python make_paper_assets.py
```
This reads `[paper].results_dir` and writes the following to `[paper].output_dir`:
- `numbers.tex`, with a `\newcommand` for every number quoted in the paper;
- the generated tables (`tab_*.tex`);
- the figures drawn by `paper_figures.py`.

It then zips the paper with these assets for upload to Overleaf. No number in the paper is typed by hand, so rerunning the experiments and this script updates the paper.

## Notes and limitations
- The subgraph prices major stablecoins at exactly 1 USD, so USD-based statistics (e.g., token correlations) miss depeg episodes. IL and fees are computed from the pool price and are not affected.
- The subgraph's hourly `volumeUSD` and `feesUSD` are zero before 2024. Fees are therefore computed from the fee-growth accumulators, and daily fees come from `poolDayDatas`.
- The subgraph's `collectedFeesToken1` duplicates `collectedFeesToken0`, so E0 validates against `feeGrowthInsideLast` instead.
