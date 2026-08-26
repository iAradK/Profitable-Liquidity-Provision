# Profitable Liquidity Provision: Mitigating DeFi Impermanent Loss — Data & Scripts

This repository contains the **data and Python scripts** used in the paper **“Profitable Liquidity Provision: Mitigating DeFi Impermanent Loss”**.
## Data overview

We collect historical data for **eight major Uniswap v3 pools** using **The Graph** public subgraph explorer (https://thegraph.com/explorer), covering **May 2021 – December 2025** (~4.5 years).
For each pool, we track the pool-implied price over time and construct the normalized price-ratio series used to compute impermanent loss (IL).

The dataset spans two fee tiers (**0.05%** and **0.30%**) and three pool categories:
- **Stable–volatile:** WETH/USDT  
- **Volatile–volatile:** WBTC/WETH  
- **Governance-token pairs:** AAVE/WETH, LINK/WETH, UNI/WETH

### Pools
| Pool address | Tokens | Fee tier |
|---|---|---|
| `0x4e68ccd3e89f51c3074ca5072bbac773960dfa36` | WETH/USDT | 0.30% |
| `0x11b815efb8f581194ae79006d24e0d814b7697f6` | WETH/USDT | 0.05% |
| `0x9db9e0e53058c89e5b94e29621a205198648425b` | WBTC/USDT | 0.30% |
| `0x4585fe77225b41b697c938b018e2ac67ac5a20c0` | WBTC/WETH | 0.05% |
| `0xcbcdf9626bc03e24f779434178a73a0b4bad62ed` | WBTC/WETH | 0.30% |
| `0x5ab53ee1d50eef2c1dd3d5402789cd27bb52c1bb` | AAVE/WETH | 0.30% |
| `0xa6cc3c2531fdaa6ae1a3ca84c2855806728693e8` | LINK/WETH | 0.30% |
| `0x1d42064fc4beb5f8aaf85f4617ae8b3b5b8bd801` | UNI/WETH | 0.30% |

## Repository contents

- **Raw pool data** (daily `poolDayDatas` exports):  
  `pool_data/updated/0x*.json`

- **Processed monthly series** (used directly for figures/tables):  
  - `tvl_monthly_end.dat` — end-of-month TVL (USD)  
  - `trading_volume_monthly.dat` — monthly trading volume (USD)

- **Scripts** (generate the processed data and figure inputs):  
  `scripts/*.py`

> Note: This repo focuses on the exact artifacts used in the submitted paper. A de-anonymized version will be released later.

## Setup

Tested with **Python 3.10+**.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Reproducing the main processed outputs

Run the scripts from the repository root:

```bash
python scripts/tvl_analysis.py
python scripts/trading_volume.py
python scripts/monthly_il.py
```

(Additional scripts are provided for other figures/metrics used in the paper.)
