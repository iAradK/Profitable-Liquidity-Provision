-- Follow-up BigQuery queries (Ethereum mainnet)

-- ============================================================
-- Query 4: First block of each month
-- Used for: the time-travel universe query (subgraph dataset 5: top pools by TVL at each month start)
-- ============================================================
SELECT
  DATE_TRUNC(DATE(timestamp), MONTH) AS month,
  MIN(number) AS first_block
FROM `bigquery-public-data.crypto_ethereum.blocks`
WHERE timestamp >= TIMESTAMP('2021-05-01') AND timestamp < TIMESTAMP('2026-12-01')
GROUP BY month
ORDER BY month;


-- ============================================================
-- Query 5: Swap counts during the long gaps in the hourly subgraph data
-- Used for: confirming the gaps are real inactivity, not subgraph indexing holes.
-- Expected: ~0 swaps per day. Many swaps would mean the subgraph is missing data for that window.
-- One SELECT per window, so each one prunes partitions on its own date range.
-- ============================================================
SELECT 'LDO/WETH 0.30%' AS pool, DATE(block_timestamp) AS day, COUNT(*) AS n_swaps
FROM `bigquery-public-data.crypto_ethereum.logs`
WHERE block_timestamp >= TIMESTAMP('2021-07-25') AND block_timestamp < TIMESTAMP('2022-03-23')
  AND address = '0xa3f558aebaecaf0e11ca4b2199cc5ed341edfd74'
  AND topics[SAFE_OFFSET(0)] = '0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67'
GROUP BY day
UNION ALL
SELECT 'WBTC/USDC 0.05%', DATE(block_timestamp), COUNT(*)
FROM `bigquery-public-data.crypto_ethereum.logs`
WHERE block_timestamp >= TIMESTAMP('2022-04-02') AND block_timestamp < TIMESTAMP('2022-07-15')
  AND address = '0x9a772018fbd77fcd2d25657e5c547baff3fd7d16'
  AND topics[SAFE_OFFSET(0)] = '0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67'
GROUP BY 2
UNION ALL
SELECT 'wstETH/WETH 0.01%', DATE(block_timestamp), COUNT(*)
FROM `bigquery-public-data.crypto_ethereum.logs`
WHERE block_timestamp >= TIMESTAMP('2023-01-29') AND block_timestamp < TIMESTAMP('2023-04-30')
  AND address = '0x109830a1aaad605bbf02a9dfa7b0b92ec2fb7daa'
  AND topics[SAFE_OFFSET(0)] = '0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67'
GROUP BY 2
UNION ALL
SELECT 'FRAX/USDC 0.05%', DATE(block_timestamp), COUNT(*)
FROM `bigquery-public-data.crypto_ethereum.logs`
WHERE block_timestamp >= TIMESTAMP('2021-05-22') AND block_timestamp < TIMESTAMP('2021-06-05')
  AND address = '0xc63b0708e2f7e69cb8a1df0e1389a98c35a76d52'
  AND topics[SAFE_OFFSET(0)] = '0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67'
GROUP BY 2
ORDER BY pool, day;


-- ============================================================
-- Query 6: Swap logs of CRV/WETH 0.30%, 2023-06-01 to 2023-11-24 (local CSV download, cut at 10 MB)
-- Used for: the swap-by-swap replay of the thin-liquidity months (E12, experiments/swaps.py)
-- File: pool_data/bigquery/job_iosgldBVFtwQFtbY8l9iSwbwq9z_.csv
-- ============================================================
SELECT block_timestamp, block_number, transaction_hash, log_index, address AS pool, data
FROM `bigquery-public-data.crypto_ethereum.logs`
WHERE topics[SAFE_OFFSET(0)] = '0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67'
  AND address = '0x919fa96e88d67499339577fa202345436bcdaf79'
  AND block_timestamp >= TIMESTAMP('2023-06-01') AND block_timestamp < TIMESTAMP('2024-05-01')
ORDER BY block_number, log_index;

-- ============================================================
-- Query 7: the remaining thin-liquidity pool-months, as written by run_experiments.py to
-- results_v2/replay_query.sql (rerun gives the current list). Save to Google Drive: the result exceeds 10 MB.
-- ============================================================
SELECT block_timestamp, block_number, transaction_hash, log_index, address AS pool, data
FROM `bigquery-public-data.crypto_ethereum.logs`
WHERE topics[SAFE_OFFSET(0)] = '0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67'
  AND (
    (address = '0xac4b3dacb91461209ae9d41ec517c2b9cb1b7daf' AND block_timestamp >= TIMESTAMP('2025-10-01') AND block_timestamp < TIMESTAMP('2025-11-01'))  -- APE/WETH 0.30%
    OR (address = '0x919fa96e88d67499339577fa202345436bcdaf79' AND block_timestamp >= TIMESTAMP('2023-01-01') AND block_timestamp < TIMESTAMP('2023-02-01'))  -- CRV/WETH 0.30%
    OR (address = '0x919fa96e88d67499339577fa202345436bcdaf79' AND block_timestamp >= TIMESTAMP('2024-01-01') AND block_timestamp < TIMESTAMP('2024-02-01'))  -- CRV/WETH 0.30%
    OR (address = '0x919fa96e88d67499339577fa202345436bcdaf79' AND block_timestamp >= TIMESTAMP('2024-02-01') AND block_timestamp < TIMESTAMP('2024-03-01'))  -- CRV/WETH 0.30%
    OR (address = '0x919fa96e88d67499339577fa202345436bcdaf79' AND block_timestamp >= TIMESTAMP('2024-04-01') AND block_timestamp < TIMESTAMP('2024-05-01'))  -- CRV/WETH 0.30%
    OR (address = '0x4c83a7f819a5c37d64b4c5a2f8238ea082fa1f4e' AND block_timestamp >= TIMESTAMP('2023-07-01') AND block_timestamp < TIMESTAMP('2023-08-01'))  -- CRV/WETH 1.00%
    OR (address = '0xa3f558aebaecaf0e11ca4b2199cc5ed341edfd74' AND block_timestamp >= TIMESTAMP('2022-08-01') AND block_timestamp < TIMESTAMP('2022-09-01'))  -- LDO/WETH 0.30%
    OR (address = '0xa3f558aebaecaf0e11ca4b2199cc5ed341edfd74' AND block_timestamp >= TIMESTAMP('2024-05-01') AND block_timestamp < TIMESTAMP('2024-06-01'))  -- LDO/WETH 0.30%
    OR (address = '0xa3f558aebaecaf0e11ca4b2199cc5ed341edfd74' AND block_timestamp >= TIMESTAMP('2024-06-01') AND block_timestamp < TIMESTAMP('2024-07-01'))  -- LDO/WETH 0.30%
    OR (address = '0x7bea39867e4169dbe237d55c8242a8f2fcdcc387' AND block_timestamp >= TIMESTAMP('2026-03-01') AND block_timestamp < TIMESTAMP('2026-04-01'))  -- USDC/WETH 1.00%
    OR (address = '0x9a772018fbd77fcd2d25657e5c547baff3fd7d16' AND block_timestamp >= TIMESTAMP('2023-12-01') AND block_timestamp < TIMESTAMP('2024-01-01'))  -- WBTC/USDC 0.05%
    OR (address = '0x9db9e0e53058c89e5b94e29621a205198648425b' AND block_timestamp >= TIMESTAMP('2022-07-01') AND block_timestamp < TIMESTAMP('2022-08-01'))  -- WBTC/USDT 0.30%
    OR (address = '0x840deeef2f115cf50da625f7368c24af6fe74410' AND block_timestamp >= TIMESTAMP('2023-07-01') AND block_timestamp < TIMESTAMP('2023-08-01'))  -- cbETH/WETH 0.05%
    OR (address = '0x109830a1aaad605bbf02a9dfa7b0b92ec2fb7daa' AND block_timestamp >= TIMESTAMP('2025-02-01') AND block_timestamp < TIMESTAMP('2025-03-01'))  -- wstETH/WETH 0.01%
  )
ORDER BY block_number, log_index;
