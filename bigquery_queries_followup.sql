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
