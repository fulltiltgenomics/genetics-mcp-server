-- Per-turn cost baseline for the cross-session memory premise (bq query --use_legacy_sql=false < this file).
-- The cluster filter is mandatory: staging rows land in the same table, and the log lines carry user
-- emails, so select aggregates only — never a row that could carry the `[user=...]` capture out.
WITH t AS (
  SELECT timestamp,
    REGEXP_EXTRACT(jsonPayload.message, r'model=([^ ]+)') AS model,
    CAST(REGEXP_EXTRACT(jsonPayload.message, r'iterations=(\d+)') AS INT64) AS iterations,
    CAST(REGEXP_EXTRACT(jsonPayload.message, r'total_input_tokens=(\d+)') AS INT64) AS input_tokens,
    CAST(REGEXP_EXTRACT(jsonPayload.message, r'total_cost=\$([0-9.]+)') AS FLOAT64) AS cost,
    REGEXP_EXTRACT(jsonPayload.message, r'\[user=([^\]]+)\]') AS u
  FROM `daly-finngenie.genetics_chat_logs.stdout`
  WHERE jsonPayload.message LIKE '%Chat complete%' AND resource.labels.cluster_name='finngenie'
    AND timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY)
)
SELECT model, COUNT(*) n, COUNT(DISTINCT u) users,
  ROUND(AVG(cost),4) mean_cost,
  ROUND(APPROX_QUANTILES(cost,100)[OFFSET(25)],4) p25_cost, ROUND(APPROX_QUANTILES(cost,100)[OFFSET(50)],4) median_cost,
  ROUND(APPROX_QUANTILES(cost,100)[OFFSET(75)],4) p75_cost, ROUND(APPROX_QUANTILES(cost,100)[OFFSET(90)],4) p90_cost,
  APPROX_QUANTILES(input_tokens,100)[OFFSET(50)] median_input_tokens, APPROX_QUANTILES(input_tokens,100)[OFFSET(90)] p90_input_tokens,
  APPROX_QUANTILES(iterations,100)[OFFSET(50)] median_iter, ROUND(COUNTIF(iterations=1)/COUNT(*),3) share_single_iter,
  ROUND(APPROX_QUANTILES(IF(iterations=1,cost,NULL),100)[OFFSET(50)],4) median_cost_single_iter
FROM t GROUP BY model ORDER BY n DESC
