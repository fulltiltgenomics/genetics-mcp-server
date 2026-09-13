-- One row per "Chat complete:" line of the named cluster's chat-backend, in the shape
-- scripts/backfill_turn_metrics.py reads. Usage (the cluster parameter is mandatory —
-- staging and production lines share the production dataset):
--
--   bq query --use_legacy_sql=false --format=json --max_rows=1000000 \
--       --parameter=cluster:STRING:finngenie < scripts/turn_metrics_from_logs.sql
--
-- Both datasets are read because the staging cluster's lines went to the production dataset
-- before the staging sink existed and to both afterwards; the QUALIFY keeps one row per log
-- entry across them. `[session=unknown]` and lines from before the session prefix existed
-- yield a NULL session_id — the turn is still attributable to its user. The rows carry user
-- emails: pipe them into the backfill, never into a file that is kept.
WITH lines AS (
  SELECT insertId, timestamp, resource.labels.cluster_name AS cluster,
         COALESCE(textPayload, JSON_VALUE(TO_JSON_STRING(jsonPayload), '$.message')) AS line
  FROM `daly-finngenie.genetics_chat_logs.stdout`
  UNION ALL
  SELECT insertId, timestamp, resource.labels.cluster_name,
         COALESCE(textPayload, JSON_VALUE(TO_JSON_STRING(jsonPayload), '$.message'))
  FROM `daly-finngenie.genetics_chat_logs_staging.stdout`
)
SELECT
  insertId AS log_id,
  FORMAT_TIMESTAMP('%Y-%m-%d %H:%M:%S', timestamp) AS created_at,
  REGEXP_EXTRACT(line, r'\[user=([^\]]+)\]') AS user_id,
  NULLIF(REGEXP_EXTRACT(line, r'\[session=([^\]]+)\]'), 'unknown') AS session_id,
  REGEXP_EXTRACT(line, r'Chat complete: model=([^ ]+)') AS model,
  CAST(REGEXP_EXTRACT(line, r'iterations=(\d+)') AS INT64) AS iterations,
  CAST(REGEXP_EXTRACT(line, r'total_input_tokens=(\d+)') AS INT64) AS input_tokens,
  CAST(REGEXP_EXTRACT(line, r'total_output_tokens=(\d+)') AS INT64) AS output_tokens,
  CAST(REGEXP_EXTRACT(line, r'total_cost=\$([0-9.]+)') AS FLOAT64) AS cost_usd
FROM lines
WHERE cluster = @cluster AND line LIKE '%Chat complete:%'
QUALIFY ROW_NUMBER() OVER (PARTITION BY insertId ORDER BY timestamp) = 1
ORDER BY timestamp
