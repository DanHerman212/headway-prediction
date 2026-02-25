-- Hourly evaluation: join predictions to actuals, compute error metrics.
-- Scheduled via bash/schedule_eval_query.sh

INSERT INTO `realtime-headway-prediction.headway_monitoring.evaluation_results`
  (eval_time, window_hours, n_predictions,
   mae_seconds, smape_pct, median_abs_error, p90_abs_error)

WITH next_actuals AS (
  -- For each prediction, find the next actual (smallest time_idx > last_observation_time_idx)
  SELECT
    p.group_id,
    p.last_observation_time_idx,
    p.headway_p50,
    p.predicted_at,
    a.service_headway AS actual,
    ROW_NUMBER() OVER (
      PARTITION BY p.group_id, p.last_observation_time_idx
      ORDER BY a.time_idx ASC
    ) AS rn
  FROM `realtime-headway-prediction.headway_monitoring.predictions` p
  INNER JOIN `realtime-headway-prediction.headway_monitoring.actuals` a
    ON  a.group_id = p.group_id
    AND a.time_idx > p.last_observation_time_idx
  WHERE p.headway_p50 IS NOT NULL
    AND a.service_headway IS NOT NULL
    AND a.service_headway > 0
    AND p.predicted_at BETWEEN
        TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 75 MINUTE)
        AND TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 15 MINUTE)
),

matched AS (
  SELECT
    group_id,
    headway_p50,
    actual,
    ABS(actual - headway_p50)  AS abs_error,
    SAFE_DIVIDE(
      ABS(actual - headway_p50),
      (ABS(actual) + ABS(headway_p50)) / 2
    ) AS smape
  FROM next_actuals
  WHERE rn = 1
)

SELECT
  CURRENT_TIMESTAMP()  AS eval_time,
  1                    AS window_hours,
  COUNT(*)             AS n_predictions,
  ROUND(AVG(abs_error), 2)                                    AS mae_seconds,
  ROUND(AVG(smape) * 100, 2)                                  AS smape_pct,
  ROUND(APPROX_QUANTILES(abs_error, 100)[OFFSET(50)], 2)     AS median_abs_error,
  ROUND(APPROX_QUANTILES(abs_error, 100)[OFFSET(90)], 2)     AS p90_abs_error
FROM matched
HAVING COUNT(*) > 0
