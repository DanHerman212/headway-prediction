-- Hourly evaluation: join predictions to actuals, compute error metrics.
-- Scheduled via bash/schedule_eval_query.sh

INSERT INTO `realtime-headway-prediction.headway_monitoring.evaluation_results`
  (eval_time, window_hours, n_predictions,
   mae_seconds, smape_pct, median_abs_error, p90_abs_error)

WITH matched AS (
  SELECT
    p.group_id,
    p.headway_p50,
    a.service_headway   AS actual,
    ABS(a.service_headway - p.headway_p50)  AS abs_error,
    SAFE_DIVIDE(
      ABS(a.service_headway - p.headway_p50),
      (ABS(a.service_headway) + ABS(p.headway_p50)) / 2
    ) AS smape
  FROM `realtime-headway-prediction.headway_monitoring.predictions` p
  INNER JOIN `realtime-headway-prediction.headway_monitoring.actuals` a
    ON  a.group_id = p.group_id
    AND a.time_idx = p.last_observation_time_idx + 1
  WHERE p.headway_p50 IS NOT NULL
    AND a.service_headway IS NOT NULL
    AND a.service_headway > 0
    -- 15-min lag so actuals have time to land
    AND p.predicted_at BETWEEN
        TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 75 MINUTE)
        AND TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 15 MINUTE)
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
