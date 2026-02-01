WITH computed_headway AS (
  SELECT
    arrival_time,
    route_id,
    train_id,
    track,
    EXTRACT(HOUR FROM arrival_time) AS hour_of_day,
    EXTRACT(DAYOFWEEK FROM arrival_time) AS day_of_week,
    ROUND(TIMESTAMP_DIFF(
      arrival_time, 
      LAG(arrival_time) OVER (PARTITION BY route_id, track ORDER BY arrival_time ASC), 
      SECOND
    ) / 60, 2) AS service_headway,
    ROUND(TIMESTAMP_DIFF(
      arrival_time, 
      LAG(arrival_time) OVER (PARTITION BY track ORDER BY arrival_time ASC), 
      SECOND
    ) / 60, 2) AS track_headway
  FROM `realtime-headway-prediction.headway_prediction.clean`
  WHERE stop_id = "A32S"
    AND route_id IN ("A", "C", "E")
    AND arrival_time > '2025-07-01' -- Filter for relevant period
),
empirical_stats AS (
  SELECT
    route_id,
    track,
    day_of_week,
    hour_of_day,
    APPROX_QUANTILES(service_headway, 100)[OFFSET(50)] as typical_headway
  FROM computed_headway
  WHERE service_headway IS NOT NULL 
    AND service_headway BETWEEN 0 AND 120
  GROUP BY 1, 2, 3, 4
)
SELECT
  t.*,
  COALESCE(s.typical_headway, t.service_headway) as scheduled_headway
FROM computed_headway t
LEFT JOIN empirical_stats s
  ON t.route_id = s.route_id
  AND t.track = s.track
  AND t.day_of_week = s.day_of_week
  AND t.hour_of_day = s.hour_of_day
WHERE t.service_headway IS NOT NULL
  AND t.track_headway IS NOT NULL
ORDER BY t.arrival_time ASC
