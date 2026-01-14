/*
  Incremental ML Feature Engineering - A/C/E All Nodes (Weekly)
  
  Source: {{ params.project_id }}.headway_dataset.clean
  Target: {{ params.project_id }}.headway_dataset.ml
  Partition Key: arrival_time (DATE)
  
  Parameters (substituted via sed):
    {{ params.project_id }} - GCP project ID
    {{ params.start_date }} - Start of date range (YYYY-MM-DD)
    {{ params.end_date }} - End of date range (YYYY-MM-DD)
  
  Logic:
  1. Defines a processing window (date range + 1 day lookback for LAG)
  2. Removes any existing data for the date range (Idempotency)
  3. Calculates headways using extended window for accurate LAG()
  4. Filters results to keep only target date range
  5. Inserts into partitioned table
*/

-- 1. Define Variables
DECLARE start_date DATE DEFAULT DATE('{{ params.start_date }}');
DECLARE end_date DATE DEFAULT DATE('{{ params.end_date }}');
DECLARE lookback_start DATE DEFAULT start_date - 1;

-- 2. Idempotency: Delete data for the date range being processed
DELETE FROM `{{ params.project_id }}.headway_dataset.ml`
WHERE DATE(arrival_time) BETWEEN start_date AND end_date;

-- 3. Transformation & Insert
INSERT INTO `{{ params.project_id }}.headway_dataset.ml`

WITH window_context AS (
  -- Read Clean Data for date range + 1 day lookback for LAG
  SELECT
    trip_uid,
    route_id,
    direction,
    stop_id,
    stop_name,
    stop_lat,
    stop_lon,
    TIMESTAMP_SECONDS(arrival_time) AS arrival_time,
    start_time_dts,
    day_type,
    CONCAT(stop_id, '_', route_id, '_', direction) AS node_id
  FROM `{{ params.project_id }}.headway_dataset.clean`
  WHERE 
    DATE(start_time_dts) BETWEEN lookback_start AND end_date
    AND route_id IN ('A', 'C', 'E')
    AND direction IN ('N', 'S')
    AND arrival_time IS NOT NULL
),

headways_computed AS (
  SELECT
    trip_uid,
    route_id,
    direction,
    stop_id,
    stop_name,
    stop_lat,
    stop_lon,
    arrival_time,
    start_time_dts,
    day_type,
    node_id,
    
    LAG(arrival_time) OVER (
      PARTITION BY node_id
      ORDER BY arrival_time
    ) AS prev_arrival_time,
    
    ROUND(
      TIMESTAMP_DIFF(
        arrival_time,
        LAG(arrival_time) OVER (
          PARTITION BY node_id
          ORDER BY arrival_time
        ),
        SECOND
      ) / 60.0,
      2
    ) AS headway_minutes

  FROM window_context
)

-- Final Selection: Keep ONLY target date's data
SELECT
  trip_uid,
  node_id,
  route_id,
  direction,
  stop_id,
  stop_name,
  stop_lat,
  stop_lon,
  arrival_time,
  prev_arrival_time,
  DATE(arrival_time) AS service_date,
  headway_minutes,
  EXTRACT(HOUR FROM arrival_time) AS hour_of_day,
  EXTRACT(MINUTE FROM arrival_time) AS minute_of_hour,
  EXTRACT(DAYOFWEEK FROM arrival_time) AS day_of_week,
  EXTRACT(HOUR FROM arrival_time) * 60 + EXTRACT(MINUTE FROM arrival_time) AS minute_of_day,
  day_type,
  CASE
    WHEN day_type = 'Weekday' 
      AND (
        (EXTRACT(HOUR FROM arrival_time) BETWEEN 7 AND 9)
        OR (EXTRACT(HOUR FROM arrival_time) BETWEEN 17 AND 19)
      )
    THEN 1
    ELSE 0
  END AS is_peak_hour
FROM headways_computed
WHERE 
  DATE(arrival_time) BETWEEN start_date AND end_date
  AND prev_arrival_time IS NOT NULL
  AND TIMESTAMP_DIFF(arrival_time, prev_arrival_time, SECOND) / 60.0 < 120
  AND TIMESTAMP_DIFF(arrival_time, prev_arrival_time, SECOND) > 0;
