/*
  Incremental ML Feature Engineering - A/C/E All Nodes
  
  Target Table: headways_all_nodes
  Partition Key: service_date (DATE)
  
  Logic:
  1. Defines a processing window (Today + Yesterday for LAG lookback)
  2. Removes any existing data for the execution date (Idempotency)
  3. Calculates headways using 2-day window for accurate LAG()
  4. Filters results to keep only target date's data
  5. Inserts into partitioned table
*/

-- 1. Define Variables
DECLARE process_date DATE DEFAULT DATE('{{ ds }}');
DECLARE lookback_start DATE DEFAULT process_date - 1;

-- 2. Idempotency: Delete data for the specific date being processed
DELETE FROM `{{ params.project_id }}.{{ params.dataset_id }}.headways_all_nodes`
WHERE service_date = process_date;

-- 3. Transformation & Insert
INSERT INTO `{{ params.project_id }}.{{ params.dataset_id }}.headways_all_nodes`

WITH window_context AS (
  -- Read Clean Data for Today AND Yesterday (Lookback for LAG)
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
  FROM `{{ params.project_id }}.{{ params.dataset_id }}.clean`
  WHERE 
    DATE(start_time_dts) BETWEEN lookback_start AND process_date
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
  DATE(arrival_time) = process_date
  AND prev_arrival_time IS NOT NULL
  AND TIMESTAMP_DIFF(arrival_time, prev_arrival_time, SECOND) / 60.0 < 120
  AND TIMESTAMP_DIFF(arrival_time, prev_arrival_time, SECOND) > 0;
