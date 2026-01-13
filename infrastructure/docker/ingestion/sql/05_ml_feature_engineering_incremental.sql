/*
  Incremental ML Feature Engineering Script (Partitioned with Lookback)
  
  Source: {{ params.project_id }}.mta_transformed.clean
  Target: {{ params.project_id }}.mta_transformed.ml
  Partition Key: arrival_date (DATE)
  
  Logic:
  1. Defines a processing window (Today + Yesterday).
  2. Removes any existing data for the execution date (Idempotency).
  3. Calculates features using the 2-day window to ensure LAG() works for the first trip of the day.
  4. Filters results to keep only Today's data.
  5. Inserts Today's data into the target table.
*/

-- 1. Define Variables
DECLARE process_date DATE DEFAULT DATE('{{ ds }}');

-- 2. Idempotency: Delete data for the specific date being processed
DELETE FROM `{{ params.project_id }}.mta_transformed.ml`
WHERE arrival_date = process_date;

-- 3. Transformation & Insert
INSERT INTO `{{ params.project_id }}.mta_transformed.ml`

WITH window_context AS (
  -- Read Clean Data for Today AND Yesterday (Lookback)
  SELECT *
  FROM `{{ params.project_id }}.mta_transformed.clean`
  WHERE DATE(start_time_dts) BETWEEN process_date - 1 AND process_date
),

target_origin AS (
  -- Self-join logic (applied to the 2-day window)
  SELECT
    target.trip_uid,
    target.stop_name,
    target.direction,
    target.route_id,
    TIMESTAMP_SECONDS(target.arrival_time) AS target_arrival,
    TIMESTAMP_SECONDS(origin.arrival_time) AS origin_arrival
  FROM window_context target
  JOIN window_context origin
  ON target.trip_uid = origin.trip_uid
  WHERE target.stop_name = "Lexington Av/53 St"
    AND target.direction = "S"
    AND target.route_id = "E"
    AND origin.stop_name = "Jamaica Center-Parsons/Archer"
    AND origin.direction = "S"
    AND origin.route_id = "E"
),

calculated_features AS (
  SELECT
    DATE(target_arrival) AS arrival_date,
    target_arrival,
    
    -- Feature: Trip Duration
    ROUND(TIMESTAMP_DIFF(target_arrival, origin_arrival, SECOND) / 60, 2) AS duration,
    
    -- Feature: Minutes Between Trains (Headway)
    -- This LAG now has access to yesterday's data if needed
    ROUND(TIMESTAMP_DIFF(target_arrival, LAG(target_arrival) 
      OVER 
        (PARTITION BY stop_name, direction, route_id
          ORDER BY target_arrival), SECOND) / 60, 2) AS mbt,
          
    -- Feature: Day of Week
    CASE 
      WHEN FORMAT_TIMESTAMP('%A', target_arrival) IN ('Saturday', 'Sunday')
        THEN 1
      ELSE 0
    END AS dow
  FROM target_origin
)

-- Final Selection: Keep ONLY Today's data
SELECT
  arrival_date,
  duration,
  mbt,
  dow
FROM calculated_features
WHERE arrival_date = process_date;
