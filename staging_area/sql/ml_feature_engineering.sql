/*
  ML Feature Engineering Script
  
  This script creates a feature table for machine learning models by:
  1. Identifying specific trips on the 'E' line going South
  2. Calculating trip duration between Jamaica Center and Lexington Av/53 St
  3. Computing headway (time between trains)
  4. Adding temporal features like day of week
*/
CREATE OR REPLACE TABLE `{{ params.project_id }}.{{ params.dataset_id }}.ml`
PARTITION BY DATE(arrival_date)
OPTIONS(
  description="ML Features, partitioned by day"
) AS

WITH target_origin AS (
  -- Self-join to connect origin (Jamaica Center) and destination (Lexington Av) for the same trip
  SELECT
    target.trip_uid, -- Primary key for joining
    target.stop_name, -- Used for partitioning in window functions
    target.direction, -- Used for partitioning in window functions
    target.route_id, -- Used for partitioning in window functions
    TIMESTAMP_SECONDS(target.arrival_time) AS target_arrival, -- Arrival time at target station (Lexington Av)
    TIMESTAMP_SECONDS(origin.arrival_time) AS origin_arrival -- Arrival time at origin station (Jamaica Center)
  FROM `{{ params.project_id }}.{{ params.dataset_id }}.clean` target
  JOIN `{{ params.project_id }}.{{ params.dataset_id }}.clean` origin
  ON target.trip_uid = origin.trip_uid
  WHERE target.stop_name = "Lexington Av/53 St"
    AND target.direction = "S"
    AND target.route_id = "E"
    AND origin.stop_name = "Jamaica Center-Parsons/Archer"
    AND origin.direction = "S"
    AND origin.route_id = "E"
    -- Filter for years 2024-2025
    AND EXTRACT(YEAR FROM TIMESTAMP_SECONDS(target.arrival_time)) BETWEEN 2024 AND 2025
    AND EXTRACT(YEAR FROM TIMESTAMP_SECONDS(origin.arrival_time)) BETWEEN 2024 AND 2025
  ORDER BY trip_uid ASC, target_arrival ASC
)
SELECT
  target_arrival AS arrival_date,
  
  -- Feature: Trip Duration (in minutes)
  -- Calculated as difference between arrival at target and origin
  ROUND(TIMESTAMP_DIFF(target_arrival, origin_arrival, SECOND) / 60, 2) AS duration,
  
  -- Feature: Minutes Between Trains (Headway)
  -- Calculated as time difference between current and previous train arrival at target
  ROUND(TIMESTAMP_DIFF(target_arrival, LAG(target_arrival) 
    OVER 
      (PARTITION BY target_origin.stop_name, target_origin.direction, target_origin.route_id
        ORDER BY target_arrival), SECOND) / 60, 2) AS mbt,
        
  -- Feature: Day of Week (One-hot encoded-ish: 1 for Weekend, 0 for Weekday)
  CASE 
    WHEN FORMAT_TIMESTAMP('%A', target_arrival) IN ('Saturday', 'Sunday')
      THEN 1
    ELSE 0
  END AS dow
FROM target_origin
ORDER BY target_arrival ASC;
