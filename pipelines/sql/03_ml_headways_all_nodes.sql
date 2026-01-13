/*
  ML Feature Engineering Script - A/C/E All Nodes
  
  This script creates a feature table for Graph WaveNet by:
  1. Computing headways for ALL stations on A, C, E lines
  2. Computing headways per node (station-line-direction tuple)
  3. Adding temporal features for time embeddings
  
  Output: One row per train arrival with headway computed
  
  Target: mta_transformed.headways_all_nodes
*/
CREATE OR REPLACE TABLE `{{ params.project_id }}.{{ params.dataset_id }}.headways_all_nodes`
PARTITION BY DATE(arrival_time)
CLUSTER BY route_id, direction, stop_id
OPTIONS(
  description="Headways for all A/C/E nodes, partitioned by day, clustered by node"
) AS

WITH arrivals_with_node AS (
  -- Create node_id for each arrival (station-line-direction tuple)
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
    -- Node ID: unique identifier for Graph WaveNet nodes
    CONCAT(stop_id, '_', route_id, '_', direction) AS node_id
  FROM `{{ params.project_id }}.{{ params.dataset_id }}.clean`
  WHERE 
    -- Filter to A/C/E lines only
    route_id IN ('A', 'C', 'E')
    -- Ensure we have valid direction
    AND direction IN ('N', 'S')
    -- Filter valid arrival times
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
    
    -- Previous arrival at the same node
    LAG(arrival_time) OVER (
      PARTITION BY node_id
      ORDER BY arrival_time
    ) AS prev_arrival_time,
    
    -- Headway in minutes (time since previous train at this node)
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

  FROM arrivals_with_node
)

SELECT
  -- Identifiers
  trip_uid,
  node_id,
  route_id,
  direction,
  stop_id,
  stop_name,
  stop_lat,
  stop_lon,
  
  -- Timestamps
  arrival_time,
  prev_arrival_time,
  DATE(arrival_time) AS service_date,
  
  -- Core feature: Headway
  headway_minutes,
  
  -- Temporal features for time embeddings
  EXTRACT(HOUR FROM arrival_time) AS hour_of_day,
  EXTRACT(MINUTE FROM arrival_time) AS minute_of_hour,
  EXTRACT(DAYOFWEEK FROM arrival_time) AS day_of_week,  -- 1=Sunday, 7=Saturday
  
  -- Derived temporal features
  EXTRACT(HOUR FROM arrival_time) * 60 + EXTRACT(MINUTE FROM arrival_time) AS minute_of_day,
  
  -- Day type
  day_type,
  
  -- Is this a peak hour? (7-9 AM or 5-7 PM on weekdays)
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
  -- Filter out first arrival (no previous train to compute headway)
  prev_arrival_time IS NOT NULL
  -- Filter out unreasonable headways (> 2 hours likely indicates service gap/overnight)
  AND TIMESTAMP_DIFF(arrival_time, prev_arrival_time, SECOND) / 60.0 < 120
  -- Filter out negative or zero headways (data quality issue)
  AND TIMESTAMP_DIFF(arrival_time, prev_arrival_time, SECOND) > 0
ORDER BY arrival_time;
