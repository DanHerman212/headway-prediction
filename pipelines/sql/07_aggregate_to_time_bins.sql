/*
  Aggregate Headways to 5-Minute Time Bins
  
  This script aggregates per-arrival headways into 5-minute time bins
  for each node, suitable for Graph WaveNet input.
  
  Logic:
  - Floor arrival_time to 5-minute intervals
  - Take the LAST observed headway in each bin (most recent state)
  - Forward-fill missing bins will be handled in Python/Beam
  
  Output: headways_binned table with (time_bin, node_id, headway)
*/
CREATE OR REPLACE TABLE `{{ params.project_id }}.{{ params.dataset_id }}.headways_binned` 
PARTITION BY DATE(time_bin)
CLUSTER BY node_id
AS

WITH binned AS (
  SELECT
    -- Floor to 5-minute bin
    TIMESTAMP_TRUNC(arrival_time, MINUTE) AS arrival_minute,
    TIMESTAMP_SECONDS(
      UNIX_SECONDS(TIMESTAMP_TRUNC(arrival_time, MINUTE)) 
      - MOD(UNIX_SECONDS(TIMESTAMP_TRUNC(arrival_time, MINUTE)), 300)
    ) AS time_bin,
    node_id,
    route_id,
    direction,
    stop_id,
    headway_minutes,
    arrival_time,
    hour_of_day,
    day_of_week,
    day_type,
    is_peak_hour
  FROM `{{ params.project_id }}.{{ params.dataset_id }}.headways_all_nodes`
),

-- Take the last headway observation in each 5-minute bin
last_in_bin AS (
  SELECT
    time_bin,
    node_id,
    route_id,
    direction,
    stop_id,
    headway_minutes,
    hour_of_day,
    day_of_week,
    day_type,
    is_peak_hour,
    -- Rank arrivals within each bin, take latest
    ROW_NUMBER() OVER (
      PARTITION BY time_bin, node_id 
      ORDER BY arrival_time DESC
    ) AS rn
  FROM binned
)

SELECT
  time_bin,
  node_id,
  route_id,
  direction,
  stop_id,
  headway_minutes,
  hour_of_day,
  day_of_week,
  day_type,
  is_peak_hour,
  DATE(time_bin) AS service_date,
  -- Time features for embeddings
  EXTRACT(HOUR FROM time_bin) * 60 + EXTRACT(MINUTE FROM time_bin) AS minute_of_day
FROM last_in_bin
WHERE rn = 1
ORDER BY time_bin, node_id;
