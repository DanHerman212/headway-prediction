/*
  West 4th Street EDA Dataset Extraction
  
  Purpose: Extract raw arrivals at West 4th St-Washington Sq (A32N/A32S) 
           for A, C, E trains with computed headways for exploratory data analysis.
  
  Source: headway_dataset.clean
  Target: headway_dataset.ml
  Date Range: Sep 13, 2025 - Jan 13, 2026 (4 months)
  
  Station Layout:
    - A32S (Southbound): A1 (local), A3 (express)
    - A32N (Northbound): A2 (local), A4 (express)
    - This query filters to LOCAL tracks only (A1, A2)
  
  Headway Logic: Composite headway (time since any previous train on same track)
                 Partitioned by stop_id, track
*/

CREATE OR REPLACE TABLE `realtime-headway-prediction.headway_dataset.ml` AS

WITH arrivals AS (
  SELECT
    route_id,
    direction,
    stop_id,
    arrival_time_ts,
    stop_name,
    day_type,
    track
  FROM `realtime-headway-prediction.headway_dataset.clean`
  WHERE 
    stop_id IN ('A32N', 'A32S')
    AND route_id IN ('A', 'C', 'E')
    AND track IN ('A1', 'A2')
    AND EXTRACT(DATE FROM arrival_time_ts) BETWEEN '2025-08-01' AND '2026-12-31'
),

headways_computed AS (
  SELECT
    route_id,
    direction,
    stop_id,
    arrival_time_ts,
    stop_name,
    day_type,
    track,
    
    -- Previous arrival on same track (composite headway)
    LAG(arrival_time_ts) OVER (
      PARTITION BY stop_id, track
      ORDER BY arrival_time_ts
    ) AS prev_arrival_time,
    
    -- Headway in seconds
    TIMESTAMP_DIFF(
      arrival_time_ts,
      LAG(arrival_time_ts) OVER (
        PARTITION BY stop_id, track
        ORDER BY arrival_time_ts
      ),
      SECOND
    ) AS headway_total_seconds
    
  FROM arrivals
)

SELECT
  route_id,
  direction,
  stop_id,
  arrival_time_ts,
  stop_name,
  day_type,
  track,
  prev_arrival_time,
  
  -- Headway components
  headway_total_seconds,
  CAST(FLOOR(headway_total_seconds / 60) AS INT64) AS headway_minutes,
  CAST(MOD(headway_total_seconds, 60) AS INT64) AS headway_seconds_remainder,
  
  -- Headway display format (MM:SS)
  CONCAT(
    CAST(FLOOR(headway_total_seconds / 60) AS STRING),
    ':',
    LPAD(CAST(MOD(headway_total_seconds, 60) AS STRING), 2, '0')
  ) AS headway_display

FROM headways_computed
ORDER BY direction, track, arrival_time_ts;
