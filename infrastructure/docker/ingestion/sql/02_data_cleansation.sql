/*
  Data Cleansing and Transformation Script
  
  This script performs the following operations:
  1. Parses the complex trip_uid string to extract trip metadata
  2. Joins raw sensor data with static stop information
  3. Enriches data with derived fields like day_type
  4. Creates a clean, denormalized table for downstream analysis
  
  Source: {{ params.project_id }}.mta_raw.raw, {{ params.project_id }}.mta_raw.stops
  Target: {{ params.project_id }}.mta_transformed.clean
*/
CREATE OR REPLACE TABLE `{{ params.project_id }}.mta_transformed.clean`
PARTITION BY DATE(start_time_dts)
OPTIONS(
  description="Cleaned and denormalized subway data, partitioned by day"
) AS
SELECT
  sd.trip_uid,
  -- Extract start timestamp from the first part of trip_uid (before the first underscore)
  TIMESTAMP_SECONDS(CAST(SUBSTR(sd.trip_uid, 1, STRPOS(sd.trip_uid, '_') - 1) AS INT64)) AS start_time_dts,
  
  -- Extract route_id (e.g., 'E', 'F') from trip_uid
  -- Logic handles different trip_uid formats (with or without '..')
  CASE
    WHEN STRPOS(SUBSTR(sd.trip_uid, STRPOS(sd.trip_uid, '_') + 1), '..') > 0 THEN SUBSTR(SUBSTR(sd.trip_uid,
        STRPOS(sd.trip_uid, '_') + 1), 1, STRPOS(SUBSTR(sd.trip_uid, STRPOS(sd.trip_uid, '_') + 1), '..') - 1)
    ELSE SUBSTR(sd.trip_uid, STRPOS(sd.trip_uid, '_') + 1)
  END AS route_id,
  
  -- Extract direction (N/S/E/W) from trip_uid
  CASE
    WHEN STRPOS(SUBSTR(sd.trip_uid, STRPOS(sd.trip_uid, '_') + 1), '..') > 0 THEN SUBSTR(SUBSTR(sd.trip_uid,
        STRPOS(sd.trip_uid, '_') + 1), STRPOS(SUBSTR(sd.trip_uid, STRPOS(sd.trip_uid, '_') + 1), '..') + 2, 1)
    ELSE NULL
  END AS direction,
  
  -- Extract path identifier from trip_uid
  CASE
    WHEN STRPOS(SUBSTR(sd.trip_uid, STRPOS(sd.trip_uid, '_') + 1), '..') > 0 THEN SUBSTR(SUBSTR(sd.trip_uid,
        STRPOS(sd.trip_uid, '_') + 1), STRPOS(SUBSTR(sd.trip_uid, STRPOS(sd.trip_uid, '_') + 1), '..') + 3)
    ELSE NULL
  END AS path_identifier,
  
  sd.stop_id,
  sd.track,
  sd.arrival_time,
  sd.departure_time,
  sd.last_observed,
  sd.marked_past,
  
  -- Arrival time as TIMESTAMP (needed by stored procedures)
  TIMESTAMP_SECONDS(sd.arrival_time) AS arrival_time_ts,
  
  -- Stop details from static stops table
  s.stop_name,
  s.stop_lat,
  s.stop_lon,
  s.location_type,
  s.parent_station,
  
  -- Calculate Day Type (Weekday vs Weekend) based on arrival time in NY timezone
  CASE
    WHEN EXTRACT(DAYOFWEEK FROM TIMESTAMP_SECONDS(sd.arrival_time) AT TIME ZONE 'America/New_York') IN (1, 7) THEN 'Weekend'
    ELSE 'Weekday'
  END AS day_type
FROM
  `{{ params.project_id }}.mta_raw.raw` AS sd
  JOIN
  `{{ params.project_id }}.mta_raw.stops` AS s
  ON sd.stop_id = s.stop_id;
