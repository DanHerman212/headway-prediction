/*
  Incremental Data Cleansing Script
  
  Source: {{ params.project_id }}.mta_raw.raw, {{ params.project_id }}.mta_raw.stops
  Target: {{ params.project_id }}.mta_transformed.clean
  Partition Key: start_time_dts (DATE)
  
  Logic:
  1. Removes any existing data for the execution date (Idempotency).
  2. Inserts processed data from 'raw' for the specific execution date.
*/

-- 1. Idempotency: Delete data for the specific date being processed
DELETE FROM `{{ params.project_id }}.mta_transformed.clean`
WHERE DATE(start_time_dts) = DATE('{{ ds }}');

-- 2. Transformation & Insert
INSERT INTO `{{ params.project_id }}.mta_transformed.clean`
SELECT
  sd.trip_uid,
  TIMESTAMP_SECONDS(CAST(SUBSTR(sd.trip_uid, 1, STRPOS(sd.trip_uid, '_') - 1) AS INT64)) AS start_time_dts,
  CASE
    WHEN STRPOS(SUBSTR(sd.trip_uid, STRPOS(sd.trip_uid, '_') + 1), '..') > 0 THEN SUBSTR(SUBSTR(sd.trip_uid,
        STRPOS(sd.trip_uid, '_') + 1), 1, STRPOS(SUBSTR(sd.trip_uid, STRPOS(sd.trip_uid, '_') + 1), '..') - 1)
    ELSE SUBSTR(sd.trip_uid, STRPOS(sd.trip_uid, '_') + 1)
  END AS route_id,
  CASE
    WHEN STRPOS(SUBSTR(sd.trip_uid, STRPOS(sd.trip_uid, '_') + 1), '..') > 0 THEN SUBSTR(SUBSTR(sd.trip_uid,
        STRPOS(sd.trip_uid, '_') + 1), STRPOS(SUBSTR(sd.trip_uid, STRPOS(sd.trip_uid, '_') + 1), '..') + 2, 1)
    ELSE NULL
  END AS direction,
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
  TIMESTAMP_SECONDS(sd.arrival_time) AS arrival_time_ts,
  s.stop_name,
  s.stop_lat,
  s.stop_lon,
  s.location_type,
  s.parent_station,
  CASE
    WHEN EXTRACT(DAYOFWEEK FROM TIMESTAMP_SECONDS(sd.arrival_time) AT TIME ZONE 'America/New_York') IN (1, 7) THEN 'Weekend'
    ELSE 'Weekday'
  END AS day_type
FROM
  `{{ params.project_id }}.mta_raw.raw` AS sd
  JOIN
  `{{ params.project_id }}.mta_raw.stops` AS s
  ON sd.stop_id = s.stop_id
WHERE
  -- Filter raw data to only process the target date
  DATE(TIMESTAMP_SECONDS(CAST(SUBSTR(sd.trip_uid, 1, STRPOS(sd.trip_uid, '_') - 1) AS INT64))) = DATE('{{ ds }}');
