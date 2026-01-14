/*
  Incremental Data Cleansing Script (Weekly)
  
  Source: {{ params.project_id }}.headway_dataset.raw, {{ params.project_id }}.headway_dataset.stops
  Target: {{ params.project_id }}.headway_dataset.clean
  Partition Key: start_time_dts (DATE)
  
  Parameters (substituted via sed):
    {{ params.project_id }} - GCP project ID
    {{ params.start_date }} - Start of date range (YYYY-MM-DD)
    {{ params.end_date }} - End of date range (YYYY-MM-DD)
  
  Logic:
  1. Removes any existing data for the date range (Idempotency).
  2. Inserts processed data from 'raw' for the date range.
*/

-- 1. Idempotency: Delete data for the date range being processed
DELETE FROM `{{ params.project_id }}.headway_dataset.clean`
WHERE DATE(start_time_dts) BETWEEN DATE('{{ params.start_date }}') AND DATE('{{ params.end_date }}');

-- 2. Transformation & Insert
INSERT INTO `{{ params.project_id }}.headway_dataset.clean`
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
  `{{ params.project_id }}.headway_dataset.raw` AS sd
  JOIN
  `{{ params.project_id }}.headway_dataset.stops` AS s
  ON sd.stop_id = s.stop_id
WHERE
  -- Filter raw data to only process the target date range
  DATE(TIMESTAMP_SECONDS(CAST(SUBSTR(sd.trip_uid, 1, STRPOS(sd.trip_uid, '_') - 1) AS INT64))) 
    BETWEEN DATE('{{ params.start_date }}') AND DATE('{{ params.end_date }}');
