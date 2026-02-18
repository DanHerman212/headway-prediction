-- ============================================
-- Script: 01_create_clean_table.sql
-- Purpose: Create cleaned table from raw data
-- ============================================
-- Produces the 7-column schema expected by
-- generate_dataset.py (Beam batch pipeline):
--   trip_uid, stop_id, track, trip_date,
--   route_id, direction, arrival_time
--
-- Transformations:
--   1. Extract route_id (first character after _)
--   2. Extract direction (first character after ..)
--   3. Derive trip_date from trip_uid epoch prefix
--   4. Convert arrival_time from unix seconds to TIMESTAMP
--   5. Filter out rows with null arrival_time
-- ============================================

CREATE OR REPLACE TABLE `${PROJECT_ID}.${BQ_DATASET}.clean` AS
SELECT
    r.trip_uid,
    r.stop_id,
    r.track,

    -- trip_date: epoch prefix of trip_uid
    TIMESTAMP_SECONDS(CAST(SPLIT(r.trip_uid, '_')[SAFE_OFFSET(0)] AS INT64)) AS trip_date,

    -- route_id: first character after first underscore
    SUBSTR(SPLIT(r.trip_uid, '_')[SAFE_OFFSET(1)], 1, 1) AS route_id,

    -- direction: first character after '..'
    SUBSTR(SPLIT(r.trip_uid, '..')[SAFE_OFFSET(1)], 1, 1) AS direction,

    -- arrival_time: unix seconds → TIMESTAMP
    TIMESTAMP_SECONDS(r.arrival_time) AS arrival_time

FROM `${PROJECT_ID}.${BQ_DATASET}.raw` r
WHERE r.arrival_time IS NOT NULL;
