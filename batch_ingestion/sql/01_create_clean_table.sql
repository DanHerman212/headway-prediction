-- ============================================
-- Script: 01_create_clean_table.sql
-- Purpose: Create cleaned table from raw data
-- ============================================
-- Transformations:
--   1. Extract date_timestamp from trip_uid (text left of _)
--   2. Extract route_id (first character after _)
--   3. Extract direction (first character after ..)
--   4. Convert unix timestamps to UTC for:
--      - arrival_time
--      - departure_time  
--      - last_observed
--      - marked_past
--   5. Join to stops table for stop metadata
-- ============================================

CREATE OR REPLACE TABLE `${PROJECT_ID}.${BQ_DATASET}.clean` AS
SELECT
    -- Original columns
    r.trip_uid,
    r.stop_id,
    r.track,
    
    -- Stop metadata from stops table
    s.stop_name,
    s.stop_lat,
    s.stop_lon,
    s.parent_station,
    
    -- Extracted: date_timestamp (unix timestamp left of _)
    CAST(SPLIT(r.trip_uid, '_')[SAFE_OFFSET(0)] AS INT64) AS date_timestamp,
    
    -- Extracted: date as timestamp type
    TIMESTAMP_SECONDS(CAST(SPLIT(r.trip_uid, '_')[SAFE_OFFSET(0)] AS INT64)) AS trip_date,
    
    -- Extracted: route_id (first character after _)
    SUBSTR(SPLIT(r.trip_uid, '_')[SAFE_OFFSET(1)], 1, 1) AS route_id,
    
    -- Extracted: direction (first character after ..)
    SUBSTR(SPLIT(r.trip_uid, '..')[SAFE_OFFSET(1)], 1, 1) AS direction,
    
    -- Converted timestamps (UTC)
    TIMESTAMP_SECONDS(r.arrival_time) AS arrival_time,
    TIMESTAMP_SECONDS(r.departure_time) AS departure_time,
    TIMESTAMP_SECONDS(r.last_observed) AS last_observed,
    TIMESTAMP_SECONDS(r.marked_past) AS marked_past

FROM `${PROJECT_ID}.${BQ_DATASET}.raw` r
LEFT JOIN `${PROJECT_ID}.${BQ_DATASET}.stops` s
    ON r.stop_id = s.stop_id;
