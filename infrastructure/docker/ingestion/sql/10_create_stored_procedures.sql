-- =============================================================================
-- 10_create_stored_procedures.sql
-- =============================================================================
-- Creates stored procedures for incremental weekly transforms.
-- Run this once after initial backfill to set up procedures for weekly updates.
--
-- Usage:
--   bq query --use_legacy_sql=false --project_id=YOUR_PROJECT < pipelines/sql/10_create_stored_procedures.sql
-- =============================================================================


-- -----------------------------------------------------------------------------
-- Procedure: sp_clean_arrivals_incremental
-- -----------------------------------------------------------------------------
-- Cleans newly loaded arrival data and appends to the clean table.
-- Only processes records not yet in clean table (based on arrival_time).
-- -----------------------------------------------------------------------------

CREATE OR REPLACE PROCEDURE `{{ params.project_id }}.mta_transformed.sp_clean_arrivals_incremental`()
BEGIN
    DECLARE max_processed_time INT64;
    
    -- Get the latest arrival_time already processed
    SET max_processed_time = (
        SELECT COALESCE(MAX(UNIX_SECONDS(arrival_time_ts)), 0)
        FROM `{{ params.project_id }}.mta_transformed.clean`
    );
    
    -- Insert new records only
    INSERT INTO `{{ params.project_id }}.mta_transformed.clean` (
        trip_uid,
        start_time_dts,
        route_id,
        direction,
        path_identifier,
        stop_id,
        track,
        arrival_time,
        departure_time,
        last_observed,
        marked_past,
        arrival_time_ts,
        stop_name,
        stop_lat,
        stop_lon,
        location_type,
        parent_station,
        day_type
    )
    SELECT
        sd.trip_uid,
        TIMESTAMP_SECONDS(CAST(SUBSTR(sd.trip_uid, 1, STRPOS(sd.trip_uid, '_') - 1) AS INT64)) AS start_time_dts,
        
        CASE
            WHEN STRPOS(SUBSTR(sd.trip_uid, STRPOS(sd.trip_uid, '_') + 1), '..') > 0 
            THEN SUBSTR(SUBSTR(sd.trip_uid, STRPOS(sd.trip_uid, '_') + 1), 1, 
                        STRPOS(SUBSTR(sd.trip_uid, STRPOS(sd.trip_uid, '_') + 1), '..') - 1)
            ELSE SUBSTR(sd.trip_uid, STRPOS(sd.trip_uid, '_') + 1)
        END AS route_id,
        
        CASE
            WHEN STRPOS(SUBSTR(sd.trip_uid, STRPOS(sd.trip_uid, '_') + 1), '..') > 0 
            THEN SUBSTR(SUBSTR(sd.trip_uid, STRPOS(sd.trip_uid, '_') + 1), 
                        STRPOS(SUBSTR(sd.trip_uid, STRPOS(sd.trip_uid, '_') + 1), '..') + 2, 1)
            ELSE NULL
        END AS direction,
        
        CASE
            WHEN STRPOS(SUBSTR(sd.trip_uid, STRPOS(sd.trip_uid, '_') + 1), '..') > 0 
            THEN SUBSTR(SUBSTR(sd.trip_uid, STRPOS(sd.trip_uid, '_') + 1), 
                        STRPOS(SUBSTR(sd.trip_uid, STRPOS(sd.trip_uid, '_') + 1), '..') + 3)
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
            WHEN EXTRACT(DAYOFWEEK FROM TIMESTAMP_SECONDS(sd.arrival_time) AT TIME ZONE 'America/New_York') IN (1, 7) 
            THEN 'Weekend'
            ELSE 'Weekday'
        END AS day_type
        
    FROM `{{ params.project_id }}.mta_raw.raw` AS sd
    JOIN `{{ params.project_id }}.mta_raw.stops` AS s ON sd.stop_id = s.stop_id
    WHERE sd.arrival_time > max_processed_time;
    
END;


-- -----------------------------------------------------------------------------
-- Procedure: sp_compute_headways_incremental
-- -----------------------------------------------------------------------------
-- Computes headways for newly cleaned data and appends to headways table.
-- Uses 2-day lookback window to ensure LAG() works correctly for first 
-- arrivals of each new day.
-- -----------------------------------------------------------------------------

CREATE OR REPLACE PROCEDURE `{{ params.project_id }}.mta_transformed.sp_compute_headways_incremental`()
BEGIN
    DECLARE max_processed_time TIMESTAMP;
    DECLARE lookback_start TIMESTAMP;
    
    -- Get the latest arrival already processed in headways table
    SET max_processed_time = (
        SELECT COALESCE(MAX(arrival_time_ts), TIMESTAMP('1970-01-01'))
        FROM `{{ params.project_id }}.mta_transformed.headways`
    );
    
    -- Lookback 2 days to ensure LAG() works for first train of new day
    SET lookback_start = TIMESTAMP_SUB(max_processed_time, INTERVAL 2 DAY);
    
    -- Compute headways for new data with lookback context
    INSERT INTO `{{ params.project_id }}.mta_transformed.headways` (
        node_id,
        stop_id,
        route_id,
        direction,
        stop_name,
        stop_lat,
        stop_lon,
        arrival_time_ts,
        prev_arrival_time_ts,
        headway_seconds,
        headway_minutes,
        day_type,
        hour_of_day,
        day_of_week
    )
    WITH arrivals_with_context AS (
        SELECT
            CONCAT(stop_id, '_', route_id, '_', direction) AS node_id,
            stop_id,
            route_id,
            direction,
            stop_name,
            stop_lat,
            stop_lon,
            arrival_time_ts,
            day_type,
            LAG(arrival_time_ts) OVER (
                PARTITION BY stop_id, route_id, direction 
                ORDER BY arrival_time_ts
            ) AS prev_arrival_time_ts
        FROM `{{ params.project_id }}.mta_transformed.clean`
        WHERE route_id IN ('A', 'C', 'E')
          AND direction IN ('N', 'S')
          AND arrival_time_ts >= lookback_start
    )
    SELECT
        node_id,
        stop_id,
        route_id,
        direction,
        stop_name,
        stop_lat,
        stop_lon,
        arrival_time_ts,
        prev_arrival_time_ts,
        TIMESTAMP_DIFF(arrival_time_ts, prev_arrival_time_ts, SECOND) AS headway_seconds,
        ROUND(TIMESTAMP_DIFF(arrival_time_ts, prev_arrival_time_ts, SECOND) / 60.0, 2) AS headway_minutes,
        day_type,
        EXTRACT(HOUR FROM arrival_time_ts AT TIME ZONE 'America/New_York') AS hour_of_day,
        EXTRACT(DAYOFWEEK FROM arrival_time_ts AT TIME ZONE 'America/New_York') AS day_of_week
    FROM arrivals_with_context
    WHERE prev_arrival_time_ts IS NOT NULL
      AND arrival_time_ts > max_processed_time
      -- Filter reasonable headways (1 min to 2 hours)
      AND TIMESTAMP_DIFF(arrival_time_ts, prev_arrival_time_ts, SECOND) BETWEEN 60 AND 7200;
    
END;
