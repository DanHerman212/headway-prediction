-- =============================================================================
-- 01_create_tables.sql
-- =============================================================================
-- Creates all BigQuery tables for the MTA data pipeline.
-- Run this once to set up the schema.
--
-- Usage:
--   bq query --use_legacy_sql=false --project_id=YOUR_PROJECT < pipelines/sql/01_create_tables.sql
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Dataset: headway_dataset - Raw tables (source data loaded from GCS)
-- -----------------------------------------------------------------------------

-- Raw sensor data from subwaydata.nyc archives
-- Schema matches the CSV files from the archive
CREATE TABLE IF NOT EXISTS `{{ params.project_id }}.headway_dataset.raw` (
    trip_uid STRING,
    stop_id STRING,
    track STRING,
    arrival_time INT64,      -- Unix timestamp
    departure_time INT64,    -- Unix timestamp  
    last_observed INT64,     -- Unix timestamp
    marked_past INT64
)
OPTIONS (
    description = 'Raw subway sensor data from subwaydata.nyc archive'
);

-- GTFS Static: Stops
CREATE TABLE IF NOT EXISTS `{{ params.project_id }}.headway_dataset.stops` (
    stop_id STRING,
    stop_name STRING,
    stop_lat FLOAT64,
    stop_lon FLOAT64,
    location_type STRING,
    parent_station STRING
)
OPTIONS (
    description = 'GTFS stops.txt - station metadata'
);

-- GTFS Static: Routes
CREATE TABLE IF NOT EXISTS `{{ params.project_id }}.headway_dataset.routes` (
    route_id STRING,
    agency_id STRING,
    route_short_name STRING,
    route_long_name STRING,
    route_desc STRING,
    route_type INT64,
    route_url STRING,
    route_color STRING,
    route_text_color STRING
)
OPTIONS (
    description = 'GTFS routes.txt - route metadata'
);

-- Historic Schedules from NY Open Data
-- Source: https://data.ny.gov/Transportation/MTA-Subway-Schedules-Beginning-January-2025/q9nv-uegs
-- Note: Timestamp columns loaded as STRING due to AM/PM format in source CSV
CREATE TABLE IF NOT EXISTS `{{ params.project_id }}.headway_dataset.schedules` (
    service_date STRING,
    service_code STRING,
    train_id STRING,
    line STRING,
    trip_line STRING,
    direction STRING,
    stop_order INT64,
    gtfs_stop_id STRING,
    arrival_time STRING,
    departure_time STRING,
    date_difference INT64,
    track STRING,
    division STRING,
    revenue_service STRING,
    timepoint STRING,
    trip_type STRING,
    path_id STRING,
    next_trip_type STRING,
    next_trip_time STRING,
    supplement_schedule_number STRING,
    schedule_file_number STRING,
    origin_gtfs_stop_id STRING,
    destination_gtfs_stop_id STRING
)
OPTIONS (
    description = 'Historic subway schedules from NY Open Data'
);

-- Service Alerts from NY Open Data
-- Source: https://data.ny.gov/Transportation/MTA-Service-Alerts-Beginning-2025/7kct-peq7
-- Note: Date column loaded as STRING due to potential format issues
CREATE TABLE IF NOT EXISTS `{{ params.project_id }}.headway_dataset.alerts` (
    alert_id INT64,
    event_id INT64,
    update_number INT64,
    date STRING,
    agency STRING,
    status_label STRING,
    affected STRING,
    header STRING,
    description STRING
)
OPTIONS (
    description = 'MTA service alerts from NY Open Data'
);


-- -----------------------------------------------------------------------------
-- Dataset: headway_dataset - Transformed tables (cleaned and feature-engineered data)
-- -----------------------------------------------------------------------------

-- Cleaned arrivals with parsed trip metadata
CREATE TABLE IF NOT EXISTS `{{ params.project_id }}.headway_dataset.clean` (
    trip_uid STRING,
    start_time_dts TIMESTAMP,
    route_id STRING,
    direction STRING,
    path_identifier STRING,
    stop_id STRING,
    track STRING,
    arrival_time INT64,
    departure_time INT64,
    last_observed INT64,
    marked_past BOOL,
    arrival_time_ts TIMESTAMP,
    stop_name STRING,
    stop_lat FLOAT64,
    stop_lon FLOAT64,
    location_type STRING,
    parent_station STRING,
    day_type STRING
)
PARTITION BY DATE(start_time_dts)
OPTIONS (
    description = 'Cleaned subway data with parsed trip metadata, partitioned by day'
);

-- NOTE: ml table is created by 03_ml_headways_all_nodes.sql with CREATE OR REPLACE
