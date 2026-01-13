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
-- Dataset: mta_raw (source data loaded from GCS)
-- -----------------------------------------------------------------------------

-- Raw sensor data from subwaydata.nyc archives
-- Schema matches the CSV files from the archive
CREATE TABLE IF NOT EXISTS `mta_raw.raw` (
    trip_uid STRING,
    stop_id STRING,
    track STRING,
    arrival_time INT64,      -- Unix timestamp
    departure_time INT64,    -- Unix timestamp  
    last_observed INT64,     -- Unix timestamp
    marked_past BOOL
)
OPTIONS (
    description = 'Raw subway sensor data from subwaydata.nyc archive'
);

-- GTFS Static: Stops
CREATE TABLE IF NOT EXISTS `mta_raw.stops` (
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
CREATE TABLE IF NOT EXISTS `mta_raw.routes` (
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

-- Historic Schedules (structure TBD based on actual file)
CREATE TABLE IF NOT EXISTS `mta_raw.schedules` (
    trip_id STRING,
    stop_id STRING,
    arrival_time STRING,
    departure_time STRING,
    stop_sequence INT64
)
OPTIONS (
    description = 'Historic subway schedules'
);

-- Service Alerts from NY Open Data
CREATE TABLE IF NOT EXISTS `mta_raw.alerts` (
    alert_id INT64,
    event_id INT64,
    update_number INT64,
    alert_date TIMESTAMP,
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
-- Dataset: mta_transformed (cleaned and feature-engineered data)
-- -----------------------------------------------------------------------------

-- Cleaned arrivals with parsed trip metadata
CREATE TABLE IF NOT EXISTS `mta_transformed.clean` (
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

-- Headways for all A/C/E nodes
CREATE TABLE IF NOT EXISTS `mta_transformed.headways_all_nodes` (
    trip_uid STRING,
    node_id STRING,
    route_id STRING,
    direction STRING,
    stop_id STRING,
    stop_name STRING,
    stop_lat FLOAT64,
    stop_lon FLOAT64,
    arrival_time TIMESTAMP,
    prev_arrival_time TIMESTAMP,
    service_date DATE,
    headway_minutes FLOAT64,
    hour_of_day INT64,
    minute_of_hour INT64,
    day_of_week INT64,
    minute_of_day INT64,
    day_type STRING,
    is_peak_hour INT64
)
PARTITION BY service_date
CLUSTER BY route_id, direction, stop_id
OPTIONS (
    description = 'Headways for all A/C/E nodes for Graph WaveNet training'
);

-- Alerts aggregated to 5-minute bins
CREATE TABLE IF NOT EXISTS `mta_transformed.alerts_binned` (
    time_bin TIMESTAMP,
    alert_a INT64,
    alert_c INT64,
    alert_e INT64,
    alert_ace INT64,
    alert_category STRING,
    alert_count INT64
)
OPTIONS (
    description = 'Service alerts aggregated to 5-minute time bins per route'
);

-- Node mapping for Graph WaveNet
CREATE TABLE IF NOT EXISTS `mta_transformed.node_mapping` (
    node_id STRING,
    node_index INT64,
    route_id STRING,
    direction STRING,
    stop_id STRING,
    stop_name STRING,
    stop_lat FLOAT64,
    stop_lon FLOAT64
)
OPTIONS (
    description = 'Mapping from node_id to integer index for tensor construction'
);
