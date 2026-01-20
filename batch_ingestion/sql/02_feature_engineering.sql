/* step 1: minimal column selection for downstream tasks */
WITH base_data AS(
    SELECT
        route_id,
        direction,
        stop_id,
        track,
        stop_name,
        arrival_time,
        trip_date
    FROM `${BQ_DATASET}.clean`
    WHERE
        stop_id = 'A32S'
        AND track IN ('A1','A3')
        AND EXTRACT(DATE FROM arrival_time) BETWEEN '2025-07-18' AND '2026-01-19'
        AND arrival_time IS NOT NULL
),

/* step 2: compute headway */
headway_features AS(
    SELECT
    *, /* carry all columns from base_data */
   TIMESTAMP_DIFF(
    arrival_time, LAG(arrival_time) OVER (
        PARTITION BY stop_id, track
        ORDER BY arrival_time
    ),
    SECOND
   ) / 60.0 as headway
FROM base_data
),

/* step 3: extract temporal features (raw) */
temporal_features AS(
SELECT
    *,
    /* Time of day in seconds since midnight */
    EXTRACT(HOUR FROM arrival_time AT TIME ZONE 'America/New_York') * 3600
    + EXTRACT(MINUTE FROM arrival_time AT TIME ZONE 'America/New_York') * 60
    + EXTRACT(SECOND FROM arrival_time AT TIME ZONE 'America/New_York') AS time_of_day_seconds,

    /* Hour of day (0 -23, NYC time) */
    EXTRACT(HOUR FROM arrival_time AT TIME ZONE 'America/New_York') AS hour_of_day,

    /* Day of week (1=Sunday, 7=Saturday) */
    EXTRACT(DAYOFWEEK FROM arrival_time AT TIME ZONE 'America/New_York') AS day_of_week
FROM headway_features /* Corrected: Added FROM clause here */
)

/* final select statement - all columns for cleaner viewing */
SELECT
    route_id,
    direction,
    stop_id,
    track,
    stop_name,
    arrival_time,
    trip_date,
    headway,
    time_of_day_seconds,
    hour_of_day,
    day_of_week
FROM temporal_features
ORDER BY track, arrival_time;
