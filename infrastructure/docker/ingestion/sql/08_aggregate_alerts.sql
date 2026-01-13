/*
  Aggregate Alerts to 5-Minute Time Bins
  
  Aggregates service alerts into 5-minute time bins per route.
  Creates binary flags indicating if an alert was active for A/C/E.
  
  Source: {{ params.project_id }}.mta_raw.alerts
  Target: {{ params.project_id }}.mta_transformed.alerts_binned
*/
CREATE OR REPLACE TABLE `{{ params.project_id }}.mta_transformed.alerts_binned` AS

WITH alerts_parsed AS (
  SELECT
    alert_id,
    event_id,
    alert_date,
    status_label,
    affected,
    header,
    -- Parse affected routes (handles "A", "A | C", "A | C | E" formats)
    CASE WHEN REGEXP_CONTAINS(affected, r'\bA\b') THEN TRUE ELSE FALSE END AS affects_a,
    CASE WHEN REGEXP_CONTAINS(affected, r'\bC\b') THEN TRUE ELSE FALSE END AS affects_c,
    CASE WHEN REGEXP_CONTAINS(affected, r'\bE\b') THEN TRUE ELSE FALSE END AS affects_e,
    -- Categorize alert severity
    CASE
      WHEN status_label IN ('delays', 'some-delays') THEN 'delay'
      WHEN status_label IN ('part-suspended', 'suspended') THEN 'suspended'
      WHEN status_label IN ('slow-speeds') THEN 'slow'
      WHEN status_label IN ('stops-skipped', 'trains-rerouted') THEN 'modified'
      ELSE 'other'
    END AS alert_category
  FROM `{{ params.project_id }}.mta_raw.alerts`
  WHERE 
    agency = 'NYCT Subway'
    -- Only A/C/E related alerts
    AND (
      REGEXP_CONTAINS(affected, r'\bA\b') 
      OR REGEXP_CONTAINS(affected, r'\bC\b') 
      OR REGEXP_CONTAINS(affected, r'\bE\b')
    )
),

-- Floor to 5-minute bins
time_bins AS (
  SELECT
    alert_id,
    event_id,
    -- Floor to 5-minute bin
    TIMESTAMP_SECONDS(
      UNIX_SECONDS(TIMESTAMP_TRUNC(alert_date, MINUTE)) 
      - MOD(UNIX_SECONDS(TIMESTAMP_TRUNC(alert_date, MINUTE)), 300)
    ) AS time_bin,
    affects_a,
    affects_c,
    affects_e,
    alert_category,
    status_label,
    header
  FROM alerts_parsed
  WHERE alert_date IS NOT NULL
)

SELECT
  time_bin,
  -- Route-level alert flags
  MAX(CASE WHEN affects_a THEN 1 ELSE 0 END) AS alert_a,
  MAX(CASE WHEN affects_c THEN 1 ELSE 0 END) AS alert_c,
  MAX(CASE WHEN affects_e THEN 1 ELSE 0 END) AS alert_e,
  -- Any A/C/E alert
  1 AS alert_ace,
  -- Most severe alert category
  MAX(alert_category) AS alert_category,
  -- Count of active alerts
  COUNT(DISTINCT event_id) AS alert_count
FROM time_bins
GROUP BY time_bin
ORDER BY time_bin;
