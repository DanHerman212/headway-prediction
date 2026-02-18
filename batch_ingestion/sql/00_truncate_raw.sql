-- ============================================
-- Script: 00_truncate_raw.sql
-- Purpose: Clear the raw table before a fresh data load
-- ============================================
-- Run this BEFORE load_to_bigquery_monthly.py to prevent
-- duplicate rows when reloading the full date range.
-- TRUNCATE preserves the table schema but removes all data.
-- ============================================

TRUNCATE TABLE `${PROJECT_ID}.${BQ_DATASET}.raw`;
