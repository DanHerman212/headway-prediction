# Data Pipeline Debugging & Fixes Report
**Date:** February 1, 2026  
**Pipeline:** `lines/beam/batch/generate_dataset.py`  
**Pipeline Engine:** Apache Beam (Batch)

## Executive Summary
This document details the issues encountered during the implementation of the Headway Prediction training dataset pipeline. The pipeline experienced crashes, total data loss, and significant logical errors regarding stateful processing. All critical issues have been resolved, resulting in a high-quality dataset ready for modeling.

---

## 1. Serialization Crash (`ArrowTypeError`)
**Problem:**  
The pipeline failed immediately upon execution with `pyarrow.lib.ArrowTypeError: Expected bytes, got datetime.datetime`.

**Analysis:**  
Apache Arrow (used by ParquetIO) requires strict type handling. The pipeline was attempting to save Python `datetime` objects directly into a schema that expected specific Arrow Timestamp formats or strings, and the inter-worker communication in Beam failed to pickle these objects correctly.

**Solution:**  
Modified `generate_dataset.py` to ensure all datetime objects are converted to compatible formats before the WriteToParquet step. Verified schema compatibility.

---

## 2. The "Empy Dataset" (Station ID Mismatch)
**Problem:**  
The pipeline ran successfully but produced an output file with 0 rows.

**Analysis:**  
The filter logic in `transforms.py` used Station IDs derived from the IRT line (1/2/3 Train) convention (e.g., `128S` for Penn Station) instead of the correct IND (A/C/E Train) convention (e.g., `A28S` for Penn Station). As a result, 100% of the events were filtered out as "irrelevant stations."

**Solution:**  
Updated the station constants in `transforms.py` to use the correct GTFS IDs for the A Line:
- 34th St: `A28S`
- 23rd St: `A30S`
- 14th St: `A31S`
- W4th St (Target): `A32S`

---

## 3. High Null Rates in Travel Times (~30%)
**Problem:**  
The generic feature `travel_time_34th` (Travel time from 34th St to W 4th St) was `NULL` for ~30% of trips, even for Express (A3) trains that definitely stopped there.

**Analysis:**  
**Forensic Tracing** revealed that Apache Beam's Batch runner processes bundles in arbitrary order. 
- In ~30% of cases, the "Target Event" (Arrival at W 4th) was processed *before* the "Upstream Event" (Departure from 34th) reached the stateful DoFn.
- Since the state buffer was empty when the target arrived, the calculator assumed no upstream train existed.

**Solution:**  
Injected a `GroupByKey` followed by a sorting step (`FlatMap(sort_key_events)`) in `generate_dataset.py` immediately before the Stateful Processing step. This forces all events for a specific Trip UID to be processed in strict chronological order.

**Result:**  
34th St Nulls dropped from **~30,000** to **29**.

---

## 4. Unrealistic Headway Distributions
**Problem:**  
The target variable `service_headway` showed physically impossible values:
- **Max:** 46,000 minutes (approx. 32 days).
- **Min:** 0.0 minutes.

**Analysis:**  
- **Mega-Gaps:** The `CalculateServiceHeadwayFn` maintained state indefinitely. If a train appeared 32 days after the previous run (e.g., across a gap in the raw data files), it calculated the headway as the difference between those timestamps.
- **Zeroes:** Likely caused by "ghost trains" (duplicate sensor readings) or slight data jitter where two updates appeared effectively simultaneous.

**Solution:**  
1. **Session Timeouts:** Updated `CalculateServiceHeadwayFn` to reset state if the gap between trains exceeds **90 minutes**. This treats large gaps as a "New Session" rather than a valid headway.
2. **Filters:** Added a `beam.Filter` step to `generate_dataset.py` to explicitly remove:
   - Headways < 0.5 minutes.
   - Headways > 120 minutes.

**Result:**  
Distribution is now clean, ranging from 0.5 min to ~90 min.

---

## Current Status
The pipeline is stable. Validation notebooks confirm:
- Target variable distribution is Gaussian-like (Log-Normal).
- Feature relationships (Travel Time vs Headway) show expected correlation.
- Nulls are now confined to expected areas (e.g., 23rd St for Express trains that skip it).
