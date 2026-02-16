# Streaming Pipeline Integration Test — Follow-Up Items

## Current State (Feb 16, 2026)

The integration test runs end-to-end: VM poller → Pub/Sub → Beam DirectRunner → Firestore. Arrival detection, enrichment, and stateful transforms (headway, travel time, track gap) compute real values from live feed diffs. Two side inputs are hardcoded with defaults.

---

## Required Follow-Ups

### 1. Build empirical lookup tables from historical data — CRITICAL
The streaming pipeline uses two side-input maps that are currently empty dicts. **These are not optional placeholders — `empirical_median` is one of the most important predictors in the model.** Running the pipeline without real values produces misleading output.

- **`empirical_map`** — keyed by `(route_id, day_type, hour)`, value is `median_headway` (float). Currently hardcoded to `8.0` for all records. **This must be computed from historical data — it captures the expected headway for a given route/time and is a primary feature the model relies on.**
- **`median_tt_map`** — keyed by `(route_id, day_type, hour, origin_station_id)`, value is `median_travel_time` (float). Currently defaults to `0.0` deviation for all records.

**Work:** Write a batch job (or notebook) that aggregates historical processed data to compute these two maps, serialize as JSON, and upload to GCS. This is the first task — nothing else should proceed until these are real.

### 2. Load side inputs from GCS in the streaming pipeline
Replace the two `{}` placeholders in `streaming_pipeline.py` with real GCS reads:
```python
# Current (hardcoded)
empirical_map={}
median_tt_map={}

# Target
empirical_map=beam.pvalue.AsSingleton(load_json_from_gcs("gs://..."))
median_tt_map=beam.pvalue.AsSingleton(load_json_from_gcs("gs://..."))
```

### 3. Validate feature values against batch pipeline output
Compare the features logged in the target station feedback against known-good output from the batch pipeline (`generate_dataset.py`) to confirm the streaming transforms produce equivalent results. Key fields to check:
- `service_headway`
- `upstream_headway_14th`
- `travel_time_14th`, `travel_time_23rd`, `travel_time_34th`
- `preceding_train_gap`
- `travel_time_deviation_*` (requires item #1)
- `empirical_median` (requires item #1)

### 4. Wire in the prediction endpoint
Once features are validated, add a step between the window buffer and Firestore that sends the 20-observation window to the Vertex AI endpoint:
```
BufferWindowFn → call endpoint → log prediction → WriteToFirestore
```
The target station log should then also show `headway_p10`, `headway_p50`, `headway_p90`.

### 5. Foreground pipeline output
Currently `make up` backgrounds the pipeline and logs to a file. Consider running the pipeline in the foreground so target station events print directly to the terminal without requiring `make logs`.

### 6. Make `make up` fully idempotent for resume
If the test is torn down and restarted, the stateful DoFns (arrival detector, headway, travel time, buffer) start cold — the first few events will have `None` for headway/travel time until state warms up. This is expected but should be documented or handled with a "warming up" indicator in the feedback log.

---

## Not Required for Test Validation (Future)
- Deploy pipeline on Dataflow (production runner) — current DirectRunner is correct for integration testing
- BDFM feed support — out of scope, ACE-only is the target
- Alerts feed — not consumed by the pipeline
