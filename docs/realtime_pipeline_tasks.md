# Realtime Inference Pipeline — Task Plan

## Overview

The goal is to build a streaming pipeline that accepts live GTFS-RT feed data and produces headway predictions in real time using the deployed TFT ONNX model. Four workstreams must be addressed to bridge the gap between raw feed data and model-ready input tensors.

---

## Current State

| Layer | What Exists | Where |
|-------|------------|-------|
| **Raw polling** | GTFS-RT poller code (to be imported) | External — user will transfer |
| **Feature engineering** | Apache Beam stateful transforms | `pipelines/beam/shared/transforms.py` |
| **Batch orchestration** | Beam batch pipeline (BQ → Parquet) | `pipelines/beam/batch/generate_dataset.py` |
| **ML preprocessing** | Pandas cleaning + imputation | `mlops_pipeline/src/data_processing.py` |
| **Model serving** | ONNX endpoint on Vertex AI | `mlops_pipeline/src/serving/` |

### Feature Inventory

The model expects these inputs (from `processing/default.yaml`):

| Category | Features | Created By |
|----------|----------|------------|
| Static categorical | `route_id` | Beam `EnrichRecordFn` |
| Time-varying known categorical | `regime_id`, `track_id` | Beam `EnrichRecordFn` |
| Time-varying known real | `time_idx`, `hour_sin`, `hour_cos`, `empirical_median` | Beam `EnrichRecordFn` + `EnrichWithEmpiricalFn` |
| Time-varying unknown categorical | `preceding_route_id` | Beam `CalculateTrackGapFn` |
| Time-varying unknown real | `service_headway`, `preceding_train_gap`, `upstream_headway_14th`, `travel_time_14th`, `travel_time_14th_deviation`, `travel_time_23rd`, `travel_time_23rd_deviation`, `travel_time_34th`, `travel_time_34th_deviation`, `stops_at_23rd` | Mixed — Beam transforms + ML pipeline imputation |

### The Gap

The ML pipeline's `clean_dataset()` performs transforms that do **not** exist in the Beam layer:

1. **Imputation** — `preceding_train_gap`, `upstream_headway_14th`, `travel_time_14th`, `travel_time_34th` filled with training-set median; deviation columns filled with `0.0`.
2. **`stops_at_23rd` flag** — binary feature derived from `travel_time_23rd` nullability; nulls filled with training-set mean.
3. **`time_idx` recalculation** — recomputed as absolute minutes from global `min(arrival_time_dt)`, replacing the Beam-generated sequential index.
4. **`empirical_median` side-input** — in batch Beam this is computed over the training split and broadcast. In streaming, this lookup table must be pre-materialized and served.

---

## Task 1 — GTFS-RT Poller

**Objective:** Poll the MTA GTFS-RT endpoint and emit structured arrival events.

**What the raw feed provides** (per `stop_time_update` in the JSON):

```
trip_id, route_id, start_time, start_date, stop_id, arrival.time, departure.time
```

**Actions:**
- [ ] Import poller code into `pipelines/beam/streaming/` (or a standalone `ingestion/` module)
- [ ] Define the canonical raw event schema: `{trip_uid, route_id, direction, stop_id, track, arrival_time}`
- [ ] Derive `direction` from `stop_id` suffix (`N`/`S`) and `track` from stop-to-track mapping (A1 = local, A3 = express for southbound 8th Ave)
- [ ] Determine polling cadence (likely 30s, matching MTA refresh rate)
- [ ] Output: raw events published to Pub/Sub topic (one message per stop_time_update at relevant stations)

**Key Decision:** The poller should filter to relevant stations early (A28S–A32S + northbound equivalents if needed) to reduce downstream volume.

---

## Task 2 — Raw-to-Baseline Transform

**Objective:** Convert raw GTFS-RT events into the baseline record format that Beam transforms expect.

The Beam `EnrichRecordFn` expects a dict with at minimum: `{arrival_time, route_id, direction, stop_id, track, trip_uid}`.

**Actions:**
- [ ] Build a lightweight Pub/Sub → Beam ingestion stage that reads raw events and normalizes field names
- [ ] Map GTFS-RT `stop_id` suffix → `direction` (e.g., `A32S` → `S`)
- [ ] Map `stop_id` → `track` (requires a static lookup — A1/A3 for 8th Ave southbound local/express)
- [ ] Generate `trip_uid` from `trip_id + start_date` (must match the batch convention)
- [ ] Validate: reject malformed events (missing arrival time, unknown route_id)
- [ ] Output: cleaned dicts ready for `EnrichRecordFn`

**Key Decision:** This can be a simple `beam.DoFn` or a standalone Python function before Beam. Keep it minimal — the existing `EnrichRecordFn` handles the heavy lifting.

---

## Task 3 — Streaming Feature Engineering (Existing Beam Transforms)

**Objective:** Wire the existing stateful Beam transforms into a streaming Dataflow pipeline.

The batch pipeline currently runs these in sequence:
1. `EnrichRecordFn` — row-level features (time_idx, group_id, regime_id, cyclical time, day_of_week)
2. `CalculateUpstreamHeadwayFn` — headway at 14th St (stateful, keyed by group_id)
3. `CalculateUpstreamTravelTimeFn` — travel times from 34th/23rd/14th (stateful, keyed by trip_uid)
4. `CalculateServiceHeadwayFn` — target headway at W 4th (stateful, keyed by group_id)
5. `CalculateTrackGapFn` — preceding train gap + route (stateful, keyed by track_id)
6. `EnrichWithEmpiricalFn` — empirical median (side input lookup)
7. `CalculateTravelTimeDeviationFn` — deviation from median travel time (side input lookup)

**Actions:**
- [ ] Port the batch pipeline DAG to streaming mode: replace `GroupByKey` + sort with native stateful processing (most transforms already use `ReadModifyWriteStateSpec` — they are streaming-ready)
- [ ] Replace `beam.io.ReadFromBigQuery` source with `beam.io.ReadFromPubSub`
- [ ] Handle the side-input tables (`empirical_median`, `median_travel_time`) — these cannot be computed on-the-fly in streaming. Options:
  - **(Preferred)** Pre-compute from training data and store in BigQuery/GCS; load as a `beam.pvalue.AsSingleton` or `AsDict` from a slowly-updating side input
  - Alternatively, use a Bigtable/Redis lookup
- [ ] Handle `ReindexTimeInGroupsFn` — in streaming, `time_idx` should be absolute minutes from a fixed epoch (matching `clean_dataset`'s approach), NOT sequential. The batch re-indexing step should be **skipped** in streaming.
- [ ] Replace Parquet sink with the inference call (or a Pub/Sub topic feeding the endpoint)

**Key Decision:** The stateful DoFns are already designed for streaming. The main porting work is (a) side-input materialization and (b) removing the batch-specific GroupByKey + sort wrappers.

---

## Task 4 — Upstream ML Transforms (Imputation + Derived Features)

**Objective:** Replicate the transforms from `clean_dataset()` that happen *after* Beam but *before* model input.

These are currently in `mlops_pipeline/src/data_processing.py` and run as Pandas operations inside the ZenML pipeline. They must be replicated in the streaming path.

**Specific transforms to replicate:**

| Transform | Batch Implementation | Streaming Approach |
|-----------|---------------------|-------------------|
| Impute `preceding_train_gap` | `fillna(median)` from training set | Use pre-computed median as constant (store in model metadata or config) |
| Impute `upstream_headway_14th` | `fillna(median)` from training set | Same — constant from training stats |
| Impute `travel_time_14th/34th` | `fillna(median)` from training set | Same — constant from training stats |
| Impute deviation columns | `fillna(0.0)` | Hardcode `0.0` default in transform |
| `stops_at_23rd` flag | `1.0` if `travel_time_23rd > 0`, else `0.0`; fill nulls with training mean | Compute flag inline; use pre-computed mean as fallback |
| `time_idx` | Minutes from global `min(arrival_time_dt)` | Use a **fixed epoch** (e.g., training-set min time) stored in model metadata; compute `(event_ts - epoch) / 60` |
| Categorical fillna | `fillna("None")` for `regime_id`, `preceding_route_id`, etc. | Apply same default in streaming DoFn |

**Actions:**
- [ ] Extract training-set statistics (medians, means) during the training pipeline and persist them alongside the model (extend `dataset_params.json` or create `imputation_stats.json`)
- [ ] Create a new `ImputeAndFinalizeFn(beam.DoFn)` that:
  - Loads stats from a side input or config
  - Applies all imputation rules
  - Computes `stops_at_23rd`
  - Computes absolute `time_idx` from fixed epoch
  - Fills categoricals with "None" default
- [ ] Place this DoFn after the existing Beam feature transforms and before inference
- [ ] Add `regime_id` to the Beam `EnrichRecordFn` output (already exists — confirm it matches training encoding)

**Key Decision:** Training statistics must be versioned with the model. When a new model is deployed, the imputation constants update with it. Store them in the same GCS model artifact directory used by the deploy step.

---

## Architecture Diagram

```
MTA GTFS-RT Feed
       │
       ▼
  ┌─────────┐
  │  Poller  │  (Task 1)
  └────┬─────┘
       │ raw events
       ▼
  ┌──────────┐
  │ Pub/Sub  │  topic: gtfs-raw-events
  └────┬─────┘
       │
       ▼
  ┌───────────────────────────────────────────┐
  │         Dataflow Streaming Pipeline        │
  │                                            │
  │  ┌─────────────────┐                       │
  │  │ Raw→Baseline    │  (Task 2)             │
  │  │ normalize fields│                       │
  │  └───────┬─────────┘                       │
  │          │                                  │
  │  ┌───────▼─────────┐                       │
  │  │ EnrichRecordFn  │                       │
  │  │ + Stateful DoFns│  (Task 3)             │
  │  │ (existing Beam) │                       │
  │  └───────┬─────────┘                       │
  │          │  side inputs: empirical_median,  │
  │          │  median_travel_time (from BQ/GCS)│
  │  ┌───────▼─────────┐                       │
  │  │ImputeAndFinalize│  (Task 4)             │
  │  │ + stops_at_23rd │                       │
  │  │ + time_idx      │                       │
  │  └───────┬─────────┘                       │
  │          │  side input: imputation_stats    │
  │          │  (from model artifacts)          │
  │          ▼                                  │
  │  ┌───────────────┐                          │
  │  │ Build encoder │  assemble 20-step window │
  │  │ window + call │  + call ONNX endpoint    │
  │  │ Vertex AI     │                          │
  │  └───────┬───────┘                          │
  └──────────┼──────────────────────────────────┘
             │
             ▼
        Predictions
     (Pub/Sub / BQ / Dashboard)
```

---

## Open Questions

1. **Encoder window buffer** — The model requires 20 historical observations per group to form the encoder input. In streaming, we need a stateful buffer (or a lookup to recent history in Bigtable/BQ) to accumulate the window before calling the endpoint.
2. **Cold start** — When the pipeline starts (or after a long gap), groups won't have 20 observations yet. Define a policy: skip prediction until buffer is full, or pad with training-set defaults.
3. **Prediction sink** — Where do predictions go? Options: Pub/Sub topic, BigQuery table, real-time dashboard, or all three.
4. **Side-input refresh** — How frequently should `empirical_median` and `median_travel_time` maps refresh? Daily is likely sufficient (recompute nightly from BQ).
5. **Monitoring** — Dataflow job metrics, prediction latency, data drift detection.

---

## Execution Order

| Priority | Task | Depends On | Estimated Effort |
|----------|------|-----------|-----------------|
| 1 | Task 1 — Poller + Pub/Sub | Poller code (user to import) | Small — wire existing code |
| 2 | Task 2 — Raw→Baseline | Task 1 output schema | Small — field mapping DoFn |
| 3 | Task 4 — Imputation stats export | Current training pipeline | Small — extend deploy step |
| 4 | Task 3 — Streaming Beam pipeline | Tasks 1, 2, 4 | Medium — port batch DAG, materialize side inputs |
| 5 | Encoder window + endpoint call | Tasks 3, 4 + deployed endpoint | Medium — stateful window buffer |
