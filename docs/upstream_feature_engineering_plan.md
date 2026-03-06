# Upstream Feature Engineering Plan

**Goal**: Engineer spatial features from upstream station events to improve TFT headway prediction for A/C/E southbound trains.

**Status**: The model currently has only temporal context (last 20 headways at *this* station) and static context (which station/route). It has zero spatial awareness — no information about what's happening at stations further up the line. Headway patterns propagate downstream, so upstream signals should be directly predictive.

---

## Route Topology Reference

### A Train (43 stops, 3 southern branches)
Express service: `A02S → A03S → A05S → A06S → A07S → A09S → A12S → A15S → A24S → A27S → A28S → A31S → A32S → A34S → A36S → A38S → A40S → A41S → A42S → A46S → A48S → A51S → A55S → A57S → A59S → A60S → A61S`
- Far Rockaway branch: `→ H02S → H03S → H04S → H06S → H07S → H08S → H09S → H10S → H11S`
- Lefferts branch: `→ A63S → A64S → A65S`
- Rockaway Park branch: `→ H02S → H03S → H04S → H12S → H13S → H14S → H15S`

### C Train (40 stops, all local)
`A09S → A10S → A11S → A12S → A14S → A15S → A16S → A17S → A18S → A19S → A20S → A21S → A22S → A24S → A25S → A27S → A28S → A30S → A31S → A32S → A33S → A34S → A36S → A38S → A40S → A41S → A42S → A43S → A44S → A45S → A46S → A47S → A48S → A49S → A50S → A51S → A52S → A53S → A54S → A55S`

### E Train (22 stops, Queens Blvd → 8th Ave)
`G05S → G06S → G07S → F05S → F06S → F07S → G08S → G14S → G21S → F09S → F11S → F12S → D14S → A25S → A27S → A28S → A30S → A31S → A32S → A33S → A34S → E01S`

### Shared Stops

| Shared by | Stop IDs | Approximate corridor |
|-----------|----------|---------------------|
| A ∩ C ∩ E | `A27S, A28S, A31S, A32S, A34S` | 8th Ave express: Penn Stn → Chambers St |
| A ∩ C only | `A09S, A12S, A15S, A24S, A36S, A38S, A40S, A41S, A42S, A46S, A48S, A51S, A55S` | Upper/lower 8th Ave + Fulton St corridor |
| C ∩ E only | `A25S, A30S, A33S` | 50th St, 23rd St, Park Place |

At the 5 core trunk stops (A27S–A34S), all three routes serve the same platforms. A gap on one route can be partially offset by the other two.

---

## Proposed Features

### Group 1: Same-Route Upstream Headway

**Concept**: For each arrival at station `k`, look at the most recent headway observed at station `k-1` (the next station upstream on the *same route*). If trains are bunching or gapping one stop north, that pattern is about to arrive here.

**Features**:

| Feature | Definition | Rationale |
|---------|-----------|-----------|
| `upstream_headway_1` | Most recent headway at station_order `k-1`, same route | Immediate upstream signal — headway patterns propagate with ~2 min lag |
| `upstream_headway_2` | Most recent headway at station_order `k-2`, same route | Earlier warning, slightly noisier |
| `upstream_headway_3` | Most recent headway at station_order `k-3`, same route | Even earlier warning, captures developing disruptions |

**Implementation logic**:
1. Sort arrivals by `(route_id, stop_id, arrival_time)`.
2. Compute headway at every station (already done — this is `minutes_until_next_train`).
3. For each row at station `k`, find the upstream station `k-1` using the route's ordered stop list.
4. Look backward in time at station `k-1` arrivals and take the most recent *completed* headway (i.e., the headway that was realized before or at the current row's arrival_time).
5. Must use `arrival_time` comparison to prevent leakage — only use headways that were fully observed before the current timestamp.

**Edge cases**:
- First station on each route (A02S, A09S, G05S) has no upstream → fill with `empirical_median`.
- Sparse upstream data during late night → fallback to the empirical median for that (route, is_weekend, hour) bin.
- A train branches: for stations on the Far Rockaway branch (H06S+), the upstream station is H04S, not A61S. The route stop lists already encode this correctly.

### Group 2: Upstream Travel Time (via trip_uid)

**Concept**: Using `trip_uid` to track individual physical trains, compute how long the last train took to travel from upstream stations to the current station. Slow travel times indicate congestion or delays in transit.

**Features**:

| Feature | Definition | Rationale |
|---------|-----------|-----------|
| `last_train_travel_time_1` | For the most recent `trip_uid` to arrive at station `k`: its arrival at `k` minus its arrival at `k-1` | Measures current inter-station travel time |
| `travel_time_deviation_1` | `last_train_travel_time_1 / median_travel_time(route, k-1→k, is_weekend, hour)` | Normalizes by segment — a ratio > 1 means this train was slow on the segment |
| `last_train_travel_time_2` | Same as above but from `k-2` to `k` | Captures delays accumulating over a longer stretch |

**Implementation logic**:
1. For each row at (route `r`, station `k`, arrival_time `t`), identify the `trip_uid` of the most recent train to arrive (this is the current row's own `trip_uid`).
2. Look up that `trip_uid`'s arrival at station `k-1`. The difference `t - arrival_at_k-1` is the inter-station travel time.
3. Compute median travel times per (route, segment, is_weekend, hour) from the training period only (same approach as `empirical_median`).
4. Divide actual travel time by median to get deviation ratio.

**Edge cases**:
- Trip may not have stopped at `k-1` (e.g., express train skipping a local stop). Use the next upstream station the trip *did* stop at, and normalize accordingly.
- Some `trip_uid`s may appear at `k` but not at `k-1` due to data gaps, trip origination, or rerouting. These get NaN → fill with 1.0 (ratio) or the median travel time (absolute).

### Group 3: Time Since Last Upstream Departure

**Concept**: How long ago did the last train of the same route pass the upstream station? This directly estimates how soon a train should be arriving.

**Features**:

| Feature | Definition | Rationale |
|---------|-----------|-----------|
| `time_since_upstream_1` | Current `arrival_time` minus the most recent arrival at station `k-1` (same route) | Short value = a train is on its way; long value = gap forming |
| `time_since_upstream_2` | Same for station `k-2` | Earlier signal, accounts for trains that may still be 2 stops away |

**Implementation logic**:
1. For each row at (route `r`, station `k`, time `t`): find all arrivals at station `k-1` for route `r` where `arrival_time < t`.
2. Take the max (most recent). Compute `t - max_arrival`.
3. This is a "known at prediction time" feature since we're only looking at past events.

**Relationship to target**: This feature has a direct mechanical relationship. If a train passed 42nd St (k-1) 1 minute ago, it should arrive at 34th St (k) in roughly 2 minutes. The actual target minus this feature approximates remaining dwell + travel time.

### Group 4: Cross-Route Upstream Signals (Shared Trunk)

**Concept**: On the shared 8th Avenue trunk (A27S–A34S and additional A∩C stops), all A/C/E trains serve the same platforms. A rider doesn't care if it's an A or C — they care when the *next train* arrives. The model should know about cross-route supply.

**Features**:

| Feature | Definition | Rationale |
|---------|-----------|-----------|
| `any_route_headway_upstream` | Most recent headway at the upstream stop considering ALL A/C/E trains (not just same route) | Combined service picture at shared stations |
| `any_route_time_since_upstream` | Time since *any* A/C/E train passed the upstream stop | More granular than same-route — captures interleaved service |
| `cross_route_trains_last_10min` | Count of A/C/E trains that passed the upstream stop in last 10 minutes | Measures aggregate throughput / supply health |

**Applicability**: These features are only meaningful at shared stops. At non-shared stops (e.g., E train in Queens, A train in Rockaway), default to same-route values. Implementation needs a lookup table of which stops are shared.

**Shared stop lookup** (derived from topology above):
```
A ∩ C: A09S, A12S, A15S, A24S, A27S, A28S, A31S, A32S, A34S, A36S, A38S, A40S, A41S, A42S, A46S, A48S, A51S, A55S
A ∩ C ∩ E: A27S, A28S, A31S, A32S, A34S
C ∩ E: A25S, A27S, A28S, A30S, A31S, A32S, A33S, A34S
```

### Group 5: Deviation Features (Composites of Existing + New)

These are arithmetic combinations of features that already exist (or will exist after Groups 1–4), making deviation signals explicit rather than requiring the model to learn them internally.

| Feature | Definition | Rationale |
|---------|-----------|-----------|
| `headway_deviation_ratio` | `rolling_mean_3 / empirical_median` | >1 = worse than normal, <1 = better. Explicit relative deviation. |
| `headway_deviation_signed` | `rolling_mean_3 - empirical_median` | Absolute deviation in minutes. Complementary to ratio. |
| `deviation_trend` | `rolling_mean_3 - rolling_mean_10` | Positive = deteriorating, negative = recovering. Direction of change. |
| `deviation_streak` | Count of last N headways above empirical_median | Persistence of disruption. 5/5 above = sustained problem; 1/5 = noise. |
| `upstream_deviation_ratio` | `upstream_headway_1 / empirical_median` (upstream station's median) | Is the upstream station experiencing abnormal headways? |

---

## Implementation Considerations

### Leakage Prevention
- All upstream lookups must use strict `arrival_time < current_row_time` to prevent using future information.
- Median lookup tables (travel time medians, headway medians) must be computed from the **training period only** (< 2025-10-29), same as `empirical_median`.
- Rolling/lagged features on upstream data must use `shift(1)` equivalent logic.

### Computational Approach
The upstream features require **cross-group joins** — looking up data from one (route, stop) group while computing features for another. This is fundamentally different from the current rolling features which operate within a single group.

Recommended approach:
1. Build a "master arrival table" sorted by `(route_id, arrival_time)` — all stations mixed together but time-ordered.
2. For each route, build an ordered stop list mapping `station_order → stop_id` (this already exists).
3. For each row, use the route's stop list to identify upstream stop(s), then do an `asof` merge on arrival_time to find the most recent upstream event.
4. `pd.merge_asof` is ideal for "find the most recent row in another table before this timestamp" — it's vectorized and efficient on sorted data.

### NaN Handling
Upstream features will have more NaNs than current features:
- First station per route: no upstream exists → `NaN`
- Early timestamps: no prior upstream arrivals yet → `NaN`
- Sparse late-night service: upstream gaps → `NaN`

Strategy: fill with the appropriate `empirical_median` or segment-level median, computed from training data.

### Feature Count Impact
Current model has ~22 features. This plan adds roughly 12–15 new features:
- Group 1: 3 (upstream headways at k-1, k-2, k-3)
- Group 2: 3 (travel time, deviation, longer segment)
- Group 3: 2 (time since upstream at k-1, k-2)
- Group 4: 3 (cross-route headway, time-since, throughput count) — at shared stops only
- Group 5: 5 (deviation composites)

Total after: ~35–37 features. Still well within TFT capacity.

---

## Suggested Implementation Order

| Priority | Group | Why first |
|----------|-------|-----------|
| 1 | Group 5: Deviation features | Trivial to compute — arithmetic on existing columns. Zero risk, immediate value. |
| 2 | Group 3: Time since upstream | Simplest upstream feature — just a time diff, no trip_uid joins needed. High signal. |
| 3 | Group 1: Upstream headway | Core spatial signal. Requires cross-group lookup but straightforward with merge_asof. |
| 4 | Group 2: Travel time | Requires trip_uid join + segment median computation. More complex but unique signal. |
| 5 | Group 4: Cross-route | Most complex — needs shared-stop mapping + multi-route aggregation. Best saved for last. |

All feature engineering goes in `bigquery_explorer.ipynb`. The training notebook (`tft_training.ipynb`) only needs its `MODEL_COLS`, `SCALE_COLS`, and `TimeSeriesDataSet` feature lists updated after the features are built.

---

## What Changes in the Training Notebook

After features are built and re-exported to parquet:

1. **MODEL_COLS**: Add all new feature column names.
2. **SCALE_COLS**: Add continuous upstream features (travel times, headways, deviations) so they get StandardScaler treatment.
3. **TimeSeriesDataSet**:
   - `time_varying_unknown_reals`: upstream headways, travel times, time-since-upstream, deviation features (these depend on realtime conditions and can't be known in advance).
   - `time_varying_known_reals`: deviation features that only depend on `empirical_median` + time (like `upstream_deviation_ratio` if computed from known schedule-like medians) — though most will be "unknown" since they depend on actual arrival data.
4. **ENCODER_LENGTH**: May benefit from increase if upstream signals introduce longer-range dependencies. Revisit after first training run with new features.
