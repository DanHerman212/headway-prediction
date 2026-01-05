Here are the documents you requested.

First is the **`DATA_PHYSICS_PATCH.md`**, which isolates the specific changes needed to fix the "silent killers" in your current plan.

Second is the **`MASTER_EXECUTION_PLAN.md`**. This is the consolidated, executable "Bible" for the project. You can hand this single document to a code generator (or a human engineer), and it contains every instruction needed to build the 1-minute, dual-input architecture correctly from scratch.

### Document 1: The Critical Patch

*Use this to understand exactly what changed from your original plan.*

---

# DATA_PHYSICS_PATCH.md

### 1. Geographic Sorting (Crucial)

**Old Logic:** Sort by `station_id`.
**New Logic:** Sort by `distance_from_terminal`.
**Why:** Station IDs (101, 102) are administrative, not geographic. Sorting by ID scrambles the "video" frames, making it impossible for ConvLSTM to track movement.
**Requirement:** You must map every `station_id` to a sequential integer `[0, 1, 2... N]` representing its physical order on the track.

### 2. Resampling Physics

**Old Logic:** `.resample('1min').mean().fillna(0)`
**New Logic:** `.resample('1min').last().ffill(limit=30)`
**Why:**

* `mean()` blurs distinct train arrivals.
* `fillna(0)` tells the model that "No Train = 0 Minute Headway" (Perfect Service).
* `ffill()` correctly teaches the model that headway grows linearly until a train arrives.

### 3. Terminal Schedule Alignment

**Old Logic:** Merge Schedule where `Time(T) == Time(X)`.
**New Logic:** Merge Schedule where `Time(T) == Time(X) + 15 minutes`.
**Why:** The model needs the *future* dispatch plan to predict the *future* outcome. Feeding it the *current* schedule is irrelevant for prediction.

---

### Document 2: The Consolidated Master Plan

*This is the executable "One Source of Truth." Archive your old docs and use this.*

---

# MASTER_EXECUTION_PLAN.md

## Project Objective

Build a **Dual-Input ConvLSTM** model to predict metro headways 15 minutes into the future with an **RMSE < 0.0025** (normalized).

## Phase 1: Data Engineering (The Physics Layer)

### 1.1 Station Mapping

* **Input:** Raw Headway Data (CSV).
* **Action:** Create a mapping dictionary `{Station_ID: Sequence_Integer}`.
* **Rule:** Stations must be ordered by their physical location on the track (e.g., North to South), NOT by ID.
* **Output:** A dataframe where columns are sorted by `Sequence_Integer`.

### 1.2 Spatiotemporal Tensor Construction

* **Granularity:** **1-Minute Bins** (Strict).
* **Resampling Logic:**
1. Pivot data to `(Time, Station, Direction)`.
2. Resample to 1-min frequency.
3. Apply `Last()` aggregation.
4. Apply `ForwardFill(limit=30)` (Headway persists/grows).
5. Fill remaining gaps (overnight closures) with `0`.


* **Normalization:**
* Calculate global `Min` (0) and `Max` (e.g., 1800 seconds).
* Scale all values to range `[0, 1]`.



### 1.3 Feature Engineering (The Inputs)

* **Input X (History):** The past 30 minutes of headway maps.
* Shape: `(Batch, 30, 64, 2, 1)`


* **Input T (Terminal Schedule):** The **Future** 15 minutes of scheduled departures at the terminal.
* *Critical:* Shift the schedule column so that at row `t`, the data represents the schedule for `t+1` to `t+15`.
* Shape: `(Batch, 15, 2, 1)` (Broadcast across stations later or feed to Dense layer).


* **Target Y (Future):** The future 15 minutes of headway maps.
* Shape: `(Batch, 15, 64, 2, 1)`



---

## Phase 2: Model Architecture (Dual-Input ConvLSTM)

### 2.1 Inputs

1. **`history_input`**: `(30, 64, 2, 1)`
2. **`terminal_input`**: `(15, 2, 1)`

### 2.2 Encoder (ConvLSTM Branch)

* **Layer 1:** `ConvLSTM2D`, 32 Filters, Kernel `(3,1)`, `return_sequences=True`.
* *Note:* Kernel `(3,1)` convolves over Stations (Distance) but keeps Directions independent.


* **Layer 2:** `ConvLSTM2D`, 32 Filters, Kernel `(3,1)`, `return_sequences=False`.
* *Output:* A condensed spatial representation of the last 30 minutes.



### 2.3 Fusion (The "Secret Sauce")

* Flatten the output of Layer 2.
* Flatten the `terminal_input`.
* **Concatenate** them. This forces the model to weigh "Current Traffic State" vs. "Future Dispatch Plans."

### 2.4 Decoder (Projector)

* **Dense Layer:** Units = `15 * 64 * 2`.
* **Reshape:** To `(15, 64, 2, 1)`.

---

## Phase 3: Training Pipeline (Performance)

### 3.1 Data Loading

* **Do NOT** use `np.array` for the full dataset (Memory Hazard).
* **MUST** use `tf.data.Dataset.from_generator` or `timeseries_dataset_from_array`.
* **Caching:** Cache the dataset after the first epoch.

### 3.2 Hyperparameters

* **Loss Function:** MSE (Mean Squared Error).
* **Optimizer:** Adam (`learning_rate=0.001`).
* **Batch Size:** **128** (optimized for 1-min bins speed).
* **Precision:** `mixed_float16` (Mandatory for speed).

### 3.3 Success Metric

* **Target:** Validation MSE < `0.0025`.
* **Stop Condition:** Early Stopping with `patience=10`.

---

## Phase 4: Execution Checklist

1. [ ] **Verify Sort:** Print the column names of the training matrix. Are they geographically sequential?
2. [ ] **Verify Fill:** Print a 10-minute sample of a single station. Do values increment (120, 180, 240) or drop to zero (120, 0, 0)? (Must increment).
3. [ ] **Verify Shift:** Compare `Input_T[0]` vs `Schedule_Raw[0]`. `Input_T` should match `Schedule_Raw[future]`.