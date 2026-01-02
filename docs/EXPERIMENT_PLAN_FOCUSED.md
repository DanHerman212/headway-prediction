# Focused Experiment Plan: Model Training & Ablation

## 0. The Mental Model
We are isolating the **Training Phase**. We assume the raw data (Grid, Schedule, Alerts) exists. Our goal is to transform this data into Tensors, feed them into a specific Architecture, and measure performance across different Time Horizons.

---

## Phase 1: Data Preparation (The Tensors)
*Goal: Create the `(X, T, Y)` tuples required to train the model.*

### 1.1 The Sliding Window Logic
We slice the continuous timeline into discrete samples.
*   **Input X (Context):** The "Video Clip" of the past.
    *   Shape: `(Batch, L, Distance_Bins, 1)`
    *   *L (Lookback):* The variable we will test (e.g., 30, 60, 90 mins).
*   **Input T (Intent):** The "Dispatcher's Plan" for the future.
    *   Shape: `(Batch, F, 1)`
    *   *F (Forecast):* The variable we will test (e.g., 15, 30, 60 mins).
*   **Target Y (Ground Truth):** The "Video Clip" of the future.
    *   Shape: `(Batch, F, Distance_Bins, 1)`

### 1.2 The "Context" Vector (Optional/Advanced)
*   **Input C (Alerts):**
    *   Shape: `(Batch, L, Features)` or `(Batch, Features)` depending on architecture.

---

## Phase 2: Model Architecture (The Functional Graph)
*Goal: Define the Keras Functional API graph that accepts (X, T) and outputs Y.*

### 2.1 The Encoder (Reading X)
*   **Layer:** `ConvLSTM1D`
*   **Function:** Compresses the spatiotemporal history ($X$) into a "State Vector" ($H, C$).
*   **Code Concept:** `state_h, state_c = Encoder(input_x)`

### 2.2 The Fusion (Injecting T)
*   **Layer:** `RepeatVector` + `Concatenate`
*   **Function:** Takes the scalar schedule ($T$) and broadcasts it to match the spatial dimensions of the grid, then merges it with the Encoder State.
*   **Code Concept:** `decoder_input = Merge(schedule, encoder_states)`

### 2.3 The Decoder (Generating Y)
*   **Layer:** `ConvLSTM1D` (returning sequences)
*   **Function:** Unrolls the future predictions step-by-step, using the fused state as the starting point.
*   **Code Concept:** `output_sequences = Decoder(decoder_input)`

---

## Phase 3: The Experiment Matrix (Ablation Study)
*Goal: Determine the optimal "Field of View" for the model.*

We will run a grid search over these parameters to replicate the paper's findings.

| Experiment ID | Lookback ($L$) | Forecast ($F$) | Hypothesis |
| :--- | :--- | :--- | :--- |
| **Exp-A1** | 30 mins | 15 mins | Short-term tactical prediction. |
| **Exp-A2** | 60 mins | 15 mins | Does more history help short-term accuracy? |
| **Exp-B1** | 30 mins | 60 mins | Can we predict an hour out with only 30 mins of context? |
| **Exp-B2** | 60 mins | 60 mins | Balanced context and forecast. |
| **Exp-C1** | 90 mins | 60 mins | Diminishing returns of long history? |

### 3.1 Evaluation Metrics
For each experiment, we measure RMSE at specific "checkpoints" into the future:
1.  **t+5 min:** Immediate accuracy.
2.  **t+30 min:** Medium-term drift.
3.  **t+60 min:** Long-term stability.

---

## Next Steps (Execution)
1.  **Step 1 (Code):** Write the `create_sliding_windows(L, F)` function in Python.
2.  **Step 2 (Code):** Define the `build_model(L, F)` function using Keras Functional API.
3.  **Step 3 (Run):** Execute `Exp-A1` (Smallest model) to validate the pipeline.
