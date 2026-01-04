# Experimentation Framework & Technical Deep Dive

## 1. Experimentation Framework Plan

### Phase 1: Configurable Data Pipeline
*   **Goal**: Transform the static `create_dataset` function into a dynamic generator driven by configuration.
*   **Key Features**:
    *   **Variable Lookback ($L$)**: Support for 30, 45, 60 minute windows (defined in bins).
    *   **Input Toggles**: Ability to include/exclude Terminal Headways ($T$) to test the "Dispatcher Intent" hypothesis.
    *   **Single-Step Target**: Generate $Y$ for $t+15$ min (single step) to enable recursive training.

### Phase 2: Model Factory (TensorFlow/Keras)
*   **Goal**: A function `build_model(config)` that returns a compiled Keras model.
*   **Architecture**:
    *   **Input**: 5D Tensor `(Batch, Time, Rows, Cols, Channels)`
        *   *Note: Even though our space is 1D (stations), ConvLSTM2D expects 2 spatial dims. We often reshape `(Stations, 1)` to fit.*
    *   **Layer**: `ConvLSTM2D` (captures spatiotemporal patterns).
    *   **Output**: Map `(Batch, Rows, 1)` representing the headway state at $t+1$.

### Phase 3: The Experiment Loop
*   **Goal**: Execute the 8 configurations defined in the abstract.
*   **Process**:
    1.  **Train**: Train Base Model (Lookback=30, Horizon=15).
    2.  **Inference**: Run Recursive Inference for 4 steps (15, 30, 45, 60 mins).
    3.  **Evaluate**: Calculate Metrics (RMSE, $R^2$) for each step.
    4.  **Repeat**: Perform for Northbound/Southbound and with/without Terminal Headways.

---

## 2. Technical Deep Dive

### A. Recursive Strategy: The "Why" and "How"

**The "Why":**
Training a model to predict 60 minutes into the future directly is difficult because uncertainty compounds over time. Instead, we train the model to be very good at predicting just the **next 15 minutes**.

**The "How":**
1.  **Training (Teacher Forcing):** We show the model inputs $X_{t-30:t}$ and ask it to predict $Y_{t+15}$. We calculate loss against the *actual* ground truth.
2.  **Inference (Recursive):**
    *   **Step 1:** Model predicts $Y_{t+15}$.
    *   **Step 2:** We take that *predicted* $Y_{t+15}$, treat it as if it were real data, append it to our input history (dropping the oldest timestamp), and ask the model: "Given this new history, what happens next?" -> Model predicts $Y_{t+30}$.
    *   **Step 3:** Repeat for $Y_{t+45}$ and $Y_{t+60}$.

This allows a short-term model to generate long-term forecasts.

### B. ConvLSTM2D vs. Conv1D

**The Data Structure:**
Our data is a "Video" of traffic. It has:
*   **Time:** The sequence of frames.
*   **Space:** The distance along the track (Rows).
*   **Channels:** Headway values.

**Conv1D (Temporal Only):**
*   **How it works:** Slides a window across *time* only. It treats the spatial dimension (stations) as just a flat vector of features.
*   **Limitation:** It doesn't understand that Station A is physically next to Station B. It just sees them as "Feature 1" and "Feature 2". It loses the *spatial* correlation of traffic waves moving down the line.

**ConvLSTM2D (Spatiotemporal):**
*   **How it works:** It uses a **2D Convolution** inside the LSTM cell.
*   **Why it fits:**
    1.  **Spatial:** The convolution operation slides over the "Space" dimension, learning that traffic at Station $i$ affects Station $i+1$.
    2.  **Temporal:** The LSTM cell maintains a "Memory State" over time, learning how these spatial patterns evolve (e.g., a traffic jam propagating backward).
*   **Result:** It captures the "Wave" physics of transit headways perfectly.

### C. Inference & Evaluation: Recursive Execution

**What it means to run recursively:**
Imagine predicting the weather for the next 4 days.
1.  **Day 1 Prediction:** You look at today's clouds and predict tomorrow's rain.
2.  **Day 2 Prediction:** You don't have "tomorrow's clouds" yet. So you use your *prediction* of tomorrow's rain as the input to predict the day after.

**In our context:**
*   **Input:** Real data from 7:00 - 8:00.
*   **Loop 1:** Predict 8:15. (Store this result).
*   **Loop 2:** Use Real data (7:15 - 8:00) + *Predicted* (8:15) to predict 8:30.
*   **Loop 3:** Use Real data (7:30 - 8:00) + *Predicted* (8:15, 8:30) to predict 8:45.

**Evaluation:**
We then take these 4 predictions (8:15, 8:30, 8:45, 9:00) and compare them against the *actual* ground truth files for those times to calculate RMSE. This tells us how well the model holds up as it relies more and more on its own guesses.
