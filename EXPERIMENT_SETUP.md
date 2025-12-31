# Experiment Setup: Real-Time Headway Prediction (Option B)

This document outlines the implementation plan for adapting the methodology from *arXiv:2510.03121* (Sections 2.2–3.1) to real-world MTA subway data.

## 1. Experiment Overview
Instead of using SimMetro (simulation), we are using **Historical MTA Data**. This requires a robust data engineering pipeline to handle noise, missing values, and irregular sampling.

*   **Objective:** Train a Deep Learning model (ConvLSTM) to predict the spatiotemporal evolution of headways on the A-line.
*   **Input ($X$):** Recent history of headways across the line (Congestion).
*   **Input ($T$):** Scheduled departures at the terminal (Dispatcher Intent).
*   **Output ($Y$):** Future headways across the line (Prediction).

---

## Phase 1: Dataset Construction (The Foundation)
*Goal: Transform raw CSV/Parquet files into processed Tensors ready for TensorFlow.*

### Step 1.1: The "Space" Dimension (Static Map)
*   **Status:** Partially Complete (`static_data_processing.ipynb`).
*   **Task:** Finalize the 0.2-mile binning logic.
*   **Output:** A lookup table mapping every `stop_id` to a `distance_bin_index` (0 to $N_d$).

### Step 1.2: The "Time" Dimension (Arrivals)
*   **Status:** Complete (`data_collection.ipynb`).
*   **Task:** Load the 6-month parquet archive.
*   **Output:** A clean DataFrame of `[arrival_time, stop_id, direction]`.

### Step 1.3: The "Target" Dimension (Schedule)
*   **Status:** Pending.
*   **Task:** Implement the logic from `TARGET_TERMINAL_HEADWAYS.md`.
*   **Output:** A time-series of scheduled terminal departures for the entire 6-month period.

### Step 1.4: The "Grid" (Spatiotemporal Matrix)
*   **Status:** Prototyped (`data_merging.ipynb`).
*   **Task:**
    1.  Create the blank matrix: `(Total_Time_Bins, Total_Distance_Bins)`.
    2.  Populate with actual headways.
    3.  **Crucial:** Apply interpolation to fill gaps (SimMetro gives perfect data; we must "repair" ours).
*   **Output:** A massive, dense 2D matrix (Time $\times$ Space) for the full 6 months.

### Step 1.5: The "Slicer" (Tensor Generation)
*   **Status:** Pending.
*   **Task:** Implement the sliding window logic (Equations 4, 5, 6).
    *   Slide a window of size $L+F$ over the Grid.
    *   Extract $X$ (Past $L$ steps).
    *   Extract $Y$ (Future $F$ steps).
    *   Align with $T$ (Schedule).
*   **Output:** Saved `.npy` or `.tfrecord` files containing the final tensors.

---

## Phase 2: Model Architecture (The Brain)
*Goal: Build the Keras model defined in the paper.*

### Step 2.1: Input Layers
*   **Visual Input:** Shape `(L, Nd, 1)` -> Processed by ConvLSTM.
*   **Terminal Input:** Shape `(F, 1)` -> Processed by Fully Connected (Dense) layers.

### Step 2.2: The Core (ConvLSTM)
*   **Task:** Implement `ConvLSTM2D` layers to capture how "traffic jams" move through space and time.

### Step 2.3: Fusion & Output
*   **Task:** Merge the "Visual" features with the "Terminal" features.
*   **Output:** A Conv3D or Conv2DTranspose layer to reconstruct the predicted heatmap $Y$.

---

## Phase 3: Training & Evaluation (The Exam)
*Goal: Train the model and measure success.*

### Step 3.1: Data Pipeline
*   **Task:** Create a `tf.data.Dataset` pipeline to efficiently load the tensors during training (batching, shuffling).

### Step 3.2: Training Loop
*   **Loss Function:** Mean Squared Error (MSE) (Pixel-wise difference between Predicted Heatmap and Actual Heatmap).
*   **Optimizer:** Adam.

### Step 3.3: Evaluation
*   **Metric:** RMSE (Root Mean Squared Error) in minutes.
*   **Visual Check:** Plot Predicted Heatmap vs. Actual Heatmap side-by-side.
