# Real-Time Headway Prediction System Plan (NYC Subway)

This plan follows the Universal Machine Learning Framework and is based on the approach described in the paper *"Real Time Headway Predictions in Urban Rail Systems and Implications for Service Control: A Deep Learning Approach"* (arXiv:2510.03121).

## Phase 1: Define the Task

### 1.1 Understand the Problem Domain
*   **Goal:** Predict the spatiotemporal propagation of train headways across an NYC subway line in real-time.
*   **Business Logic:** Dispatchers need to know how current delays and terminal decisions affect downstream service.
*   **Key Concept:** Treat the subway line as a spatial grid and time as a sequence, predicting the "headway heatmap" into the future.

### 1.2 Data Collection Strategy
*   **Primary Data Source:** MTA GTFS-Realtime Feeds (specifically the Vehicle Positions and Trip Updates entities).
*   **Historical Data:** Access to a data archive (April 2021 - Present) for 24 route_ids.
*   **Static Data:** MTA GTFS Static feed (stops.txt, shapes.txt) to map physical locations to "distance bins".
*   **Target Line:** **A Line** (longest line in the system).
*   **Selected History:** TBD (Suggesting recent 6-12 months).

### 1.3 Success Metrics
*   **Quantitative:** Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and $R^2$ score between predicted and actual headways.
*   **Qualitative:** Visual alignment of predicted vs. actual headway heatmaps.

## Phase 2: Develop a Model

### 2.1 Data Preparation (The Pipeline)
*   **Step 1: Ingestion:** Script to poll MTA API every 30-60 seconds and store raw JSON/Protobuf.
*   **Step 2: Discretization (Grid Generation):**
    *   Divide the subway line into $N_d$ fixed-distance bins (e.g., 64 bins).
    *   Divide time into fixed bins $\Delta T$ (e.g., 1 minute).
*   **Step 3: Headway Calculation:**
    *   Calculate headway $H(t, j, k)$ for each cell (time $t$, distance $j$, direction $k$).
    *   Formula: Average time difference between consecutive trains in that bin.
    *   Impute missing values (no train in bin) using interpolation or station-based logic.
*   **Step 4: Tensor Construction:**
    *   **Historical Input ($X$):** Shape $(L, N_d, N_{dir}, 1)$. Lookback $L=30$ mins.
    *   **Terminal Input ($T$):** Shape $(F, N_{dir}, 1)$. Future scheduled/planned headways at terminals. Prediction horizon $F=15$ mins.
    *   **Target ($Y$):** Shape $(F, N_d, N_{dir}, 1)$. Actual future headways.
*   **Step 5: Normalization:** Min-Max scaling to $[0, 1]$.

### 2.2 Model Architecture
*   **Type:** ConvLSTM (Convolutional Long Short-Term Memory).
*   **Reasoning:** Captures both spatial dependencies (trains affecting trains behind them) and temporal evolution.
*   **Configuration (based on paper):**
    *   Filters: 32
    *   Kernel Size: (To be determined, likely $3 \times 3$)
    *   Layers: Stacked ConvLSTM layers.
    *   Fusion: Merge Historical ($X$) and Terminal ($T$) inputs.

### 2.3 Training & Evaluation
*   **Baseline:** Simple historical average or "last observed headway" carried forward.
*   **Protocol:**
    *   Split data into Train (70%), Validation (15%), Test (15%).
    *   Loss Function: MSE.
    *   Optimizer: Adam (LR=0.001).
    *   Early Stopping: Monitor validation loss.

## Phase 3: Deploy the Model

### 3.1 Deployment Architecture
*   **Backend:** Python (FastAPI) to serve predictions.
*   **Inference Engine:** PyTorch or TensorFlow/Keras model loader.
*   **Real-Time Loop:**
    1.  Fetch live MTA data.
    2.  Preprocess into current $X$ tensor.
    3.  Fetch/Construct $T$ tensor (from schedule or dispatcher input).
    4.  Run Inference.
    5.  Return predicted heatmap.

### 3.2 User Interface
*   **Dashboard:** A web page displaying the "Headway Heatmap".
    *   Y-axis: Distance (Stations).
    *   X-axis: Time (Past 30 mins + Future 15-60 mins).
    *   Color: Headway duration (Green=Short, Red=Long).

### 3.3 Monitoring
*   **Drift Detection:** Monitor if prediction error increases over time (e.g., due to schedule changes or major incidents).
*   **Data Collection:** Continuously save live inputs and predictions to build the "next generation" dataset.

---

## Execution Steps (Immediate Next Actions)
1.  **Environment Setup:** Create a Python virtual environment and install dependencies (pandas, numpy, torch/tensorflow, protobuf, gtfs-realtime-bindings).
2.  **Data Acquisition Script:** Write a script to fetch and save MTA GTFS-Realtime data.
3.  **Static Data Mapping:** Parse `stops.txt` and `shapes.txt` to define the "Distance Bins" for the chosen line.
