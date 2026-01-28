Based on your current setup using a **stacked GRU with dual output** in a **Vertex AI/Hydra** pipeline, and your specific goal to beat the "scheduled headway" baseline for the composite A/C/E lines at W 4th St, here are three architectural experiments you should add to your source code.

Given the "composite" nature of your data (interweaving schedules of A, C, and E trains), your target distribution is likely **multi-modal** (e.g., a peak at 4 mins for C, 8 mins for E). A standard GRU with MSE loss will average these, leading to poor performance.

### 1. The "Low-Hanging Fruit": GRU + Mixture Density Network (MDN) Head

**Why:** Your current GRU predicts a single scalar (the mean headway). For composite routes, the "time to next train" is not a single bell curve—it's a mixture of distributions depending on which train (A, C, or E) is coming next. An MDN head predicts the *parameters* (means, variances, weights) of a Gaussian Mixture Model rather than a single value.
**Implementation:**

* **Modify the Head:** Replace your final `Dense(1)` layer with a layer that outputs `3 * K` values (where `K` is the number of mixtures, e.g., 5). These outputs correspond to the mixing coefficients (), means (), and standard deviations ().
* **Loss Function:** Switch from MSE to **Negative Log-Likelihood (NLL)** of the mixture distribution.
* **Pipeline Integration:** This fits directly into your existing `(Batch, Time, Features)` data structure without changing your upstream data loaders.

### 2. The SOTA Contender: Temporal Fusion Transformer (TFT)

**Why:** TFT is currently state-of-the-art for multi-horizon forecasting because it explicitly handles different types of inputs that RNNs struggle to separate:

* **Static Covariates:** The "Route ID" (A vs C vs E) or "Day of Week".
* **Known Future Inputs:** The *Scheduled* Arrival Time (you know this in advance!).
* **Observed Inputs:** The *Actual* past headways.
**Implementation:**
* **Architecture:** Use the `MultiHeadAttention` mechanisms to weigh the importance of past delays from upstream stations (e.g., 14th St) vs. static schedule rules.
* **Interpretablity:** TFT provides variable selection weights, so you can see if the model is ignoring your "Route ID" feature or over-relying on it.

### 3. Strategy Change: Residual Learning (The "Baseline Beater")

**Why:** You mentioned your baseline is the "scheduled headway." Instead of training your Deep Learning model to predict the *total* headway (which is 90% the schedule and 10% noise), train it to predict the **deviation** (delay).

* **Input:** Scheduled Headway + Upstream Delays + Time of Day.
* **Target:** `Actual_Headway - Scheduled_Headway`.
* **Architecture:** Your existing Stacked GRU can be used here.
* **Benefit:** This simplifies the learning landscape. The model effectively learns "The C train is usually 2 minutes later than the schedule at 5 PM," which is easier than learning the entire schedule from scratch.

### Recommended Hydra Experiment Config

Since you are using Hydra, you can structure your experiments to swap these architectures easily:

```yaml
# conf/model/architecture.yaml

defaults:
  - override /model: gru_mdn  # or 'tft', 'residual_gru'

model:
  name: "gru_mdn"
  params:
    hidden_units: [64, 32]
    mixture_components: 5  # For MDN
    dropout: 0.2

```

### Summary of Next Steps

1. **Immediate:** Switch your target variable to **residual** (Delay) rather than raw headway.
2. **Short-term:** Implement an **MDN head** on your GRU to capture the multi-modal nature of the A/C/E combined schedule.
3. **Research:** If you have upstream data (e.g., trains leaving 34th St or 14th St), investigate **ConvLSTM** or **Graph Neural Networks (GNN)** to model the spatial propagation of delays down the line.