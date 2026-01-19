# ConvLSTM Encoder-Decoder Architecture — Layer-by-Layer Explanation

## Overview: The Data Flow

```
Historical Headways (30 min) ──► ENCODER ──► Hidden State ──► DECODER ──► Predicted Headways (15 min)
                                                   ▲
Future Schedule (15 min) ──► Broadcast ──► Reshape ─┘
```

---

## Input Layer 1: `headway_input`
```
Shape: (Batch, 30, 66, 2)
        Batch × Time × Stations × Directions
```

**What it represents:** The observed headways at every station for the past 30 minutes.

**Example:** At timestep t=10, Station 25, Direction 0 (southbound), the value might be `7.5` — meaning "a southbound train arrived 7.5 minutes ago."

---

## Input Layer 2: `schedule_input`
```
Shape: (Batch, 15, 2)
        Batch × Time × Terminals
```

**What it represents:** The dispatcher's planned headways for the next 15 minutes at each terminal.

**Example:** `[8.0, 10.0]` means "Terminal 0 plans 8-minute headways, Terminal 1 plans 10-minute headways."

---

## Layer: `broadcast_schedule` (Custom BroadcastScheduleLayer)
```
Input:  (B, 15, 2)
Output: (B, 15, 66, 2)
```

**What it does:** Copies each terminal's schedule value to all 66 stations.

**Why needed:** The ConvLSTM operates on spatial grids. The schedule must have the same spatial dimensions as headway data so the decoder can process them together.

**Conceptually:**
```
Terminal 0: 8 min  →  Station 0: 8, Station 1: 8, ..., Station 65: 8  (Direction 0)
Terminal 1: 10 min →  Station 0: 10, Station 1: 10, ..., Station 65: 10  (Direction 1)
```

---

## Layer: `reshape_headway_5d`
```
Input:  (B, 30, 66, 2)
Output: (B, 30, 66, 2, 1)
```

**What it does:** Adds a channel dimension for ConvLSTM2D compatibility.

**Why needed:** ConvLSTM2D expects 5D input: `(Batch, Time, Height, Width, Channels)`. Our spatial grid is:
- Height = 66 (stations along the line)
- Width = 2 (directions)
- Channels = 1 (headway value)

---

## Layer: `reshape_schedule_5d`
```
Input:  (B, 15, 66, 2)
Output: (B, 15, 66, 2, 1)
```

**What it does:** Same reshape for the broadcasted schedule — adds channel dimension.

---

## Layer: `encoder_convlstm_1` (ConvLSTM2D)
```
Input:  (B, 30, 66, 2, 1)
Output: (B, 30, 66, 2, 32)
```

**Parameters:** 38,144
- Filters: 32
- Kernel: (3, 3)
- Activation: ReLU

**What it does:** 
1. Slides a 3×3 convolution kernel across the station-direction grid
2. At each timestep, captures **local spatial patterns** (e.g., "stations 10-12 all have high headways")
3. The LSTM component maintains **temporal memory** across the 30 timesteps

**Contribution to prediction:** Learns low-level spatiotemporal features like:
- "Delays at station X tend to propagate to station X+1"
- "When headway increases at one station, it often increases at neighbors too"

---

## Layer: `encoder_bn_1` (BatchNormalization)
```
Input:  (B, 30, 66, 2, 32)
Output: (B, 30, 66, 2, 32)
```

**What it does:** Normalizes activations across the batch to stabilize training.

**Contribution:** Prevents internal covariate shift, allows higher learning rates, acts as mild regularization.

---

## Layer: `encoder_convlstm_2` (ConvLSTM2D with return_state=True)
```
Input:  (B, 30, 66, 2, 32)
Output: 
  - Sequence: (B, 30, 66, 2, 32)  — not used
  - state_h:  (B, 66, 2, 32)      — final hidden state
  - state_c:  (B, 66, 2, 32)      — final cell state
```

**Parameters:** 73,856

**What it does:** 
1. Second layer of spatiotemporal feature extraction
2. **Crucially exports `state_h` and `state_c`** — the "memory" of the encoder

**Contribution to prediction:** 
- Builds higher-level abstractions: "There's a bunching pattern developing between stations 20-35"
- The exported states encode the **entire 30-minute history** into a compressed spatial representation

---

## Layer: `encoder_bn_2` (BatchNormalization)
```
Shape: (B, 30, 66, 2, 32) — normalizes the sequence output
```

---

## 🔗 State Transfer: The Critical Bridge

```
state_h, state_c from Encoder → initial_state of Decoder
```

**What happens:** The decoder starts with "memory" of all observed delays. It doesn't start from scratch — it knows the current system state.

---

## Layer: `decoder_convlstm_1` (ConvLSTM2D with initial_state)
```
Input:  (B, 15, 66, 2, 1)  — the broadcasted future schedule
Initial State: [state_h, state_c] from encoder
Output: (B, 15, 66, 2, 32)
```

**Parameters:** 38,144

**What it does:**
1. **Primed with encoder's memory** (knows current delays)
2. **Driven by future schedule** (knows dispatcher's plan)
3. Generates 15 timesteps of predicted features

**Contribution to prediction:** This is where the model answers: "Given the current delays AND the future dispatch plan, what will happen next?"

The schedule acts as a **control signal** — the decoder learns:
- "If schedule says 5-min headways but current delays are 10 min, bunching will worsen"
- "If schedule says 15-min headways (holding), bunching might dissipate"

---

## Layer: `decoder_bn_1` (BatchNormalization)
```
Shape: (B, 15, 66, 2, 32)
```

---

## Layer: `decoder_convlstm_2` (ConvLSTM2D)
```
Input:  (B, 15, 66, 2, 32)
Output: (B, 15, 66, 2, 32)
```

**Parameters:** 73,856

**What it does:** Second decoder layer adds capacity for modeling complex propagation dynamics.

**Contribution:** Refines predictions — captures subtle interactions like:
- How delays in one direction affect the other direction
- Non-linear effects of schedule changes

---

## Layer: `decoder_bn_2` (BatchNormalization)
```
Shape: (B, 15, 66, 2, 32)
```

---

## Layer: `output_projection` (Conv3D)
```
Input:  (B, 15, 66, 2, 32)
Output: (B, 15, 66, 2, 1)
```

**Parameters:** 289
- Filters: 1
- Kernel: (3, 3, 1)
- Activation: ReLU

**What it does:** 
1. Projects 32 feature channels down to 1 (the predicted headway)
2. 3×3 kernel provides final spatial smoothing
3. **ReLU ensures non-negative outputs** (headways can't be negative)

---

## Layer: `output_reshape`
```
Input:  (B, 15, 66, 2, 1)
Output: (B, 15, 66, 2)
```

**What it does:** Strips the channel dimension for clean output shape.

---

## Final Output
```
Shape: (Batch, 15, 66, 2)
        Batch × Forecast_Time × Stations × Directions
```

**Interpretation:** For each of the next 15 minutes, at each of the 66 stations, in each direction — the predicted headway in minutes.

---

## Summary Table

| Layer | Purpose | Key Contribution |
|-------|---------|------------------|
| `headway_input` | Accept history | Raw observations |
| `schedule_input` | Accept control signal | Dispatcher's plan |
| `broadcast_schedule` | Expand to spatial grid | Make schedule spatially compatible |
| `encoder_convlstm_1` | Extract features | Local spatiotemporal patterns |
| `encoder_convlstm_2` | Encode + export state | Compress history into memory |
| **State Transfer** | Bridge past→future | Memory of current delays |
| `decoder_convlstm_1` | Conditioned generation | Fuse memory + future plan |
| `decoder_convlstm_2` | Refine predictions | Complex propagation dynamics |
| `output_projection` | Map to headway | Single positive value per cell |

---

## The Learning Objective

```python
loss = MSE(predicted_headways, actual_future_headways)
```

The model learns weights that minimize the difference between:
- What it predicts will happen (given history + plan)
- What actually happened (ground truth)

Over millions of examples, it learns the dynamics of how delays propagate through the subway system.

---

## Model Statistics

- **Total Parameters:** 224,673 (877.63 KB)
- **Trainable Parameters:** 224,481
- **Non-trainable Parameters:** 192 (BatchNorm statistics)

## Data Compatibility

| Data File | Raw Shape | Model Input |
|-----------|-----------|-------------|
| `headway_matrix_full.npy` | `(T, 66, 2, 1)` | `(B, 30, 66, 2)` |
| `schedule_matrix_full.npy` | `(T, 2, 1)` | `(B, 15, 2)` |
