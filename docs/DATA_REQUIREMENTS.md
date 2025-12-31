# Data Representation Requirements: NYC Subway Headway Prediction

This document defines the mathematical and structural requirements for constructing the dataset, based on the methodology in *arXiv:2510.03121*.

## 1. Discretization (The Grid)

The core concept is to transform continuous train operations into a discrete spatiotemporal grid.

### A. Time Binning (Equation 1)
We segment the operational time window $[t_{start}, t_{end}]$ into bins of uniform size $\Delta T$.

$$time\_bin_i = \lfloor \frac{t - t_{start}}{\Delta T} \rfloor$$

**Variable Breakdown:**
*   $t$: The specific timestamp of a train event (e.g., 08:12:00).
*   $t_{start}$: The starting time of the operational day (e.g., 00:00:00).
*   $\Delta T$ (Delta T): The size of one time bucket (e.g., 5 minutes).
*   $\lfloor \dots \rfloor$: The **Floor Function**. It means "round down to the nearest whole number."
    *   *Example:* If the result is 2.4, the floor is 2. This puts the event into "Bin #2".

### B. Distance Binning (Equation 2)
We divide the track length into $N_d$ equally sized segments.

$$distance\_bin_j = \lfloor \frac{d - d_{min}}{(d_{max} - d_{min}) / N_d} \rfloor$$

**Variable Breakdown:**
*   $d$: The distance of the train from the start of the line (e.g., 10.5 miles).
*   $d_{min}$: The location of the very first station (0.0 miles).
*   $d_{max}$: The location of the very last station (e.g., 32.0 miles).
*   $N_d$: The total number of bins we want to create (e.g., 160 bins).
*   $(d_{max} - d_{min}) / N_d$: This calculates the **Bin Width** (e.g., 0.2 miles).

## 2. Aggregation (The Value)

### Headway Calculation (Equation 3)
For each grid cell $(t, j, k)$ (Time $t$, Distance $j$, Direction $k$), the value is the mean of all headways observed in that cell.

$$H(t, j, k) = \frac{1}{N_{t,j,k}} \sum_{i \in S_{t,j,k}} h_i$$

**Variable Breakdown:**
*   $H(t, j, k)$: The final average headway value for this specific grid cell.
*   $N_{t,j,k}$: The **Count** of how many train events landed in this cell.
*   $S_{t,j,k}$: The **Set** (collection) of all train events in this cell.
*   $\sum$ (Sigma): The **Summation** symbol. It means "add up all the values".
*   $h_i$: The individual headway value for a single train $i$.
*   *Plain English:* "Add up all the headways ($h_i$) and divide by the number of trains ($N$) to get the average."

**Imputation:** Cells with no events ($N_{t,j,k} = 0$) must be filled using interpolation (linear or station-based logic) to create a continuous field.

## 3. Normalization
All headway values $H(t, j, k)$ are normalized to the interval $[0, 1]$ to ensure numerical stability for the neural network.

$$H_{norm} = \frac{H}{H_{max}}$$

*   $H_{max}$: Maximum expected headway (e.g., 60 minutes).

## 4. Tensor Construction (Input/Output Sequences)

The final stage involves sliding a window over the spatiotemporal grid to create the training examples.

### A. Historical Input Tensor ($X$) - Equation 4
Captures the recent history of the system (The "Rearview Mirror").

$$X = [H(t-L, j, k), \dots, H(t-1, j, k)] \in \mathbb{R}^{L \times N_d \times N_{dir} \times 1}$$

**Variable Breakdown:**
*   $X$: The input data structure fed into the AI model.
*   $t$: The current time step (right now).
*   $L$: **Look-back Window**. How many past time steps we include (e.g., 6 steps = 30 mins).
*   $H(t-L, \dots)$: The state of the subway line $L$ steps ago.
*   $\mathbb{R}^{L \times N_d \times N_{dir} \times 1}$: **Dimensions**.
    *   $\mathbb{R}$: Real numbers (decimals).
    *   $L$: Time depth (e.g., 6).
    *   $N_d$: Space width (e.g., 160 bins).
    *   $N_{dir}$: Number of directions (2: North/South).
    *   $1$: The number of "channels" (just Headway).

### B. Terminal Input Tensor ($T$) - Equation 5
Captures the scheduled service at the origin (The "Dispatcher Plan").

$$T = [H(t, s, k), \dots, H(t+F-1, s, k)] \in \mathbb{R}^{F \times N_{dir} \times 1}$$

**Variable Breakdown:**
*   $T$: The terminal schedule input.
*   $F$: **Forecast Horizon**. How many future time steps we want to predict (e.g., 6 steps = 30 mins).
*   $s$: The specific distance bin index for the **Start Station** (Terminal).
*   $H(t+F-1, \dots)$: The scheduled headway at the terminal at the end of the forecast.

### C. Target Output Tensor ($Y$) - Equation 6
Captures the actual future state of the system (The "Windshield View").

$$Y = [H(t, j, k), \dots, H(t+F-1, j, k)] \in \mathbb{R}^{F \times N_d \times N_{dir} \times 1}$$

**Variable Breakdown:**
*   $Y$: The "Ground Truth" or "Answer Key" the model tries to predict.
*   It has the same dimensions as the prediction output: $F$ steps into the future, across all $N_d$ distance bins.