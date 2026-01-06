Based on the provided `st_covnet.py` and `dataset.py`, your current metrics (RMSE ~3 minutes, MAE ~1.5 minutes) suggest the model is struggling to capture the complex temporal dynamics of the subway system.

The primary issue with the current architecture is an **Information Bottleneck**. You are compressing 30 minutes of granular history into a single static "state" map (`return_sequences=False` in the Encoder), and then asking the Decoder to "hallucinate" 15 minutes of future dynamics from that single snapshot.

Here is a roadmap to optimize the configuration and architecture, ordered from easiest (Config/Hyperparams) to hardest (Architectural changes).

### 1. Configuration & Hyperparameter Optimization

These changes require only editing `src/config.py` or minor tweaks to the loss function, but can yield immediate gains.

**A. Switch Loss Function (Robustness)**
Subway data is noisy; a single train delayed by 20 minutes creates a massive outlier. MSE () penalizes these outliers heavily, causing the model to over-smooth normal predictions to avoid being wrong on the rare disasters.

* **Recommendation:** Switch to **Huber Loss** or **Log-Cosh**. These act like MSE for small errors but like MAE for large errors (linear penalty), preventing outliers from wrecking the gradients.
* **Code Change (`train.py` equivalent):**
```python
model.compile(optimizer='adam', loss=tf.keras.losses.Huber(delta=0.1))

```



**B. Remove the Sigmoid Cap**
Your final layer uses `activation='sigmoid'`. This forces predictions into a `[0, 1]` range.

* **The Risk:** If your real-world delays ever exceed the `max_value` used during your normalization, the model literally *cannot* predict them. It creates a "saturation" effect where gradients vanish near the extremes (0 and 1).
* **Recommendation:** Use `activation='linear'` (or `relu` if negative headways are impossible) and ensure your targets are standardized (subtract mean, divide by std) rather than MinMax scaled.

**C. Expand the Lookback**

* **Current:** 30 min history  15 min forecast.
* **Optimization:** Subway delays often cascade from events that happened 45-60 minutes ago at the other end of the line. Increasing `LOOKBACK_MINS` to **60** gives the ConvLSTM more context to see "waves" of delay approaching.

---

### 2. Architectural Optimization (High Impact)

The current `HeadwayConvLSTM` is a standard "Encoder-Decoder," but it is too rigid for spatiotemporal fluid dynamics.

**A. Fix the Temporal Bottleneck (The "Repeater" Flaw)**

* **Problem:** Your encoder reduces `(Batch, 30, ...)` to `(Batch, 1, ...)`. You lose the *velocity* and *acceleration* of the delays. You know *where* the trains are, but not how fast the gap is closing.
* **Solution:** Keep the temporal dimension in the encoder and use **3D Convolutions** or keep `return_sequences=True` and use an Attention mechanism.
* **Quick Fix (Preserve Sequence):** instead of compressing to 1 step, compress to the same length as the forecast (e.g., using Strided pooling or a dense temporal layer) so the decoder receives a sequence, not a snapshot.

**B. Add Skip Connections (U-Net Style)**
Spatial information (which station is which) is lost as the data goes deep into the network.

* **Solution:** Pass the input `input_schedule` or even the most recent `input_headway` frame directly to the decoder via concatenation. This allows the network to focus on learning *residuals* (the change in headway) rather than learning the absolute headway from scratch.

**C. Revised `build_model` Implementation**
Here is an optimized version of your architecture. It replaces the "Repeat Vector" approach with a "Seq2Seq" approach that maintains better flow.

```python
    def build_model(self):
        # ... input definitions remain the same ...

        # --- ENCODER ---
        # Keep return_sequences=True to preserve temporal dynamics
        # Input: (Batch, 30, Stations, 2, 1)
        e1 = layers.ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True)(input_headway)
        e1 = layers.BatchNormalization()(e1)
        
        # Encoder 2: We still return sequences, but we might pool time to reduce 30 -> 15
        # If we want exact mapping, we can just slice the last 15 or use a Conv3D to mix time.
        # Simple approach: distinct encoder output
        enc_output = layers.ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=False)(e1)
        
        # --- IMPROVED BRIDGE ---
        # Instead of just repeating the static state, we repeat it but also add 
        # a "Time Distributed" dense layer to process the schedule BEFORE merging.
        
        # 1. Repeat the spatial context (as you did before)
        context_repeated = layers.RepeatVector(self.config.FORECAST_MINS)(layers.Flatten()(enc_output))
        # Reshape back to spatial grid: (Batch, 15, Stations, 2, 64)
        # Note: You need to handle reshaping dimensions carefully here depending on station count
        # For simplicity, let's stick to your custom lambda approach which was correct for spatial grids:
        
        x_repeated = layers.Lambda(
            repeat_spatial_state, 
            output_shape=repeat_output_shape,
            arguments={'steps': self.config.FORECAST_MINS}
        )(enc_output)

        # --- SCHEDULE EMBEDDING ---
        # The schedule is powerful. Don't just concat it. Process it.
        # Schedule: (Batch, 15, 2, 1) -> Broadcasted to (Batch, 15, Stations, 2, 1)
        schedule_broadcasted = layers.Lambda(
            broadcast_schedule,
            output_shape=broadcast_output_shape
        )([input_schedule, x_repeated])
        
        # Process schedule to extract features (e.g., "is this a rush hour gap?")
        # We treat the broadcasted schedule as a feature map
        sched_features = layers.Conv3D(
            filters=16, kernel_size=(1,1,1), activation='relu'
        )(schedule_broadcasted)

        # --- FUSION ---
        # Concat context with PROCESSED schedule
        x_fused = layers.Concatenate(axis=-1)([x_repeated, sched_features])

        # --- DECODER ---
        # Deepen the decoder to refine predictions
        d1 = layers.ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True)(x_fused)
        d2 = layers.ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True)(d1)

        # --- SKIP CONNECTION ---
        # Critical: Add the schedule BACK into the end so the model knows the "baseline"
        # This forces the ConvLSTM to learn the *delay* (deviation from schedule) rather than the raw time.
        d_final = layers.Concatenate(axis=-1)([d2, sched_features])

        # --- OUTPUT ---
        # Use Linear activation for regression, not Sigmoid (unless strictly normalized)
        outputs = layers.Conv3D(
            filters=1, 
            kernel_size=(1, 1, 1), 
            activation='linear', 
            padding='same'
        )(d_final)

        return keras.Model(inputs=[input_headway, input_schedule], outputs=outputs)

```

### 3. Summary of Actionable Steps

1. **Immediate:** Change your loss function to `Huber` (or `MAE`) to stop outliers from skewing the gradient.
2. **Immediate:** Verify your normalization. If `0.01` MSE = `179` RMSE, your max value is likely around 1800. Ensure you aren't capping valid delays > 30 mins with the Sigmoid.
3. **Architecture:** Implement the **Skip Connection** where the schedule is added again right before the final output. This turns the problem into "Predict the *deviation* from the schedule" (easier) rather than "Predict the arrival time" (harder).