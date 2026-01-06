# ConvLSTM for Spatiotemporal Data: Complete Research Plan

## Executive Summary

This document provides a comprehensive research, architecture design, and testing plan for building ConvLSTM (Convolutional LSTM) models for spatiotemporal data analysis. It consolidates best practices from academic literature, industry implementations, and lessons learned from this project.

**Primary Resources:**
- Original ConvLSTM Paper: Shi et al. (2015) "Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting"
- TensorFlow/Keras Documentation: ConvLSTM2D API
- PyTorch Implementation: torchvision ConvLSTM modules
- Applied Research: Traffic prediction, weather forecasting, video analysis

---

## Part 1: Theoretical Foundation

### 1.1 What is ConvLSTM?

ConvLSTM extends traditional LSTM by replacing fully-connected operations with convolutional operations, allowing the model to:
- **Preserve spatial structure** in data (e.g., grid topology, image dimensions)
- **Capture temporal dependencies** across sequential timesteps
- **Learn spatiotemporal patterns** jointly rather than separately

**Key Equation:**
```
Traditional LSTM: Matrix multiplication (loses spatial structure)
ConvLSTM: Convolution operation (preserves spatial structure)
```

### 1.2 When to Use ConvLSTM

**Ideal Use Cases:**
- Weather prediction (radar images over time)
- Traffic forecasting (road networks with temporal patterns)
- Video analysis (spatial frames + temporal sequences)
- Transit systems (station networks over time) ← **Your Use Case**

**When NOT to Use:**
- Purely temporal data without spatial structure (use LSTM)
- Static spatial data without temporal evolution (use CNN)
- Very long sequences (consider Transformers or TCN)

### 1.3 Core Architecture Principles

```
Input → ConvLSTM Encoder → State Representation → ConvLSTM Decoder → Output
           ↓                                              ↑
    Captures patterns                            Generates predictions
```

**Critical Components:**
1. **Cell State (C_t)**: Long-term spatial memory
2. **Hidden State (H_t)**: Short-term spatial activation
3. **Gates**: Input, Forget, Output (all use convolution)
4. **Kernel Size**: Defines spatial receptive field

---

## Part 2: Research Resources & Literature Review

### 2.1 Essential Papers (Reading Order)

1. **Foundation (MUST READ):**
   - Shi et al. (2015) - "Convolutional LSTM Network" [[NeurIPS](https://arxiv.org/abs/1506.04214)]
   - Introduces ConvLSTM, provides mathematical formulation
   
2. **Architecture Variants:**
   - Spatiotemporal Convolutional Networks (ST-ConvNet)
   - Trajectory GRU (extends ConvLSTM with trajectory-based attention)
   - PredRNN and PredRNN++ (recurrent architectures with spatiotemporal memory)

3. **Application Papers:**
   - Traffic Forecasting: "Deep Spatiotemporal Residual Networks" (ST-ResNet)
   - Weather: "Deep Learning for Precipitation Nowcasting"
   - Video Prediction: "Unsupervised Learning of Video Representations"

### 2.2 Official Implementations

**TensorFlow/Keras:**
```python
from tensorflow.keras.layers import ConvLSTM2D

# Official API
ConvLSTM2D(
    filters=64,              # Number of output filters
    kernel_size=(3, 3),      # Spatial convolution kernel
    padding='same',          # Preserve spatial dimensions
    return_sequences=True,   # Return full sequence or last step
    activation='tanh',       # Cell activation (default)
    recurrent_activation='sigmoid',  # Gate activation (default)
    data_format='channels_last'  # (batch, time, rows, cols, channels)
)
```

**Key Documentation:**
- TensorFlow: https://www.tensorflow.org/api_docs/python/tf/keras/layers/ConvLSTM2D
- Keras Guide: https://keras.io/examples/vision/conv_lstm/
- PyTorch: https://github.com/ndrplz/ConvLSTM_pytorch

### 2.3 Open Source Examples

**High-Quality Repositories:**
1. **Keras Examples**: Official precipitation nowcasting tutorial
2. **PyTorch ConvLSTM**: ndrplz/ConvLSTM_pytorch (well-documented)
3. **Traffic Prediction**: lehaifeng/T-GCN (Graph ConvLSTM)
4. **Video Analysis**: Multiple video prediction repos on Papers with Code

---

## Part 3: Architecture Design Plan

### 3.1 Input Data Requirements

**Shape Conventions:**
```python
# Input format: (Batch, Time, Height, Width, Channels)
# Your case: (Batch, Time, Stations, Directions, Features)

# Example:
batch_size = 32
lookback_steps = 30      # Historical timesteps
forecast_steps = 15      # Future timesteps to predict
spatial_dim_1 = 156      # Number of stations
spatial_dim_2 = 2        # Directions (uptown/downtown)
channels = 1             # Features per location
```

**Data Preparation Checklist:**
- ✅ Normalize to [0, 1] or standardize (mean=0, std=1)
- ✅ Handle missing values (interpolation or masking)
- ✅ Ensure consistent dtypes (float32 for memory efficiency)
- ✅ Split train/val/test chronologically (no data leakage)

### 3.2 Encoder-Decoder Architecture Blueprint

**Standard Pattern:**
```python
class ConvLSTMModel:
    def __init__(self, input_shape, output_steps):
        # input_shape: (time, spatial1, spatial2, channels)
        
        # ENCODER: Compress temporal information
        encoder_input = Input(shape=input_shape)
        
        # Layer 1: Extract low-level spatiotemporal features
        x = ConvLSTM2D(filters=32, kernel_size=(3,3), 
                       padding='same', return_sequences=True,
                       activation='tanh')(encoder_input)
        x = BatchNormalization()(x)
        
        # Layer 2: Extract high-level features
        x = ConvLSTM2D(filters=64, kernel_size=(3,3),
                       padding='same', return_sequences=False,
                       activation='tanh')(x)
        x = BatchNormalization()(x)
        # Output: (batch, spatial1, spatial2, 64)
        
        # BRIDGE: Project state to decoder
        # Option A: Repeat state for each output timestep
        # Option B: Use separate learned projection
        
        # DECODER: Generate future predictions
        decoder_input = RepeatVector(output_steps)(x)
        
        # Decode predictions
        x = ConvLSTM2D(filters=32, kernel_size=(3,3),
                       padding='same', return_sequences=True)(decoder_input)
        
        # OUTPUT: Project to desired channels
        output = TimeDistributed(Conv2D(filters=1, kernel_size=(1,1),
                                        activation='linear'))(x)
```

### 3.3 Architecture Patterns

**Pattern 1: Sequence-to-One (Compression)**
```python
# Use case: Classify video, detect anomalies
return_sequences = False  # Only return last timestep
```

**Pattern 2: Sequence-to-Sequence (Translation)**
```python
# Use case: Predict future frames, forecast time series
return_sequences = True  # Return all timesteps
```

**Pattern 3: Multi-Input Fusion (Your Use Case)**
```python
# Combine historical data + external features (e.g., schedules)
input_history = Input(shape=(30, 156, 2, 1))
input_schedule = Input(shape=(15, 2, 1))

# Encode history
encoded = ConvLSTM2D(...)(input_history)

# Broadcast and fuse schedule
schedule_broadcast = ... # Custom broadcast layer
fused = Concatenate()([encoded, schedule_broadcast])

# Decode
output = ConvLSTM2D(...)(fused)
```

### 3.4 Critical Design Decisions

| Decision | Options | Recommendation | Rationale |
|----------|---------|---------------|-----------|
| **Kernel Size** | (3,3), (5,5), (7,7) | Start with (3,3) | Smaller = local patterns, Larger = global context |
| **Filters** | 16, 32, 64, 128 | 32→64→32 pyramid | Gradual compression, prevent overfitting |
| **Padding** | 'same', 'valid' | 'same' | Preserve spatial dimensions |
| **Activation** | tanh, relu, swish | tanh (cell), sigmoid (gates) | tanh prevents gradient explosion |
| **Stacking** | 1-4 layers | 2-3 layers | Deeper = more capacity but harder to train |
| **Return Sequences** | True/False | Depends on task | True for seq2seq, False for compression |

---

## Part 4: Implementation Best Practices

### 4.1 CuDNN Acceleration (CRITICAL)

**Problem:** ConvLSTM is computationally expensive.  
**Solution:** Use CuDNN-optimized implementation (10-50x speedup on GPU).

**Requirements for CuDNN:**
```python
ConvLSTM2D(
    activation='tanh',                # MUST be tanh (not relu/swish)
    recurrent_activation='sigmoid',   # MUST be sigmoid
    recurrent_dropout=0,              # MUST be 0 (no recurrent dropout)
    unroll=False,                     # MUST be False
    use_bias=True,                    # Default is fine
)
```

**Validation:**
```python
# Check if CuDNN is used
import tensorflow as tf
print("GPUs Available:", tf.config.list_physical_devices('GPU'))

# This will use CuDNN if conditions are met
model.compile(...)
# Look for "Using CuDNN" in training logs
```

### 4.2 Memory Optimization

**Issue:** ConvLSTM stores intermediate states → high memory usage.

**Solutions:**
1. **Batch Size:** Reduce from 128 to 64 or 32 if OOM occurs
2. **Mixed Precision:** Use float16 for faster training
   ```python
   from tensorflow.keras import mixed_precision
   mixed_precision.set_global_policy('mixed_float16')
   ```
3. **Gradient Checkpointing:** Trade compute for memory
4. **Avoid Repeated Broadcasting:** Pre-compute static tensors

### 4.3 Shape Debugging

**Common Errors:**
```
❌ "Incompatible shapes: [32, 15, 156, 2, 64] vs [32, 156, 2, 64]"
❌ "Input 0 is incompatible with layer: expected ndim=5, found ndim=4"
```

**Debugging Strategy:**
```python
# Add shape prints after each layer
x = ConvLSTM2D(...)(input)
print(f"After ConvLSTM: {x.shape}")  # Should be 5D

# Use Lambda layers to inspect shapes during training
from tensorflow.keras.layers import Lambda
debug = Lambda(lambda x: tf.print("Shape:", tf.shape(x), output_stream=sys.stdout))(x)
```

### 4.4 Common Pitfalls and Solutions

| Pitfall | Symptom | Solution |
|---------|---------|----------|
| **Wrong data format** | Shape errors | Use `channels_last` (default) |
| **Not using CuDNN** | Very slow training | Check activation functions |
| **Exploding gradients** | NaN loss | Use gradient clipping, lower LR |
| **Memory overflow** | OOM errors | Reduce batch size, use mixed precision |
| **Wrong return_sequences** | Shape mismatch | Set based on encoder/decoder role |
| **Broadcasting errors** | Concatenation fails | Match dimensions explicitly |

---

## Part 5: Training Strategy

### 5.1 Loss Functions

**For Regression (Your Case):**
```python
# Option 1: MSE (penalizes large errors heavily)
loss = 'mse'

# Option 2: MAE (robust to outliers)
loss = 'mae'

# Option 3: Huber (combines MSE + MAE)
loss = tf.keras.losses.Huber(delta=1.0)

# Option 4: Custom weighted loss (emphasize certain regions/times)
def weighted_mse(y_true, y_pred):
    weights = tf.constant([1.0, 1.5, 2.0, ...])  # Higher weight for later timesteps
    return tf.reduce_mean(weights * tf.square(y_true - y_pred))
```

### 5.2 Optimization

**Recommended Optimizer:**
```python
optimizer = tf.keras.optimizers.Adam(
    learning_rate=1e-3,        # Start here
    clipvalue=1.0,             # Prevent gradient explosion
)

# Learning rate schedule
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=1000,
    decay_rate=0.9
)
```

### 5.3 Callbacks

```python
callbacks = [
    # Early stopping
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    ),
    
    # Model checkpoint
    tf.keras.callbacks.ModelCheckpoint(
        filepath='best_model.h5',
        monitor='val_loss',
        save_best_only=True
    ),
    
    # Learning rate reduction
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    ),
    
    # TensorBoard logging
    tf.keras.callbacks.TensorBoard(
        log_dir='./logs',
        histogram_freq=1
    )
]
```

### 5.4 Hyperparameter Tuning

**Priority Order:**
1. **Learning Rate:** [1e-4, 5e-4, 1e-3, 5e-3]
2. **Filters:** [32, 64, 128]
3. **Kernel Size:** [(3,3), (5,5)]
4. **Number of Layers:** [2, 3, 4]
5. **Batch Size:** [16, 32, 64, 128]

**Tools:**
- Keras Tuner
- Weights & Biases (wandb)
- Optuna

---

## Part 6: Testing and Validation Plan

### 6.1 Unit Tests

**Test 1: Shape Consistency**
```python
import unittest

class TestConvLSTM(unittest.TestCase):
    def test_encoder_output_shape(self):
        """Verify encoder produces correct output shape."""
        model = build_encoder()
        input_shape = (None, 30, 156, 2, 1)
        input_tensor = tf.keras.Input(shape=input_shape[1:])
        output = model(input_tensor)
        
        # Expected: (None, 30, 156, 2, 64) if return_sequences=True
        self.assertEqual(output.shape[-1], 64)
    
    def test_full_model_forward_pass(self):
        """Verify model runs without errors."""
        model = build_full_model()
        batch = tf.random.normal((32, 30, 156, 2, 1))
        output = model(batch, training=False)
        
        # Expected output: (32, 15, 156, 2, 1)
        self.assertEqual(output.shape, (32, 15, 156, 2, 1))
    
    def test_gradient_flow(self):
        """Verify gradients are computed correctly."""
        model = build_full_model()
        x = tf.random.normal((2, 30, 156, 2, 1))
        y = tf.random.normal((2, 15, 156, 2, 1))
        
        with tf.GradientTape() as tape:
            pred = model(x, training=True)
            loss = tf.reduce_mean(tf.square(y - pred))
        
        grads = tape.gradient(loss, model.trainable_variables)
        
        # Check no None gradients
        self.assertTrue(all(g is not None for g in grads))
```

**Test 2: Data Pipeline**
```python
def test_data_generator(self):
    """Verify data generator produces correct batches."""
    gen = SubwayDataGenerator(...)
    train_ds = gen.make_dataset()
    
    for (x_headway, x_schedule), y_target in train_ds.take(1):
        # Check shapes
        self.assertEqual(x_headway.shape[1], 30)  # Lookback
        self.assertEqual(x_schedule.shape[1], 15)  # Forecast
        self.assertEqual(y_target.shape[1], 15)    # Forecast
        
        # Check values in valid range
        self.assertTrue(tf.reduce_all(x_headway >= 0))
        self.assertTrue(tf.reduce_all(x_headway <= 1))
```

### 6.2 Integration Tests

**Test 3: Training Loop**
```python
def test_minimal_training(self):
    """Test model trains for a few steps without errors."""
    model = build_full_model()
    model.compile(optimizer='adam', loss='mse')
    
    # Create dummy data
    x = np.random.rand(100, 30, 156, 2, 1).astype('float32')
    y = np.random.rand(100, 15, 156, 2, 1).astype('float32')
    
    # Train for 2 epochs
    history = model.fit(x, y, epochs=2, batch_size=16, verbose=0)
    
    # Check loss decreased
    self.assertLess(history.history['loss'][-1], 
                   history.history['loss'][0])
```

### 6.3 Performance Tests

**Test 4: Overfitting Sanity Check**
```python
def test_overfit_single_batch(self):
    """Model should overfit a single batch (sanity check)."""
    model = build_full_model()
    model.compile(optimizer='adam', loss='mse')
    
    # Single batch
    x = np.random.rand(16, 30, 156, 2, 1).astype('float32')
    y = np.random.rand(16, 15, 156, 2, 1).astype('float32')
    
    # Train until near-zero loss
    model.fit(x, y, epochs=100, verbose=0)
    final_loss = model.evaluate(x, y, verbose=0)
    
    # Should achieve very low loss on same batch
    self.assertLess(final_loss, 0.01)
```

**Test 5: Benchmark Speed**
```python
import time

def test_training_speed(self):
    """Measure training throughput."""
    model = build_full_model()
    model.compile(optimizer='adam', loss='mse')
    
    x = np.random.rand(1000, 30, 156, 2, 1).astype('float32')
    y = np.random.rand(1000, 15, 156, 2, 1).astype('float32')
    
    start = time.time()
    model.fit(x, y, epochs=1, batch_size=32, verbose=0)
    duration = time.time() - start
    
    print(f"Training 1000 samples took {duration:.2f}s")
    # Set threshold based on hardware
    self.assertLess(duration, 60)  # Should complete in <60s on GPU
```

### 6.4 Validation Metrics

**Metrics to Track:**
```python
# 1. Point-wise metrics
MAE = mean_absolute_error(y_true, y_pred)
RMSE = sqrt(mean_squared_error(y_true, y_pred))
MAPE = mean_absolute_percentage_error(y_true, y_pred)

# 2. Temporal metrics (per timestep)
mae_per_step = [mae(y_true[:, t], y_pred[:, t]) for t in range(15)]

# 3. Spatial metrics (per station)
mae_per_station = [mae(y_true[:, :, s], y_pred[:, :, s]) for s in range(156)]

# 4. Directional metrics
mae_uptown = mae(y_true[..., 0, :], y_pred[..., 0, :])
mae_downtown = mae(y_true[..., 1, :], y_pred[..., 1, :])
```

---

## Part 7: Debugging Guide

### 7.1 Symptom-Based Debugging

**Issue: NaN Loss**
```python
# Causes:
# 1. Learning rate too high
# 2. Gradient explosion
# 3. Bad data (inf, nan values)

# Solutions:
# Check data
assert not np.isnan(data).any()
assert not np.isinf(data).any()

# Clip gradients
optimizer = Adam(learning_rate=1e-4, clipvalue=1.0)

# Add batch normalization
x = ConvLSTM2D(...)(x)
x = BatchNormalization()(x)
```

**Issue: Model Not Learning (Flat Loss)**
```python
# Causes:
# 1. Learning rate too low
# 2. Dead neurons
# 3. Data not shuffled

# Solutions:
# Increase learning rate
optimizer = Adam(learning_rate=1e-3)  # from 1e-5

# Check activations
from tensorflow.keras import backend as K
activation_model = Model(inputs=model.input, 
                        outputs=model.layers[5].output)
activations = activation_model.predict(x_sample)
print(f"Mean activation: {activations.mean()}")  # Should not be 0

# Shuffle data
train_ds = train_ds.shuffle(10000)
```

**Issue: Out of Memory**
```python
# Solutions:
# 1. Reduce batch size
BATCH_SIZE = 16  # from 128

# 2. Use gradient accumulation
# 3. Enable mixed precision
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

# 4. Clear session between runs
from tensorflow.keras import backend as K
K.clear_session()
```

### 7.2 Visualization Tools

```python
# Plot predictions vs ground truth
def visualize_predictions(model, x_test, y_test, idx=0):
    pred = model.predict(x_test[idx:idx+1])
    
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    for t in range(15):
        ax = axes[t // 5, t % 5]
        ax.plot(y_test[idx, t, :, 0, 0], label='True')
        ax.plot(pred[0, t, :, 0, 0], label='Pred')
        ax.set_title(f't+{t+1}')
        ax.legend()
    plt.show()

# Plot training curves
def plot_training_history(history):
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
```

---

## Part 8: Production Checklist

### 8.1 Pre-Deployment

- [ ] Model achieves acceptable validation metrics
- [ ] Model passes all unit tests
- [ ] Model inference time meets requirements
- [ ] Model size is acceptable (<500MB recommended)
- [ ] Model can run on target hardware (CPU/GPU)
- [ ] Data preprocessing is reproducible
- [ ] Model versioning is implemented

### 8.2 Deployment Considerations

**Model Export:**
```python
# SavedModel format (recommended)
model.save('model_v1', save_format='tf')

# Or HDF5 format
model.save('model_v1.h5')

# TFLite for mobile/edge
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
```

**Inference Optimization:**
```python
# Batch predictions for efficiency
def predict_batch(model, inputs, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    dataset = dataset.batch(batch_size)
    predictions = model.predict(dataset)
    return predictions

# Use model.predict() instead of model() for production
# predict() is optimized for inference
```

---

## Part 9: Advanced Topics

### 9.1 Attention Mechanisms

**Add attention to focus on important spatial regions:**
```python
class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.conv = Conv2D(1, kernel_size=1, activation='sigmoid')
    
    def call(self, x):
        # x: (batch, time, height, width, channels)
        attention_weights = self.conv(x)
        return x * attention_weights
```

### 9.2 Multi-Scale Processing

**Capture patterns at different spatial scales:**
```python
# Parallel branches with different kernel sizes
branch_3x3 = ConvLSTM2D(filters=32, kernel_size=(3,3), padding='same')(x)
branch_5x5 = ConvLSTM2D(filters=32, kernel_size=(5,5), padding='same')(x)
branch_7x7 = ConvLSTM2D(filters=32, kernel_size=(7,7), padding='same')(x)

# Concatenate
multi_scale = Concatenate()([branch_3x3, branch_5x5, branch_7x7])
```

### 9.3 Transfer Learning

**Pre-train on related tasks:**
```python
# Pre-train encoder on reconstruction task
encoder_input = Input(shape=(30, 156, 2, 1))
encoded = ConvLSTM2D(...)(encoder_input)
reconstructed = ConvLSTM2D(...)(encoded)

autoencoder = Model(encoder_input, reconstructed)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(x_train, x_train, epochs=10)

# Extract encoder and freeze weights
encoder = Model(encoder_input, encoded)
encoder.trainable = False

# Fine-tune on prediction task
```

---

## Part 10: Project-Specific Recommendations

### 10.1 For Your Subway Headway Prediction

**Strengths of Current Approach:**
- ✅ Dual input design (history + schedule) is sound
- ✅ Using ConvLSTM for spatial station network is appropriate
- ✅ Encoder-decoder structure is standard practice

**Suggested Improvements:**

1. **Replace RepeatVector with Temporal Slicing:**
   ```python
   # Instead of repeating static state
   # Use the last 15 steps of the 30-step encoding
   bridge = Lambda(lambda x: x[:, -15:, :, :, :])(encoded_sequence)
   ```

2. **Asymmetric Kernels for Schedule:**
   ```python
   # Separate spatial and temporal processing
   spatial_conv = Conv3D(filters=16, kernel_size=(1, 3, 3))(schedule)
   temporal_conv = Conv3D(filters=16, kernel_size=(3, 1, 1))(spatial_conv)
   ```

3. **Add Skip Connections:**
   ```python
   # Connect encoder directly to decoder
   decoder_input = Concatenate()([bridge, schedule_features, encoder_output])
   ```

### 10.2 Experimental Roadmap

**Phase 1: Baseline (Week 1-2)**
- Implement basic ConvLSTM encoder-decoder
- Train on small subset (1 month data)
- Target: MAE < 5 minutes

**Phase 2: Optimization (Week 3-4)**
- Add schedule fusion
- Tune hyperparameters
- Target: MAE < 3 minutes

**Phase 3: Advanced (Week 5-6)**
- Add attention mechanisms
- Multi-scale processing
- Target: MAE < 2 minutes

**Phase 4: Production (Week 7-8)**
- Model compression
- Deployment pipeline
- Real-time inference testing

---

## Conclusion

This research plan provides a complete framework for building ConvLSTM models for spatiotemporal data. Key takeaways:

1. **Start Simple:** Basic encoder-decoder before adding complexity
2. **Test Rigorously:** Unit tests, integration tests, performance tests
3. **Debug Systematically:** Use shape prints, visualization, gradual complexity
4. **Optimize Strategically:** CuDNN acceleration, mixed precision, batch size
5. **Document Everything:** Architecture decisions, hyperparameters, results

**Next Steps:**
1. Review this document with your team
2. Set up testing framework (pytest + unittest)
3. Implement baseline model following Part 3
4. Run validation experiments from Part 6
5. Iterate based on results

**Resources:**
- Keep TensorFlow docs open: https://www.tensorflow.org/api_docs/python/tf/keras/layers/ConvLSTM2D
- Join communities: r/MachineLearning, Keras Slack
- Track experiments: Weights & Biases, MLflow

Good luck with your research! 🚀
