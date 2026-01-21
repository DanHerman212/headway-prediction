"""Test that dataset batching is aligned with drop_remainder."""
import numpy as np
import tensorflow as tf

BATCH_SIZE = 64
LOOKBACK = 20
N_SAMPLES = 1000

X = np.random.randn(N_SAMPLES, 8).astype(np.float32)
X_split = X[:900]
max_samples = len(X_split) - LOOKBACK
X_split = X_split[:max_samples + LOOKBACK]

print(f"Total samples after lookback: {max_samples}")
print(f"Expected full batches: {max_samples // BATCH_SIZE}")
print(f"Last batch size: {max_samples % BATCH_SIZE}")

dataset_no_drop = tf.keras.utils.timeseries_dataset_from_array(
    data=X_split,
    targets=None,
    sequence_length=LOOKBACK,
    sequence_stride=1,
    shuffle=False,
    batch_size=BATCH_SIZE,
)

route_targets = X[LOOKBACK:900, 1:4]
headway_targets = X[LOOKBACK:900, 0:1]

target_dataset_no_drop = tf.data.Dataset.from_tensor_slices({
    'route': route_targets,
    'headway': headway_targets
}).batch(BATCH_SIZE, drop_remainder=False)

target_dataset_with_drop = tf.data.Dataset.from_tensor_slices({
    'route': route_targets,
    'headway': headway_targets
}).batch(BATCH_SIZE, drop_remainder=True)

input_batches = list(dataset_no_drop.as_numpy_iterator())
target_batches_no_drop = list(target_dataset_no_drop.as_numpy_iterator())
target_batches_with_drop = list(target_dataset_with_drop.as_numpy_iterator())

print(f"\nWithout drop_remainder:")
print(f"  Input batches: {len(input_batches)}")
print(f"  Target batches: {len(target_batches_no_drop)}")
print(f"  Last input batch shape: {input_batches[-1].shape}")
print(f"  Last target route shape: {target_batches_no_drop[-1]['route'].shape}")

print(f"\nWith drop_remainder on targets:")
print(f"  Input batches: {len(input_batches)}")
print(f"  Target batches: {len(target_batches_with_drop)}")
if len(target_batches_with_drop) > 0:
    print(f"  Last target route shape: {target_batches_with_drop[-1]['route'].shape}")

if len(input_batches) == len(target_batches_with_drop):
    print("\nResult: Batch counts match with drop_remainder=True")
else:
    print(f"\nResult: Mismatch {len(input_batches)} vs {len(target_batches_with_drop)}")
