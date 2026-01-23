from numpy import np
from tensorflow import tf
from tensorflow.keras import keras

n = 52_000
train_samples = n * 0.6
val_samples = n * 0.2

input_x = np.array()
input_t = np.array()
input_r = np.array()

sequence_length = 20
batch_size = 64

def create_datasets(
        input_x, 
        input_t, 
        input_r, 
        train_samples, 
        val_samples, 
        sequence_length, 
        batch_size
        ) -> tf.data.Dataset:
    # Train Dataset
    train_dataset = keras.utils.timeseries_dataset_from_array(
        data=input_x[:-sequence_length],
        targets=(input_t[sequence_length:, None],input_r[sequence_length:]),
        sequence_length=sequence_length,
        sampling_rate=1,
        batch_size=batch_size,
        shuffle=True,
        start_index=0,
        end_index=train_samples
    )