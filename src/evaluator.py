# plot loss: standard training curves
# plot spatiotemporal prediction
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from src.config import Config

class Evaluator:
    def __init__(self, config: Config):
        self.config = config

    def plot_loss(self, history):
        """Plots training vs validation loss from keras history object"""
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(loss) + 1)

        plt.figure(figsize=(10, 5))
        plt.plot(epochs, loss, 'b-', label='Training Loss')
        plt.plot(epochs, val_loss, 'r--', label='Validation Loss')
        plt.title('Training vs Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_spatiotemporal_prediction(self, model, dataset, sample_idx=0, direction=0):
        """
        generates side by side comparison of ground truth vs predictions

        args:
            model: trained keras model
            dataset: tf.data.Dataset to draw a batch from
            sample_idx: which sample in the batch to vizualize
            direction: 0 for northbound, 1 for southbound
        """

        # 1. fetch a single batch
        # we use .take(1) to get just one batch of data
        for inputs, targets in dataset.take(1):
            # inputs: ((headway, schedule), terminal target headway)
            # targets (batch, 15, stations, 2, 1)

            # 2 generate prediction
            preds = model.predict(inputs, verbose=0)

            # 3 extract the specific sample and direction
            # shape (15, stations) -> transpose to (Stations, 15) for plotting
            # we multiply by 230 to convert back to minutes (denormalize)
            y_true = targets[sample_idx, :, :, direction, 0].numpy().T * 30
            y_pred = preds[sample_idx, :, :, direction, 0].T * 30

            # 4 plot
            fig, axes = plt.subplots(1, 2, figsize=(18, 8))

            # plot ground truth
            # vmin/vmax ensures the color scale is consistent (0 to 30 mins)
            im1 = axes[0].imshow(y_true, aspect='auto', cmap='RdYlGn_r', origin='lower', vmin=0, vmax=30)
            axes[0].set_title(f"Ground Truth (Next {self.config.FORECAST_MINS} min)")
            axes[0].set_xlabel("Time Steps (Future)")
            axes[0].set_ylabel("Station Index")
            plt.colorbar(im1, ax=axes[0], label='Headway (min)')

            # Plot Prediction
            im2 = axes[1].imshow(y_pred, aspect='auto', cmap='RdYlGn_r', origin='lower', vmin=0, vmax=30)
            axes[1].set_title(f"Model Prediction (Next {self.config.FORECAST_MINS} min)")
            axes[1].set_xlabel("Time Steps (Future)")
            axes[1].set_ylabel("Station Index")
            plt.colorbar(im2, ax=axes[1], label='Headway (min)')

            plt.suptitle(f"Spatiotemporal Forecast (Direction {direction})")
            plt.tight_layout()
            plt.show()

            # stop after 1 batch
            break