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
        Generates side-by-side comparison of ground truth vs predictions in 'Micrograph' style.
        (a) Actual, (b) Predicted.
        """
        # 1. Fetch a single batch
        for inputs, targets in dataset.take(1):
            preds = model.predict(inputs, verbose=0)

            # Extract sample and convert to minutes (denormalize)
            # targets shape: (batch, time, stations, directions, channels)
            y_true = targets[sample_idx, :, :, direction, 0].numpy().T * 30
            y_pred = preds[sample_idx, :, :, direction, 0].T * 30

            # 2. Setup Plot
            fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
            
            # Common settings
            cmap = 'RdYlGn_r'  # Red-Yellow-Green reversed
            vmin, vmax = 0, 30
            
            # Plot A: Actual
            im1 = axes[0].imshow(y_true, aspect='auto', cmap=cmap, origin='lower', 
                               vmin=vmin, vmax=vmax, interpolation='nearest')
            axes[0].set_title("(a) Actual", fontsize=12, pad=10)
            axes[0].set_xlabel("Time (Future Steps)", fontsize=10)
            axes[0].set_ylabel("Station Index", fontsize=10)
            axes[0].grid(False)
            
            # Plot B: Predicted
            im2 = axes[1].imshow(y_pred, aspect='auto', cmap=cmap, origin='lower', 
                               vmin=vmin, vmax=vmax, interpolation='nearest')
            axes[1].set_title("(b) Predicted", fontsize=12, pad=10)
            axes[1].set_xlabel("Time (Future Steps)", fontsize=10)
            axes[1].set_yticks([])  # Hide Y axis for second plot
            axes[1].grid(False)

            # Add single colorbar
            cbar = fig.colorbar(im2, ax=axes.ravel().tolist(), pad=0.02, aspect=30)
            cbar.set_label('Headway (min)', rotation=270, labelpad=15)
            
            plt.tight_layout()
            plt.show()
            break