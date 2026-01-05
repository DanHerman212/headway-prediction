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
        Plots predictions stitched with history: Input (30m) + Prediction (15m).
        Matches the style of the research abstract with a dotted line separating past/future.
        """
        # 1. Fetch a single batch
        for inputs, targets in dataset.take(1):
            preds = model.predict(inputs, verbose=0)
            
            # Unpack Inputs (we need the history to stitch the full view)
            # inputs is a dict: {'headway_input': ..., 'schedule_input': ...}
            past_data = inputs['headway_input']

            # Extract sample, direction, channel 0, and denormalize (* 30 min)
            # Shape: (Time, Stations) -> Transpose to (Stations, Time) for imshow
            
            # Past: (30, Stations)
            past = past_data[sample_idx, :, :, direction, 0].numpy().T * 30
            
            # Future True: (15, Stations)
            future_true = targets[sample_idx, :, :, direction, 0].numpy().T * 30
            
            # Future Pred: (15, Stations)
            future_pred = preds[sample_idx, :, :, direction, 0].T * 30

            # Stitch them together along the time axis (axis 1)
            # Result: (Stations, 45)
            full_true = np.concatenate([past, future_true], axis=1)
            full_pred = np.concatenate([past, future_pred], axis=1)

            # 2. Setup Plot
            fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
            
            # Common settings
            cmap = 'RdYlGn_r'  # Green=Fast (Low Headway), Red=Slow (High Headway)
            vmin, vmax = 0, 30
            
            # Helper to plotting
            def style_ax(ax, data, title):
                im = ax.imshow(data, aspect='auto', cmap=cmap, origin='lower', 
                             vmin=vmin, vmax=vmax, interpolation='nearest')
                
                # Dotted Line at t=0 (Index 29.5 is the boundary between step 29 and 30)
                # Past is indices 0..29 (30 steps). Future starts at 30.
                ax.axvline(x=29.5, color='blue', linestyle='--', linewidth=1.5, alpha=0.8)
                
                # Labels
                ax.set_title(title, fontsize=12, pad=10)
                ax.set_xlabel("Time (Minutes relative to Now)", fontsize=10)
                ax.set_ylabel("Station ID", fontsize=10)
                
                # Custom X Ticks to show -30, 0, +15
                # Current indices: 0 (t-30), 30 (t=0), 45 (t+15)
                ax.set_xticks([0, 15, 30, 45])
                ax.set_xticklabels(["-30", "-15", "Now", "+15"])
                
                return im

            # Plot A: Actual Sequence
            style_ax(axes[0], full_true, "(a) Actual (Input + Future)")
            
            # Plot B: Predicted Sequence
            im = style_ax(axes[1], full_pred, "(b) Predicted (Input + Model Output)")
            
            # Hide Y labels on second plot to save space
            axes[1].set_yticks([])
            axes[1].set_ylabel("")

            # Colorbar
            cbar = fig.colorbar(im, ax=axes.ravel().tolist(), pad=0.02, aspect=30)
            cbar.set_label('Headway (min)', rotation=270, labelpad=15)
            
            # plt.suptitle(f"Spatiotemporal Traffic Flow (Direction {direction})", y=0.95)
            plt.show()
            break