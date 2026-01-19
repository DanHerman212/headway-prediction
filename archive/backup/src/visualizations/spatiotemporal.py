"""
Spatiotemporal visualization for headway predictions.

Creates heatmap comparisons of predicted vs actual headway matrices,
visualizing the spatial (station) and temporal (time) dimensions.
"""

from typing import Optional, TYPE_CHECKING
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import tensorflow as tf

if TYPE_CHECKING:
    from src.tracking import Tracker


class SpatiotemporalCallback(tf.keras.callbacks.Callback):
    """
    Visualizes spatiotemporal headway predictions as heatmaps.
    
    Creates a figure with:
    - Actual headway matrix (ground truth)
    - Predicted headway matrix
    - Difference (error) matrix
    - Per-station time series comparison
    
    This is specific to the headway prediction model architecture
    where predictions have shape (batch, time, stations, directions, 1).
    """
    
    def __init__(
        self,
        tracker: "Tracker",
        validation_data: Optional[tf.data.Dataset] = None,
        freq: int = 5,
        num_samples: int = 2,
        station_names: Optional[list] = None,
        max_headway: float = 30.0,
    ):
        """
        Args:
            tracker: Tracker instance for logging
            validation_data: Validation dataset
            freq: Log every N epochs
            num_samples: Number of samples to visualize
            station_names: Optional list of station names for axis labels
            max_headway: Maximum headway value (for denormalization)
        """
        super().__init__()
        self.tracker = tracker
        self.validation_data = validation_data
        self.freq = freq
        self.num_samples = num_samples
        self.station_names = station_names
        self.max_headway = max_headway
    
    def on_epoch_end(self, epoch, logs=None):
        """Generate and log visualization at epoch end."""
        if epoch % self.freq != 0:
            return
        
        if self.validation_data is None:
            return
        
        try:
            # Get sample batch
            for x, y_true in self.validation_data.take(1):
                y_pred = self.model.predict(x, verbose=0)
                break
            
            # Generate visualizations for each sample
            for i in range(min(self.num_samples, len(y_true))):
                # Create heatmap comparison
                heatmap_image = self._create_heatmap_comparison(
                    y_true[i].numpy(),
                    y_pred[i],
                    title=f"Epoch {epoch} - Sample {i+1}"
                )
                self.tracker.log_image(
                    f"predictions/heatmap_sample_{i+1}",
                    heatmap_image,
                    step=epoch
                )
                
                # Create time series comparison for a few stations
                timeseries_image = self._create_timeseries_comparison(
                    y_true[i].numpy(),
                    y_pred[i],
                    title=f"Epoch {epoch} - Sample {i+1}"
                )
                self.tracker.log_image(
                    f"predictions/timeseries_sample_{i+1}",
                    timeseries_image,
                    step=epoch
                )
            
            self.tracker.flush()
            
        except Exception as e:
            print(f"Warning: Failed to create spatiotemporal visualization: {e}")
    
    def _create_heatmap_comparison(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = ""
    ) -> np.ndarray:
        """
        Create side-by-side heatmaps of actual vs predicted.
        
        Args:
            y_true: Ground truth (time, stations, directions, 1)
            y_pred: Predictions (time, stations, directions, 1)
            title: Figure title
            
        Returns:
            Image as numpy array (1, H, W, 3)
        """
        # Squeeze and denormalize
        y_true = y_true.squeeze() * self.max_headway
        y_pred = y_pred.squeeze() * self.max_headway
        
        # For multi-directional data, take mean across directions
        if y_true.ndim == 3:
            y_true = y_true.mean(axis=-1)
            y_pred = y_pred.mean(axis=-1)
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Common colorbar range
        vmin = 0
        vmax = max(y_true.max(), y_pred.max())
        
        # Actual
        im0 = axes[0].imshow(
            y_true.T,  # Transpose so stations are on y-axis
            cmap='viridis',
            aspect='auto',
            vmin=vmin,
            vmax=vmax
        )
        axes[0].set_title('Actual Headway')
        axes[0].set_xlabel('Time Step')
        axes[0].set_ylabel('Station')
        plt.colorbar(im0, ax=axes[0], label='Minutes')
        
        # Predicted
        im1 = axes[1].imshow(
            y_pred.T,
            cmap='viridis',
            aspect='auto',
            vmin=vmin,
            vmax=vmax
        )
        axes[1].set_title('Predicted Headway')
        axes[1].set_xlabel('Time Step')
        axes[1].set_ylabel('Station')
        plt.colorbar(im1, ax=axes[1], label='Minutes')
        
        # Difference (error)
        diff = y_true - y_pred
        diff_max = np.abs(diff).max()
        im2 = axes[2].imshow(
            diff.T,
            cmap='RdBu',
            aspect='auto',
            vmin=-diff_max,
            vmax=diff_max
        )
        axes[2].set_title('Error (Actual - Predicted)')
        axes[2].set_xlabel('Time Step')
        axes[2].set_ylabel('Station')
        plt.colorbar(im2, ax=axes[2], label='Minutes')
        
        # Add station labels if provided
        if self.station_names and len(self.station_names) <= 20:
            for ax in axes:
                ax.set_yticks(range(len(self.station_names)))
                ax.set_yticklabels(self.station_names, fontsize=6)
        
        fig.suptitle(title, fontsize=12)
        fig.tight_layout()
        
        # Convert to numpy array
        image = self._fig_to_array(fig)
        plt.close(fig)
        
        return image
    
    def _create_timeseries_comparison(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "",
        num_stations: int = 4
    ) -> np.ndarray:
        """
        Create time series plots for selected stations.
        
        Args:
            y_true: Ground truth (time, stations, directions, 1)
            y_pred: Predictions (time, stations, directions, 1)
            title: Figure title
            num_stations: Number of stations to plot
            
        Returns:
            Image as numpy array (1, H, W, 3)
        """
        # Squeeze and denormalize
        y_true = y_true.squeeze() * self.max_headway
        y_pred = y_pred.squeeze() * self.max_headway
        
        # For multi-directional data, take mean across directions
        if y_true.ndim == 3:
            y_true = y_true.mean(axis=-1)
            y_pred = y_pred.mean(axis=-1)
        
        # Select evenly spaced stations
        n_stations = y_true.shape[1]
        station_indices = np.linspace(0, n_stations - 1, num_stations, dtype=int)
        
        # Create figure
        fig, axes = plt.subplots(num_stations, 1, figsize=(12, 3 * num_stations), sharex=True)
        if num_stations == 1:
            axes = [axes]
        
        time_steps = np.arange(y_true.shape[0])
        
        for i, (ax, station_idx) in enumerate(zip(axes, station_indices)):
            # Get station name
            if self.station_names and station_idx < len(self.station_names):
                station_name = self.station_names[station_idx]
            else:
                station_name = f"Station {station_idx}"
            
            # Plot actual vs predicted
            ax.plot(time_steps, y_true[:, station_idx], 'b-', label='Actual', linewidth=2)
            ax.plot(time_steps, y_pred[:, station_idx], 'r--', label='Predicted', linewidth=2)
            
            # Calculate RMSE for this station
            rmse = np.sqrt(np.mean((y_true[:, station_idx] - y_pred[:, station_idx]) ** 2))
            
            ax.set_ylabel('Headway (min)')
            ax.set_title(f'{station_name} (RMSE: {rmse:.1f} min)')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, self.max_headway * 1.1)
        
        axes[-1].set_xlabel('Time Step')
        fig.suptitle(f'{title} - Time Series Comparison', fontsize=12)
        fig.tight_layout()
        
        # Convert to numpy array
        image = self._fig_to_array(fig)
        plt.close(fig)
        
        return image
    
    def _fig_to_array(self, fig) -> np.ndarray:
        """
        Convert matplotlib figure to numpy array.
        
        Args:
            fig: Matplotlib figure
            
        Returns:
            Image as numpy array (1, H, W, 3)
        """
        fig.canvas.draw()
        
        # Get the RGBA buffer
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape((h, w, 3))
        
        # Add batch dimension
        return buf[np.newaxis, ...]
