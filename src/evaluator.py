# Evaluator module for headway prediction model
# Provides training curves, metrics in real units, and paper-style visualizations
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from src.config import Config


class Evaluator:
    """
    Production-ready evaluator for headway prediction models.
    Converts normalized metrics to real-world units (seconds/minutes).
    Generates paper-style visualizations for interpretation.
    """
    
    def __init__(self, config: Config, scaler=None):
        """
        Args:
            config: Model configuration object
            scaler: Fitted sklearn scaler (MinMaxScaler) for inverse transform.
                   If None, assumes data range is [0, 30] minutes.
        """
        self.config = config
        self.scaler = scaler
        
        # Infer data range from scaler or use default
        if scaler is not None:
            self.data_min = scaler.data_min_[0]
            self.data_max = scaler.data_max_[0]
        else:
            self.data_min = 0.0
            self.data_max = 30.0  # Default assumption: headways up to 30 min
        
        self.data_range = self.data_max - self.data_min
    
    def _to_minutes(self, normalized_value):
        """Convert normalized value to minutes."""
        return normalized_value * self.data_range + self.data_min
    
    def _to_seconds(self, normalized_value):
        """Convert normalized value to seconds."""
        return self._to_minutes(normalized_value) * 60.0
    
    def _mae_to_seconds(self, mae_norm):
        """Convert normalized MAE to seconds."""
        return mae_norm * self.data_range * 60.0
    
    def _mse_to_rmse_seconds(self, mse_norm):
        """Convert normalized MSE to RMSE in seconds."""
        return np.sqrt(mse_norm) * self.data_range * 60.0

    def plot_training_curves(self, history, save_path=None):
        """
        Plots training curves with dual y-axis: normalized and real units.
        
        Args:
            history: Keras History object from model.fit()
            save_path: Optional path to save the figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        epochs = range(1, len(history.history['loss']) + 1)
        
        # --- RMSE Curve (already in seconds from custom metric) ---
        ax1 = axes[0]
        
        # Check if using custom rmse_seconds metric or raw loss
        if 'rmse_seconds' in history.history:
            ax1.plot(epochs, history.history['rmse_seconds'], 'b-', label='Train RMSE', linewidth=2)
            ax1.plot(epochs, history.history['val_rmse_seconds'], 'r-', label='Val RMSE', linewidth=2)
            ax1.set_ylabel('RMSE (seconds)', fontsize=11)
            # Add target lines
            ax1.axhline(y=60, color='g', linestyle='--', label='60s target', alpha=0.7)
            ax1.axhline(y=90, color='orange', linestyle='--', label='90s target', alpha=0.7)
        else:
            # Fallback: convert MSE loss to RMSE seconds
            ax1.plot(epochs, history.history['loss'], 'b-', label='Train Loss', linewidth=2)
            ax1.plot(epochs, history.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
            ax1.set_ylabel('MSE (normalized)', fontsize=11)
            # Secondary y-axis for RMSE in seconds
            ax1_right = ax1.twinx()
            ax1_right.set_ylabel('RMSE (seconds)', color='gray', fontsize=10)
            y_lim = ax1.get_ylim()
            ax1_right.set_ylim(self._mse_to_rmse_seconds(y_lim[0]), 
                               self._mse_to_rmse_seconds(y_lim[1]))
            ax1_right.tick_params(axis='y', labelcolor='gray')
        
        ax1.set_xlabel('Epoch', fontsize=11)
        ax1.set_title('RMSE (seconds)', fontsize=12)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # --- R² Curve ---
        ax2 = axes[1]
        if 'r_squared' in history.history:
            ax2.plot(epochs, history.history['r_squared'], 'b-', label='Train R²', linewidth=2)
            ax2.plot(epochs, history.history['val_r_squared'], 'r-', label='Val R²', linewidth=2)
            ax2.axhline(y=0.9, color='g', linestyle='--', label='0.9 target', alpha=0.7)
            ax2.axhline(y=0.95, color='orange', linestyle='--', label='0.95 target', alpha=0.7)
            ax2.set_ylabel('R²', fontsize=11)
            ax2.set_title('R² (Coefficient of Determination)', fontsize=12)
            ax2.set_ylim(0, 1.05)
        elif 'mae' in history.history:
            # Fallback to MAE if R² not available
            ax2.plot(epochs, history.history['mae'], 'b-', label='Train MAE', linewidth=2)
            ax2.plot(epochs, history.history['val_mae'], 'r-', label='Val MAE', linewidth=2)
            ax2.set_ylabel('MAE (normalized)', fontsize=11)
            ax2.set_title('Mean Absolute Error', fontsize=12)
            # Secondary y-axis for MAE in seconds
            ax2_right = ax2.twinx()
            ax2_right.set_ylabel('MAE (seconds)', color='gray', fontsize=10)
            y_lim = ax2.get_ylim()
            ax2_right.set_ylim(self._mae_to_seconds(y_lim[0]), 
                               self._mae_to_seconds(y_lim[1]))
            ax2_right.tick_params(axis='y', labelcolor='gray')
        
        ax2.set_xlabel('Epoch', fontsize=11)
        ax2.legend(loc='lower right' if 'r_squared' in history.history else 'upper right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Training curves saved to {save_path}")
        plt.show()
    
    def print_metrics_summary(self, history):
        """
        Prints a formatted summary of training metrics in real-world units.
        Includes production readiness assessment using RMSE and R².
        """
        # Extract metrics - handle both new (rmse_seconds, r_squared) and legacy (mae) formats
        if 'val_rmse_seconds' in history.history:
            # New metrics: already in seconds
            best_rmse_sec = min(history.history['val_rmse_seconds'])
            final_rmse_sec = history.history['val_rmse_seconds'][-1]
        else:
            # Legacy: convert from MSE loss
            best_val_loss = min(history.history['val_loss'])
            final_val_loss = history.history['val_loss'][-1]
            best_rmse_sec = self._mse_to_rmse_seconds(best_val_loss)
            final_rmse_sec = self._mse_to_rmse_seconds(final_val_loss)
        
        if 'val_r_squared' in history.history:
            best_r2 = max(history.history['val_r_squared'])
            final_r2 = history.history['val_r_squared'][-1]
        else:
            best_r2 = None
            final_r2 = None
        
        print("=" * 60)
        print("📊 TRAINING METRICS SUMMARY")
        print("=" * 60)
        print(f"\n📈 Data Statistics:")
        print(f"   Headway range: {self.data_min:.1f} - {self.data_max:.1f} minutes")
        print(f"   Data span: {self.data_range:.1f} minutes ({self.data_range * 60:.0f} seconds)")
        
        print(f"\n🎯 Best Validation Performance:")
        print(f"   RMSE: {best_rmse_sec:.1f} seconds ({best_rmse_sec/60:.2f} minutes)")
        if best_r2 is not None:
            print(f"   R²:   {best_r2:.4f}")
        
        print(f"\n📉 Final Epoch Performance:")
        print(f"   RMSE: {final_rmse_sec:.1f} seconds ({final_rmse_sec/60:.2f} minutes)")
        if final_r2 is not None:
            print(f"   R²:   {final_r2:.4f}")
        
        # Production readiness criteria (using RMSE in seconds)
        print(f"\n" + "=" * 60)
        print("🚦 PRODUCTION READINESS ASSESSMENT")
        print("=" * 60)
        
        # RMSE criteria
        rmse_criteria = [
            ("Excellent (real-time displays)", 60),   # ≤ 60 seconds
            ("Good (trip planning)", 90),             # ≤ 90 seconds
            ("Acceptable (general info)", 120),       # ≤ 2 minutes
            ("Needs improvement", 180),               # > 3 minutes
        ]
        
        print("\n📏 RMSE Thresholds:")
        for level, threshold in rmse_criteria:
            status = "✅" if best_rmse_sec <= threshold else "❌"
            print(f"   {status} {level}: RMSE ≤ {threshold}s ({threshold/60:.1f} min)")
        
        # R² criteria
        if best_r2 is not None:
            print("\n📐 R² Thresholds:")
            r2_criteria = [
                ("Excellent", 0.95),
                ("Good", 0.90),
                ("Acceptable", 0.80),
            ]
            for level, threshold in r2_criteria:
                status = "✅" if best_r2 >= threshold else "❌"
                print(f"   {status} {level}: R² ≥ {threshold}")
        
        return {
            'best_rmse_seconds': best_rmse_sec,
            'final_rmse_seconds': final_rmse_sec,
            'best_r_squared': best_r2,
            'final_r_squared': final_r2,
        }

    # Keep legacy method for backward compatibility
    def plot_loss(self, history):
        """Legacy method - redirects to plot_training_curves."""
        self.plot_training_curves(history)

    def plot_spatiotemporal_prediction(self, model, dataset, sample_idx=0, direction=0, save_path=None):
        """
        Paper-style visualization (Figure 7): Side-by-side heatmaps of actual vs predicted.
        Shows Input (30min history) + Future (15min forecast) with dotted line at t=0.
        
        Matches Usama & Koutsopoulos (2025) Figure 7:
        - Left: Actual (Input + Ground Truth Future)
        - Right: Input + Model Predicted Future
        - Colorbar in SECONDS (0-1200s range like paper)
        - Dotted line separating past from future
        
        Args:
            model: Trained Keras model
            dataset: tf.data.Dataset to sample from
            sample_idx: Which sample in the batch to visualize
            direction: 0 or 1 for train direction
            save_path: Optional path to save the figure
        """
        # 1. Fetch a single batch
        for inputs, targets in dataset.take(1):
            preds = model.predict(inputs, verbose=0)
            
            # Unpack Inputs
            past_data = inputs['headway_input']

            # --- Inverse Transform to real units (seconds) ---
            def to_seconds(tensor):
                """Convert normalized tensor to seconds."""
                data = tensor.numpy() if hasattr(tensor, 'numpy') else tensor
                # Denormalize: value * range + min, then * 60 for seconds
                return (data * self.data_range + self.data_min) * 60.0

            past_data_sec = to_seconds(past_data)
            targets_sec = to_seconds(targets)
            preds_sec = to_seconds(preds)

            # Extract sample, direction, channel 0
            # Shape: (Time, Stations) -> Transpose to (Stations, Time) for imshow
            
            # Past: (30, Stations)
            past = past_data_sec[sample_idx, :, :, direction, 0].T
            
            # Future True: (15, Stations)
            future_true = targets_sec[sample_idx, :, :, direction, 0].T
            
            # Future Pred: (15, Stations)
            future_pred = preds_sec[sample_idx, :, :, direction, 0].T

            # Stitch them together along the time axis (axis 1)
            # Result: (Stations, 45)
            full_true = np.concatenate([past, future_true], axis=1)
            full_pred = np.concatenate([past, future_pred], axis=1)

            # 2. Setup Plot (matching paper Figure 7 style)
            fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
            
            # Paper uses 0-1200 seconds range (0-20 min)
            cmap = 'RdYlGn_r'  # Red=High headway (bad), Green=Low headway (good)
            vmin, vmax = 0, 1200  # Seconds (matches paper Figure 7)
            
            def style_ax(ax, data, title):
                im = ax.imshow(data, aspect='auto', cmap=cmap, origin='lower', 
                             vmin=vmin, vmax=vmax, interpolation='nearest')
                
                # Dotted Line at prediction boundary (between index 29 and 30)
                ax.axvline(x=29.5, color='black', linestyle=':', linewidth=2, alpha=0.9)
                
                # Add "Input" and "Predicted" labels like Figure 7
                ax.text(14, data.shape[0] + 2, 'Input', ha='center', fontsize=10, fontweight='bold')
                if 'Predicted' in title:
                    ax.text(37, data.shape[0] + 2, 'Predicted', ha='center', fontsize=10, fontweight='bold')
                
                # Labels matching paper style
                ax.set_title(title, fontsize=12, pad=20)
                ax.set_xlabel("Time (relative to prediction point)", fontsize=10)
                ax.set_ylabel("Station (Distance from Terminal)", fontsize=10)
                
                # X-axis: show actual time labels
                # 0-29 = past (-30 to -1 min), 30-44 = future (0 to +14 min)
                tick_positions = [0, 10, 20, 29.5, 35, 44]
                tick_labels = ['-30', '-20', '-10', '0', '+5', '+15']
                ax.set_xticks([0, 10, 20, 30, 35, 44])
                ax.set_xticklabels(['-30', '-20', '-10', '0', '+5', '+15'])
                
                return im

            # Plot A: Actual (like paper Figure 7a)
            style_ax(axes[0], full_true, "(a) Actual")
            
            # Plot B: Input + Predicted (like paper Figure 7b)
            im = style_ax(axes[1], full_pred, "(b) Input + Predicted")
            
            # Simplify right plot y-axis
            axes[1].set_yticks([])
            axes[1].set_ylabel("")

            # Colorbar matching paper (seconds, 0-1200)
            cbar = fig.colorbar(im, ax=axes.ravel().tolist(), pad=0.02, aspect=30,
                               ticks=[0, 200, 400, 600, 800, 1000, 1200])
            cbar.set_label('Headway (seconds)', rotation=270, labelpad=15, fontsize=11)
            
            direction_name = "Northbound" if direction == 0 else "Southbound"
            plt.suptitle(f"Headway Heatmaps - {direction_name} Direction", y=0.98, fontsize=13)
            
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"💾 Prediction heatmap saved to {save_path}")
            plt.show()
            break
    
    def full_evaluation(self, model, history, test_dataset, save_dir=None):
        """
        Complete evaluation pipeline for production deployment.
        
        Args:
            model: Trained Keras model
            history: Training history from model.fit()
            test_dataset: tf.data.Dataset for test evaluation
            save_dir: Directory to save evaluation artifacts
        
        Returns:
            dict: Evaluation metrics summary
        """
        import os
        
        print("\n" + "=" * 60)
        print("🔬 FULL MODEL EVALUATION")
        print("=" * 60)
        
        # 1. Print metrics summary
        metrics = self.print_metrics_summary(history)
        
        # 2. Plot training curves
        print("\n📈 Training Curves:")
        curves_path = os.path.join(save_dir, "training_curves.png") if save_dir else None
        self.plot_training_curves(history, save_path=curves_path)
        
        # 3. Plot spatiotemporal prediction
        print("\n🗺️ Spatiotemporal Prediction Visualization:")
        for direction in [0, 1]:
            direction_name = "northbound" if direction == 0 else "southbound"
            pred_path = os.path.join(save_dir, f"prediction_{direction_name}.png") if save_dir else None
            self.plot_spatiotemporal_prediction(
                model, test_dataset, 
                sample_idx=0, direction=direction,
                save_path=pred_path
            )
        
        # 4. Deployment recommendation
        print("\n" + "=" * 60)
        print("📋 PRODUCTION DEPLOYMENT RECOMMENDATION")
        print("=" * 60)
        
        rmse_sec = metrics['best_rmse_seconds']
        r2 = metrics.get('best_r_squared')
        
        if rmse_sec <= 60:
            print(f"\n✅ READY FOR PRODUCTION")
            print(f"   Model achieves {rmse_sec:.1f}s RMSE - suitable for real-time displays")
        elif rmse_sec <= 90:
            print(f"\n✅ PRODUCTION VIABLE") 
            print(f"   Model achieves {rmse_sec:.1f}s RMSE - good for trip planning applications")
        elif rmse_sec <= 120:
            print(f"\n⚠️ ACCEPTABLE WITH CAVEATS")
            print(f"   Model achieves {rmse_sec:.1f}s RMSE - consider for general information only")
        else:
            print(f"\n❌ NEEDS IMPROVEMENT")
            print(f"   Model achieves {rmse_sec:.1f}s RMSE - consider hyperparameter tuning")
        
        if r2 is not None:
            if r2 >= 0.95:
                print(f"   R² = {r2:.4f} - Excellent explanatory power")
            elif r2 >= 0.90:
                print(f"   R² = {r2:.4f} - Good explanatory power")
            elif r2 >= 0.80:
                print(f"   R² = {r2:.4f} - Acceptable explanatory power")
            else:
                print(f"   R² = {r2:.4f} - Consider model improvements")
        
        return metrics