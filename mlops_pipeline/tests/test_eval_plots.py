
import sys
import os
import shutil
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.eval import generate_plots, generate_html_report, reconstruct_hour, get_peak_mask

def test_reconstruct_hour():
    print("Testing reconstruct_hour...")
    # 06:00 AM -> 90 degrees? No, 0 is midnight.
    # 0 = 0 rad, 6 = pi/2, 12 = pi, 18 = 3pi/2
    hours = np.array([0, 6, 12, 18])
    radians = 2 * np.pi * hours / 24
    
    sin_vals = np.sin(radians)
    cos_vals = np.cos(radians)
    
    reconstructed = reconstruct_hour(sin_vals, cos_vals)
    
    print(f"Input: {hours}")
    print(f"Reconstructed: {reconstructed}")
    
    # Check closeness
    assert np.allclose(hours, reconstructed, atol=1e-5)
    print("PASS: reconstruct_hour")

def test_get_peak_mask():
    print("Testing get_peak_mask...")
    # Peaks: 7-10, 16-19
    hours = np.array([5.0, 7.5, 9.9, 12.0, 16.5, 18.9, 20.0])
    expected = np.array([False, True, True, False, True, True, False])
    
    mask = get_peak_mask(hours)
    print(f"Hours: {hours}")
    print(f"Mask: {mask}")
    
    assert np.array_equal(mask, expected)
    print("PASS: get_peak_mask")

def test_visualization():
    print("Testing visualization generation...")
    
    output_dir = "test_output"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    # Fake Data
    N = 100
    y_true = np.random.uniform(100, 600, N)
    # y_pred is true + noise
    y_pred = y_true + np.random.normal(0, 50, N)
    
    # Fake hours
    hours = np.linspace(0, 23, N)
    hour_radians = 2 * np.pi * hours / 24
    hour_sin = np.sin(hour_radians)
    hour_cos = np.cos(hour_radians)
    
    # Run Plot Generation
    # Note: generate_plots saves to current working directory by default in the code,
    # or creates files. The code in eval.py does `plt.savefig("prediction_scatter.png")`.
    # It doesn't take output_dir as arg.
    # We should switch to the output dir context or run in place.
    
    cwd = os.getcwd()
    try:
        os.chdir(output_dir)
        plots = generate_plots(y_true, y_pred, hour_sin, hour_cos)
    finally:
        os.chdir(cwd)
        
    print(f"Generated plots: {plots}")
    assert len(plots) > 0
    # Expected parity plot and timeseries
    assert "prediction_parity_scatter.png" in plots
    assert "prediction_timeseries.png" in plots
    
    for p in plots:
        full_p = os.path.join(output_dir, p)
        assert os.path.exists(full_p)
        print(f"Verified {full_p} exists.")
        
    # Test HTML Generation
    metrics = {"mae": 50.5, "accuracy": 0.88}
    html_path = os.path.join(output_dir, "report.html")
    
    # We need to pass full paths to generate_html_report if it reads them?
    # generate_html_report reads `plot_files`. If we passed relative paths ["scatter.png"],
    # it expects them to be in CWD?
    # Let's inspect generate_html_report.
    # It opens plot_file.
    
    # If we are in output_dir, it works.
    
    # Let's clean paths to be absolute for the test
    abs_plots = [os.path.join(output_dir, p) for p in plots]
    
    generate_html_report(metrics, abs_plots, html_path)
    
    assert os.path.exists(html_path)
    print(f"Verified {html_path} exists.")
    
    print("PASS: visualization")
    
    # Cleanup
    # shutil.rmtree(output_dir)

if __name__ == "__main__":
    test_reconstruct_hour()
    test_get_peak_mask()
    test_visualization()
    print("ALL TESTS PASSED")
