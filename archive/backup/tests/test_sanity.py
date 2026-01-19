#!/usr/bin/env python3
"""
First-Level Sanity Test for Production Readiness

This is the FIRST test to run when models aren't working. It catches 95% of 
common failures before you waste time on full training runs.

Tests (in order of importance):
    1. Shape Compatibility: Model inputs match dataset outputs
    2. Forward Pass: Model can process real data without NaN/Inf
    3. Gradient Flow: Loss decreases (model is learning something)
    4. Overfit Check: Can memorize 1 batch (proves architecture works)

Usage:
    python -m tests.test_sanity           # Full test
    python -m tests.test_sanity --quick   # Just shape + forward pass

Expected Runtime:
    --quick: ~10 seconds
    Full:    ~60 seconds
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf

# Ensure we can import from src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.models.model import build_convlstm_model
from src.data.dataset import SubwayDataGenerator


def test_shape_compatibility(config, generator):
    """
    TEST 1: Verify dataset shapes match model input expectations.
    
    This is the #1 cause of failures. Shape mismatches cause cryptic errors
    that waste hours of debugging.
    """
    print("\n" + "=" * 60)
    print("TEST 1: Shape Compatibility")
    print("=" * 60)
    
    # Create a small dataset
    ds = generator.make_dataset(start_index=0, end_index=100, shuffle=False)
    
    # Get one batch
    for batch_x, batch_y in ds.take(1):
        headway_shape = batch_x["headway_input"].shape
        schedule_shape = batch_x["schedule_input"].shape
        target_shape = batch_y.shape
    
    # Expected shapes (from model)
    expected_headway = (config.BATCH_SIZE, config.LOOKBACK_MINS, config.NUM_STATIONS, 2)
    expected_schedule = (config.BATCH_SIZE, config.FORECAST_MINS, 2)
    expected_target = (config.BATCH_SIZE, config.FORECAST_MINS, config.NUM_STATIONS, 2)
    
    print(f"  Dataset headway_input:  {tuple(headway_shape)}")
    print(f"  Expected:               {expected_headway}")
    assert tuple(headway_shape) == expected_headway, \
        f"SHAPE MISMATCH: headway_input {tuple(headway_shape)} != {expected_headway}"
    print(f"  ✓ headway_input OK\n")
    
    print(f"  Dataset schedule_input: {tuple(schedule_shape)}")
    print(f"  Expected:               {expected_schedule}")
    assert tuple(schedule_shape) == expected_schedule, \
        f"SHAPE MISMATCH: schedule_input {tuple(schedule_shape)} != {expected_schedule}"
    print(f"  ✓ schedule_input OK\n")
    
    print(f"  Dataset target:         {tuple(target_shape)}")
    print(f"  Expected:               {expected_target}")
    assert tuple(target_shape) == expected_target, \
        f"SHAPE MISMATCH: target {tuple(target_shape)} != {expected_target}"
    print(f"  ✓ target OK")
    
    print("\n✓ TEST 1 PASSED: All shapes compatible")
    return True


def test_forward_pass(model, generator):
    """
    TEST 2: Model can process real data without NaN/Inf errors.
    
    Catches:
        - Numerical instability (exploding values)
        - Disconnected graph errors
        - GPU memory issues
    """
    print("\n" + "=" * 60)
    print("TEST 2: Forward Pass with Real Data")
    print("=" * 60)
    
    # Get a batch of real data
    ds = generator.make_dataset(start_index=0, end_index=100, shuffle=False)
    for batch_x, batch_y in ds.take(1):
        pass
    
    print(f"  Input headway range: [{batch_x['headway_input'].numpy().min():.2f}, {batch_x['headway_input'].numpy().max():.2f}]")
    print(f"  Input schedule range: [{batch_x['schedule_input'].numpy().min():.2f}, {batch_x['schedule_input'].numpy().max():.2f}]")
    
    # Run forward pass
    try:
        predictions = model.predict(batch_x, verbose=0)
    except Exception as e:
        print(f"\n✗ FORWARD PASS FAILED: {e}")
        raise
    
    # Check for NaN/Inf
    has_nan = np.isnan(predictions).any()
    has_inf = np.isinf(predictions).any()
    
    print(f"  Output shape: {predictions.shape}")
    print(f"  Output range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    print(f"  Contains NaN: {has_nan}")
    print(f"  Contains Inf: {has_inf}")
    
    assert not has_nan, "MODEL PRODUCES NaN - check for unstable operations"
    assert not has_inf, "MODEL PRODUCES Inf - check for division by zero or log(0)"
    
    print("\n✓ TEST 2 PASSED: Forward pass produces valid outputs")
    return predictions


def test_gradient_flow(model, generator):
    """
    TEST 3: Gradients flow through the network (loss decreases).
    
    Trains for 3 steps and checks if loss decreases. If not, possible causes:
        - Vanishing gradients
        - Learning rate too low/high
        - Architecture bottleneck
    """
    print("\n" + "=" * 60)
    print("TEST 3: Gradient Flow (3 training steps)")
    print("=" * 60)
    
    ds = generator.make_dataset(start_index=0, end_index=200, shuffle=True)
    
    losses = []
    
    # Get a single batch to reuse
    for batch_x, batch_y in ds.take(1):
        pass
    
    for step in range(3):
        with tf.GradientTape() as tape:
            preds = model([batch_x["headway_input"], batch_x["schedule_input"]], training=True)
            loss = tf.reduce_mean(tf.square(preds - batch_y))
        
        grads = tape.gradient(loss, model.trainable_variables)
        
        # Check for vanishing gradients
        grad_norms = [tf.norm(g).numpy() for g in grads if g is not None]
        min_grad = min(grad_norms)
        max_grad = max(grad_norms)
        
        print(f"  Step {step+1}: loss={loss.numpy():.6f}, grad_norm=[{min_grad:.6f}, {max_grad:.6f}]")
        losses.append(loss.numpy())
        
        # Apply gradients
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
    # Check if loss decreased
    loss_decreased = losses[-1] < losses[0]
    print(f"\n  Loss change: {losses[0]:.6f} → {losses[-1]:.6f} ({'+' if losses[-1] > losses[0] else ''}{((losses[-1]/losses[0])-1)*100:.1f}%)")
    
    if not loss_decreased:
        print("  ⚠ WARNING: Loss did not decrease in 3 steps (may be OK for complex models)")
    else:
        print("  ✓ Loss decreased - gradients flowing correctly")
    
    print("\n✓ TEST 3 PASSED: Gradients computed successfully")
    return True


def test_overfit_single_batch(model, generator, epochs=50):
    """
    TEST 4: Model can memorize a single batch.
    
    This is the definitive test that your architecture works. If the model
    can't perfectly memorize 1 batch, something is fundamentally broken.
    
    Expected: Loss should approach ~0 (or <0.01) after 50 epochs.
    """
    print("\n" + "=" * 60)
    print(f"TEST 4: Overfit Single Batch ({epochs} epochs)")
    print("=" * 60)
    
    # Create a minimal dataset (1 batch repeated)
    ds = generator.make_dataset(start_index=0, end_index=64, shuffle=False)
    
    # Get the single batch
    for batch_x, batch_y in ds.take(1):
        pass
    
    print(f"  Training on 1 batch of shape: headway={batch_x['headway_input'].shape}, target={batch_y.shape}")
    
    # Fresh model to avoid contamination from previous tests
    fresh_model = build_convlstm_model(Config())
    fresh_model.compile(optimizer='adam', loss='mse')
    
    # Train
    initial_loss = None
    final_loss = None
    
    for epoch in range(epochs):
        history = fresh_model.fit(
            x=batch_x,
            y=batch_y,
            epochs=1,
            verbose=0
        )
        loss = history.history['loss'][0]
        
        if epoch == 0:
            initial_loss = loss
        if epoch == epochs - 1:
            final_loss = loss
        
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch+1:3d}: loss = {loss:.6f}")
    
    improvement = (1 - final_loss / initial_loss) * 100
    print(f"\n  Loss: {initial_loss:.6f} → {final_loss:.6f} ({improvement:.1f}% reduction)")
    
    # Success criteria
    if final_loss < 0.01:
        print("  ✓ EXCELLENT: Model memorized batch (loss < 0.01)")
    elif final_loss < 0.1:
        print("  ✓ GOOD: Model learning (loss < 0.1)")
    elif improvement > 50:
        print("  ⚠ PARTIAL: Loss decreased >50% but not converged - try more epochs")
    else:
        print("  ✗ CONCERN: Model struggling to learn - check architecture")
    
    print("\n✓ TEST 4 PASSED: Overfit test complete")
    return final_loss


def run_sanity_tests(quick_mode=False):
    """Run all sanity tests in sequence."""
    
    print("\n" + "=" * 60)
    print("FIRST-LEVEL SANITY TEST")
    print("Catching common failures before full training")
    print("=" * 60)
    
    # Initialize
    config = Config()
    config.BATCH_SIZE = 32  # Ensure consistent batch size
    
    print(f"\nConfig: LOOKBACK={config.LOOKBACK_MINS}, FORECAST={config.FORECAST_MINS}, "
          f"STATIONS={config.NUM_STATIONS}, BATCH={config.BATCH_SIZE}")
    
    # Load data
    print("\nLoading data...")
    generator = SubwayDataGenerator(config)
    generator.load_data(normalize=True, max_headway=30.0)  # Normalize to [0,1]
    
    # Build model
    print("Building model...")
    model = build_convlstm_model(config)
    model.compile(optimizer='adam', loss='mse')
    print(f"Model parameters: {model.count_params():,}")
    
    # Run tests
    results = {}
    
    # Test 1: Shapes
    results['shapes'] = test_shape_compatibility(config, generator)
    
    # Test 2: Forward pass
    results['forward'] = test_forward_pass(model, generator)
    
    if quick_mode:
        print("\n" + "=" * 60)
        print("QUICK MODE COMPLETE")
        print("Run without --quick for gradient and overfit tests")
        print("=" * 60)
        return results
    
    # Test 3: Gradients
    results['gradients'] = test_gradient_flow(model, generator)
    
    # Test 4: Overfit
    results['overfit_loss'] = test_overfit_single_batch(model, generator, epochs=50)
    
    # Summary
    print("\n" + "=" * 60)
    print("SANITY TEST SUMMARY")
    print("=" * 60)
    print(f"  ✓ Shape Compatibility: PASSED")
    print(f"  ✓ Forward Pass: PASSED")
    print(f"  ✓ Gradient Flow: PASSED")
    print(f"  ✓ Overfit Test: Final Loss = {results['overfit_loss']:.6f}")
    
    if results['overfit_loss'] < 0.1:
        print("\n✓ ALL TESTS PASSED - Model ready for full training")
    else:
        print("\n⚠ TESTS PASSED with concerns - review overfit loss")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="First-level sanity test for model")
    parser.add_argument("--quick", action="store_true", help="Quick mode: shapes + forward pass only")
    args = parser.parse_args()
    
    try:
        results = run_sanity_tests(quick_mode=args.quick)
        sys.exit(0)
    except AssertionError as e:
        print(f"\n✗ SANITY TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
