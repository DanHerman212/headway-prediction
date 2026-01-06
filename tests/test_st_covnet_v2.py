import unittest
import numpy as np
import tensorflow as tf
import sys
import os

# Add project root to python path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.st_convnet_v2 import HeadwayConvLSTM

class TestHeadwayConvLSTMV2(unittest.TestCase):
    
    def setUp(self):
        # Use small dimensions for faster testing
        self.n_stations = 5 
        self.lookback = 30
        self.forecast = 15
        self.batch_size = 2
        
        self.builder = HeadwayConvLSTM(
            n_stations=self.n_stations,
            lookback=self.lookback, 
            forecast=self.forecast
        )

    def test_build_and_shapes(self):
        """Test that model builds, compiles, and infers consistent shapes"""
        
        print("\n--- Starting Architecture Test ---")
        
        # 1. Build Model
        # Note: calling 'built_model' as defined in your class
        model = self.builder.build_model()
        self.assertIsInstance(model, tf.keras.Model)
        
        # 2. Compile 
        # This catches graph disconnection errors (e.g. unused tensor inputs)
        model.compile(optimizer='adam', loss='mae') 

        # 3. Create Dummy Data
        # Input 1: Headway [Batch, Lookback, Station, Direction, Channel]
        x_headway = np.random.rand(self.batch_size, self.lookback, self.n_stations, 2, 1).astype(np.float32)
        
        # Input 2: Schedule [Batch, Forecast, Direction, Channel]
        x_schedule = np.random.rand(self.batch_size, self.forecast, 2, 1).astype(np.float32)

        print(f"Input Headway Shape: {x_headway.shape}")
        print(f"Input Schedule Shape: {x_schedule.shape}")

        # 4. Run Forward Pass (Prediction)
        try:
            output = model.predict([x_headway, x_schedule])
        except Exception as e:
            self.fail(f"Model prediction crashed: {e}")

        # 5. Verify Output Shape
        # Expected: [Batch, Forecast, Station, Direction, 1]
        expected_shape = (self.batch_size, self.forecast, self.n_stations, 2, 1)
        
        self.assertEqual(output.shape, expected_shape, 
                         f"Shape Mismatch! Expected {expected_shape}, got {output.shape}")
        
        print(f"SUCCESS: Model produced valid output with shape {output.shape}")
        print("--- Test Complete ---")

if __name__ == '__main__':
    unittest.main()