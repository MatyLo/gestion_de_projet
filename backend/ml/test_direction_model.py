"""
Test Direction Predictor Model Only
"""
import joblib
import numpy as np
from circular_regressor import CircularRegressor

# Load direction model
model = joblib.load('wildfire_direction_predictor.joblib')

# Create 5 test samples (23 features each)
X_test = np.random.randn(5, 23)

# Predict directions
predictions = model.predict(X_test)

print("Direction Predictor Test")
print("=" * 40)
for i, direction in enumerate(predictions):
    direction = direction % 360
    print(f"Sample {i+1}: {direction:.1f}°")
