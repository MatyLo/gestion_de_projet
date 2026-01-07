"""
Prediction script for wildfire spread classifier and regressor models.
Tests both models with synthetic batch data.
"""

import numpy as np
from joblib import load
import json

# Feature names in the exact order expected by the models
FEATURE_NAMES = [
    "elevation_mean", "elevation_max", 
    "wind_dir_mean", "wind_dir_max",
    "wind_speed_mean", "wind_speed_max",
    "temp_min_mean", "temp_min_max",
    "temp_max_mean", "temp_max_max",
    "humidity_mean", "humidity_max",
    "drought_mean", "drought_max",
    "vegetation_mean", "vegetation_max",
    "fire_sum", "fire_mean",
    "wind_east", "wind_north",
    "wind_elevation", "drought_vegetation",
    "fire_shape_ratio"
]

NUM_FEATURES = len(FEATURE_NAMES)
print(f"Expected number of features: {NUM_FEATURES}")
print(f"Feature names: {FEATURE_NAMES}\n")


def create_test_batch(batch_size=5):
    """
    Create synthetic test data batch with realistic ranges.
    
    Args:
        batch_size: Number of samples to generate
    
    Returns:
        np.ndarray: Array of shape (batch_size, NUM_FEATURES)
    """
    np.random.seed(42)
    
    # Define realistic ranges for each feature (based on typical wildfire data)
    feature_ranges = {
        "elevation_mean": (0, 3000),           # meters
        "elevation_max": (100, 4000),          # meters
        "wind_dir_mean": (0, 360),             # degrees
        "wind_dir_max": (0, 360),              # degrees
        "wind_speed_mean": (0, 30),            # km/h
        "wind_speed_max": (0, 80),             # km/h
        "temp_min_mean": (-20, 20),            # Celsius
        "temp_min_max": (-10, 30),             # Celsius
        "temp_max_mean": (10, 40),             # Celsius
        "temp_max_max": (20, 50),              # Celsius
        "humidity_mean": (10, 100),            # percentage
        "humidity_max": (20, 100),             # percentage
        "drought_mean": (0, 1),                # normalized
        "drought_max": (0, 1),                 # normalized
        "vegetation_mean": (0, 1),             # normalized
        "vegetation_max": (0, 1),              # normalized
        "fire_sum": (0, 1000),                 # count
        "fire_mean": (0, 100),                 # count
        "wind_east": (-30, 30),                # m/s
        "wind_north": (-30, 30),               # m/s
        "wind_elevation": (-20, 20),           # m/s
        "drought_vegetation": (0, 1),          # interaction term (normalized)
        "fire_shape_ratio": (0.1, 5),          # ratio
    }
    
    # Generate batch
    batch_data = np.zeros((batch_size, NUM_FEATURES))
    
    for i, feature_name in enumerate(FEATURE_NAMES):
        min_val, max_val = feature_ranges[feature_name]
        batch_data[:, i] = np.random.uniform(min_val, max_val, batch_size)
    
    return batch_data


def load_models():
    """Load the trained models."""
    print("Loading models...")
    classifier = load('./wildfire_spread_classifier_advanced.joblib')
    regressor = load('./wildfire_spread_regressor_advanced.joblib')
    print("✓ Models loaded successfully\n")
    return classifier, regressor


def predict_batch(classifier, regressor, X_batch):
    """
    Make predictions on a batch of samples.
    
    Args:
        classifier: Trained classifier model
        regressor: Trained regressor model
        X_batch: Input features (batch_size, NUM_FEATURES)
    
    Returns:
        dict: Predictions and probabilities
    """
    # Classifier predictions (binary: spread/no-spread)
    spread_pred = classifier.predict(X_batch)
    spread_prob = classifier.predict_proba(X_batch)  # probabilities for each class
    
    # Regressor predictions (continuous: spread intensity/ratio)
    intensity_pred = regressor.predict(X_batch)
    
    return {
        'spread_class': spread_pred,
        'spread_probability': spread_prob,
        'spread_intensity': intensity_pred
    }


def format_results(X_batch, predictions):
    """
    Format results for easy reading.
    
    Args:
        X_batch: Input features
        predictions: Predictions dict
    
    Returns:
        list: Formatted results for each sample
    """
    results = []
    
    for idx in range(X_batch.shape[0]):
        sample = {
            'sample_id': idx + 1,
            'input_features': {
                FEATURE_NAMES[i]: float(X_batch[idx, i]) 
                for i in range(NUM_FEATURES)
            },
            'predictions': {
                'spread_class': int(predictions['spread_class'][idx]),
                'spread_probability': {
                    'no_spread': float(predictions['spread_probability'][idx, 0]),
                    'spread': float(predictions['spread_probability'][idx, 1])
                },
                'spread_intensity': float(predictions['spread_intensity'][idx])
            }
        }
        results.append(sample)
    
    return results


def main():
    """Main execution."""
    print("=" * 70)
    print("WILDFIRE SPREAD PREDICTION - BATCH INFERENCE")
    print("=" * 70 + "\n")
    
    # Load models
    classifier, regressor = load_models()
    
    # Create test batch
    print("Creating synthetic test batch...")
    batch_size = 5
    X_batch = create_test_batch(batch_size)
    print(f"✓ Generated {batch_size} test samples")
    print(f"✓ Input shape: {X_batch.shape}\n")
    
    # Make predictions
    print("Making predictions...")
    predictions = predict_batch(classifier, regressor, X_batch)
    print("✓ Predictions completed\n")
    
    # Format and display results
    results = format_results(X_batch, predictions)
    
    print("=" * 70)
    print("PREDICTION RESULTS")
    print("=" * 70 + "\n")
    
    for result in results:
        print(f"Sample ID: {result['sample_id']}")
        print(f"  Input Features (first 5): {list(result['input_features'].items())[:5]}")
        print(f"  Spread Class: {'YES' if result['predictions']['spread_class'] == 1 else 'NO'}")
        print(f"  Spread Probability:")
        print(f"    - No Spread: {result['predictions']['spread_probability']['no_spread']:.4f}")
        print(f"    - Spread: {result['predictions']['spread_probability']['spread']:.4f}")
        print(f"  Intensity Prediction: {result['predictions']['spread_intensity']:.4f}")
        print()
    
    # Save results to JSON
    output_file = "predictions_output.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved to {output_file}\n")
    
    # Summary statistics
    print("=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    spread_count = sum(1 for r in results if r['predictions']['spread_class'] == 1)
    print(f"Samples with spread predicted: {spread_count}/{batch_size}")
    print(f"Average spread probability: {np.mean(predictions['spread_probability'][:, 1]):.4f}")
    print(f"Average intensity prediction: {np.mean(predictions['spread_intensity']):.4f}")
    print(f"Intensity range: [{np.min(predictions['spread_intensity']):.4f}, {np.max(predictions['spread_intensity']):.4f}]")


if __name__ == "__main__":
    main()
