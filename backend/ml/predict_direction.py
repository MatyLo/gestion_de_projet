"""
Complete Guide: Using the Wildfire Direction Predictor Model
============================================================

This script shows exactly how to:
1. Load the trained direction predictor model
2. Prepare input data (23 features)
3. Make predictions
4. Interpret the results
"""

import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import json


# ==========================================
# 📋 PART 1: UNDERSTANDING THE MODEL
# ==========================================

"""
MODEL ARCHITECTURE:
- Model File: wildfire_direction_predictor.joblib
- Model Type: Ensemble (combination of RandomForest, GradientBoosting, etc.)
- Input: 23 features (extracted from GEE data)
- Output: Fire propagation direction (0-360 degrees)
  * 0° = North
  * 90° = East
  * 180° = South
  * 270° = West
"""

# Feature order (CRITICAL - MUST be in this exact order)
FEATURE_NAMES = [
    # Environmental stats (mean, max) - 16 features
    "elevation_mean", "elevation_max", 
    "wind_dir_mean", "wind_dir_max",
    "wind_speed_mean", "wind_speed_max",
    "temp_min_mean", "temp_min_max",
    "temp_max_mean", "temp_max_max",
    "humidity_mean", "humidity_max",
    "drought_mean", "drought_max",
    "vegetation_mean", "vegetation_max",
    # Fire stats - 2 features
    "fire_sum", "fire_mean",
    # Wind vector components - 2 features
    "wind_east", "wind_north",
    # Interaction terms - 2 features
    "wind_elevation", "drought_vegetation",
    # Fire shape - 1 feature
    "fire_shape_ratio"
]

NUM_FEATURES = len(FEATURE_NAMES)
print(f"✓ Model expects {NUM_FEATURES} features")
print(f"✓ Feature order: {FEATURE_NAMES}\n")


# ==========================================
# 📂 PART 2: LOADING THE MODEL
# ==========================================

def load_direction_model(model_path='backend/ml/wildfire_direction_predictor.joblib'):
    """
    Load the trained direction prediction model.
    
    Args:
        model_path: Path to the joblib model file
    
    Returns:
        Loaded model object
    """
    try:
        model = joblib.load(model_path)
        print(f"✓ Model loaded: {model_path}")
        print(f"  Model type: {type(model).__name__}")
        return model
    except FileNotFoundError:
        print(f"✗ Model file not found: {model_path}")
        raise
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        raise


# ==========================================
# 🔧 PART 3: PREPARING FEATURES
# ==========================================

def prepare_features_from_gee_data(gee_data_dict):
    """
    Extract the 23 features from Google Earth Engine data.
    
    Args:
        gee_data_dict: Dictionary from collect_gee_data() containing:
            - elevation: 2D array (64x64)
            - th (wind direction): 2D array
            - vs (wind speed): 2D array
            - tmmn (temp min): 2D array
            - tmmx (temp max): 2D array
            - sph (specific humidity): 2D array
            - pdsi (drought): 2D array
            - NDVI (vegetation): 2D array
            - PrevFireMask: 2D array (previous fire locations)
    
    Returns:
        np.array: Shape (1, 23) ready for model prediction
    """
    
    features = []
    
    # ===== 1. ENVIRONMENTAL VARIABLES (mean & max) =====
    # Each environmental variable contributes 2 features (mean, max)
    
    env_vars = {
        'elevation': gee_data_dict['elevation'],
        'wind_dir': gee_data_dict['th'],
        'wind_speed': gee_data_dict['vs'],
        'temp_min': gee_data_dict['tmmn'],
        'temp_max': gee_data_dict['tmmx'],
        'humidity': gee_data_dict['sph'],
        'drought': gee_data_dict['pdsi'],
        'vegetation': gee_data_dict['NDVI']
    }
    
    for var_name, data_array in env_vars.items():
        flat_data = np.asarray(data_array).flatten()
        flat_data = np.nan_to_num(flat_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        features.append(np.mean(flat_data))  # mean
        features.append(np.max(flat_data))   # max
    
    print("✓ Environmental features (16): elevation, wind_dir, wind_speed, temp_min, temp_max, humidity, drought, vegetation")
    
    
    # ===== 2. FIRE MASK STATISTICS =====
    
    prev_fire_mask = np.asarray(gee_data_dict['PrevFireMask']).flatten()
    prev_fire_mask = np.nan_to_num(prev_fire_mask, nan=0.0)
    
    features.append(np.sum(prev_fire_mask))   # fire_sum: total fire pixels
    features.append(np.mean(prev_fire_mask))  # fire_mean: proportion of burned area
    
    print("✓ Fire statistics (2): fire_sum, fire_mean")
    
    
    # ===== 3. WIND VECTOR COMPONENTS =====
    # Convert wind direction and speed to East/North components
    
    wind_dir = np.asarray(gee_data_dict['th']).flatten()
    wind_speed = np.asarray(gee_data_dict['vs']).flatten()
    
    wind_dir = np.nan_to_num(wind_dir, nan=0.0)
    wind_speed = np.nan_to_num(wind_speed, nan=0.0)
    
    # Mean values
    wind_dir_mean = np.mean(wind_dir)
    wind_speed_mean = np.mean(wind_speed)
    
    # Convert to Cartesian components
    # East component: positive = wind from west blowing east
    # North component: positive = wind from south blowing north
    wind_east = wind_speed_mean * np.cos(np.radians(wind_dir_mean))
    wind_north = wind_speed_mean * np.sin(np.radians(wind_dir_mean))
    
    features.append(wind_east)
    features.append(wind_north)
    
    print(f"✓ Wind components (2): wind_east={wind_east:.2f}, wind_north={wind_north:.2f}")
    
    
    # ===== 4. INTERACTION TERMS =====
    # These capture non-linear relationships between variables
    
    elev_mean = np.mean(np.asarray(gee_data_dict['elevation']).flatten())
    drought_mean = np.mean(np.asarray(gee_data_dict['pdsi']).flatten())
    vegetation_mean = np.mean(np.asarray(gee_data_dict['NDVI']).flatten())
    
    wind_elevation = wind_speed_mean * elev_mean  # How elevation affects wind effects
    drought_vegetation = drought_mean * vegetation_mean  # Vegetation response to drought
    
    features.append(wind_elevation)
    features.append(drought_vegetation)
    
    print(f"✓ Interaction terms (2): wind_elevation={wind_elevation:.2f}, drought_vegetation={drought_vegetation:.2f}")
    
    
    # ===== 5. FIRE SHAPE RATIO =====
    # Quantifies how elongated the fire perimeter is
    
    try:
        fire_mask_2d = prev_fire_mask.reshape(64, 64)
        
        if np.sum(fire_mask_2d) > 0:
            y_idx, x_idx = np.where(fire_mask_2d > 0)
            
            # Bounding box dimensions
            fire_width = np.max(x_idx) - np.min(x_idx) + 1
            fire_height = np.max(y_idx) - np.min(y_idx) + 1
            
            # Ratio (width/height): values > 1 = wider than tall, < 1 = taller than wide
            fire_shape_ratio = fire_width / (fire_height + 1e-6)
        else:
            fire_shape_ratio = 1.0  # No fire = square
            
    except Exception as e:
        print(f"  ⚠️  Could not compute fire shape ratio: {e}")
        fire_shape_ratio = 1.0
    
    features.append(fire_shape_ratio)
    
    print(f"✓ Fire shape ratio (1): {fire_shape_ratio:.2f}")
    
    
    # ===== 6. FINALIZE =====
    
    features_array = np.array(features, dtype=float)
    
    # Final cleanup
    features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Verify we have exactly 23 features
    assert len(features_array) == NUM_FEATURES, \
        f"Feature count mismatch: got {len(features_array)}, expected {NUM_FEATURES}"
    
    print(f"\n✅ Features prepared: shape {features_array.shape}")
    
    return features_array.reshape(1, -1)  # Shape (1, 23)


# ==========================================
# 🎯 PART 4: MAKING PREDICTIONS
# ==========================================

def predict_fire_direction(model, features_array):
    """
    Predict fire propagation direction.
    
    Args:
        model: Loaded model
        features_array: Prepared features (shape 1x23)
    
    Returns:
        dict with direction and confidence
    """
    
    # Get prediction
    direction_deg = model.predict(features_array)[0]
    
    # Normalize to 0-360
    direction_deg = direction_deg % 360
    
    # Interpret direction
    direction_names = {
        (0, 45): "North",
        (45, 90): "Northeast",
        (90, 135): "East",
        (135, 180): "Southeast",
        (180, 225): "South",
        (225, 270): "Southwest",
        (270, 315): "West",
        (315, 360): "Northwest"
    }
    
    direction_name = "Unknown"
    for (start, end), name in direction_names.items():
        if (direction_deg >= start and direction_deg < end) or \
           (end == 360 and direction_deg >= start):
            direction_name = name
            break
    
    result = {
        'direction_degrees': float(direction_deg),
        'direction_name': direction_name,
        'interpretation': f"Fire will spread towards {direction_name} ({direction_deg:.1f}°)"
    }
    
    # Try to get confidence if model supports it
    if hasattr(model, 'predict_proba'):
        try:
            proba = model.predict_proba(features_array)
            result['confidence'] = float(np.max(proba))
        except:
            pass
    
    return result


# ==========================================
# 📊 PART 5: COMPLETE PREDICTION PIPELINE
# ==========================================

def predict_fire_direction_complete(gee_data_dict, model_path='backend/ml/wildfire_direction_predictor.joblib'):
    """
    Complete pipeline: GEE data → Features → Prediction
    
    Args:
        gee_data_dict: Data from collect_gee_data()
        model_path: Path to the model
    
    Returns:
        dict with prediction results
    """
    
    print("=" * 70)
    print("WILDFIRE DIRECTION PREDICTION - COMPLETE PIPELINE")
    print("=" * 70)
    
    # Step 1: Load model
    print("\n📂 Step 1: Loading model...")
    model = load_direction_model(model_path)
    
    # Step 2: Prepare features
    print("\n🔧 Step 2: Preparing features from GEE data...")
    features = prepare_features_from_gee_data(gee_data_dict)
    
    # Step 3: Make prediction
    print("\n🎯 Step 3: Making prediction...")
    prediction = predict_fire_direction(model, features)
    
    # Step 4: Display results
    print("\n" + "=" * 70)
    print("🔥 PREDICTION RESULTS")
    print("=" * 70)
    print(f"Direction: {prediction['direction_degrees']:.1f}°")
    print(f"Cardinal: {prediction['direction_name']}")
    print(f"Interpretation: {prediction['interpretation']}")
    
    if 'confidence' in prediction:
        print(f"Confidence: {prediction['confidence']:.2%}")
    
    return prediction


# ==========================================
# 🧪 PART 6: EXAMPLE USAGE
# ==========================================

def example_with_dummy_data():
    """
    Example with synthetic data (no GEE API needed)
    """
    
    print("\n" + "=" * 70)
    print("EXAMPLE: Using synthetic data")
    print("=" * 70)
    
    # Create dummy GEE data
    dummy_gee_data = {
        'elevation': np.ones((64, 64)) * 1000,
        'th': np.ones((64, 64)) * 90,      # Wind from West
        'vs': np.ones((64, 64)) * 10,      # 10 km/h wind speed
        'tmmn': np.ones((64, 64)) * 15,
        'tmmx': np.ones((64, 64)) * 25,
        'sph': np.ones((64, 64)) * 0.01,
        'pdsi': np.ones((64, 64)) * 0.5,   # Moderate drought
        'NDVI': np.ones((64, 64)) * 0.6,   # Healthy vegetation
        'PrevFireMask': np.zeros((64, 64))
    }
    
    # Add some fire history
    dummy_gee_data['PrevFireMask'][20:40, 20:40] = 1
    
    # Prepare features
    features = prepare_features_from_gee_data(dummy_gee_data)
    
    # Make prediction with random forest (simple model)
    print("\n📊 Feature summary:")
    for i, name in enumerate(FEATURE_NAMES):
        print(f"  {name:25s} = {features[0, i]:10.4f}")
    
    # Load and predict
    model = load_direction_model()
    prediction = predict_fire_direction(model, features)
    
    print(f"\n✅ With westerly wind (90°), fire predicted to spread: {prediction['direction_name']}")
    
    return prediction


# ==========================================
# 📝 PART 7: PRACTICAL INTEGRATION
# ==========================================

"""
HOW TO USE IN YOUR APPLICATION:

1. From Web Request (API endpoint):
   ---
   from predict_direction import predict_fire_direction_complete
   from recup_data import collect_all_data  # Your GEE collection function
   
   @app.post("/predict/direction")
   def predict_direction_endpoint(lat: float, lon: float, date: str):
       # Collect GEE data
       gee_data = collect_all_data(lat, lon, datetime.fromisoformat(date))
       
       # Get prediction
       prediction = predict_fire_direction_complete(gee_data)
       
       return prediction

2. From Batch Processing:
   ---
   import pandas as pd
   
   # Load CSV with fire locations
   fires_df = pd.read_csv('active_fires.csv')
   
   predictions = []
   for _, row in fires_df.iterrows():
       gee_data = collect_all_data(row['lat'], row['lon'], row['date'])
       pred = predict_fire_direction_complete(gee_data)
       predictions.append(pred)
   
   # Save results
   pd.DataFrame(predictions).to_csv('fire_directions.csv', index=False)

3. Real-time Monitoring:
   ---
   from datetime import datetime, timedelta
   
   # Get active fire locations from FIRMS API
   active_fires = get_firms_data()
   
   for fire in active_fires:
       # Use date 7 days ago (GEE has lag)
       analysis_date = datetime.now() - timedelta(days=7)
       
       gee_data = collect_all_data(fire['latitude'], fire['longitude'], analysis_date)
       direction = predict_fire_direction_complete(gee_data)
       
       # Alert system
       send_alert(fire['id'], direction['direction_degrees'])
"""


if __name__ == "__main__":
    # Run example with synthetic data
    example_with_dummy_data()
    
    print("\n" + "=" * 70)
    print("Ready to use! See 'PRACTICAL INTEGRATION' section in code.")
    print("=" * 70)
