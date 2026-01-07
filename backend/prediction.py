"""
Fire Spread Prediction Module
Refactored from predict_spread.py for FastAPI integration
"""
import numpy as np
import joblib
import math
import os
import requests
from datetime import datetime
from typing import Dict, Optional, Tuple
import ee
ee.Initialize(project='airy-galaxy-471607-b6')

# Import CircularRegressor before loading the model (required for unpickling)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "ml")
import sys
sys.path.insert(0, MODELS_DIR)
from circular_regressor import CircularRegressor

# Load models from the ml directory
classifier = joblib.load(os.path.join(MODELS_DIR, "wildfire_spread_classifier_advanced.joblib"))
regressor = joblib.load(os.path.join(MODELS_DIR, "wildfire_spread_regressor_advanced.joblib"))
direction_predictor = joblib.load(os.path.join(MODELS_DIR, "wildfire_direction_predictor.joblib"))


def get_weather_data(lat: float, lng: float) -> Optional[Dict]:
    """Fetch real weather data for the fire location using Open-Meteo API"""
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lng}&current=temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m"
        response = requests.get(url, timeout=5)
        data = response.json()
        
        if 'current' in data:
            return {
                'temp_max': data['current']['temperature_2m'],
                'humidity': data['current']['relative_humidity_2m'],
                'wind_speed': data['current']['wind_speed_10m'],
                'wind_direction': data['current']['wind_direction_10m']
            }
    except Exception as e:
        print(f"Weather API error: {e}")
    
    return None


def get_elevation_data(lat: float, lng: float) -> float:
    """Fetch elevation data for the fire location using Open-Meteo Elevation API"""
    try:
        url = f"https://api.open-meteo.com/v1/elevation?latitude={lat}&longitude={lng}"
        response = requests.get(url, timeout=5)
        data = response.json()
        
        if 'elevation' in data and len(data['elevation']) > 0:
            return data['elevation'][0]
    except Exception as e:
        print(f"Elevation API error: {e}")
    
    return 500  # Default elevation


def get_drought_vegetation(lat: float, lng: float) -> Tuple[float, float]:
    """Generate realistic drought and vegetation values based on location and season"""
    current_month = datetime.now().month
    
    # US regions with different climate patterns
    west_coast = lng < -115
    southwest = lng < -100 and lat < 37
    southeast = lng > -90 and lat < 36
    northeast = lng > -80 and lat > 36
    
    # Base drought index (0-5 scale, higher = more drought)
    if west_coast:
        base_drought = 4.0 if 5 <= current_month <= 10 else 2.0
    elif southwest:
        base_drought = 3.5
    elif southeast:
        base_drought = 2.5
    elif northeast:
        base_drought = 1.5
    else:
        base_drought = 2.0
    
    # Add randomness (±1.0)
    drought = max(0, min(5, base_drought + (np.random.random() * 2 - 1)))
    
    # Vegetation index (0-1 scale, higher = more vegetation)
    if west_coast:
        base_veg = 0.4
    elif southwest:
        base_veg = 0.3
    elif southeast:
        base_veg = 0.7
    elif northeast:
        base_veg = 0.6
    else:
        base_veg = 0.5
    
    # Seasonal adjustment
    if 3 <= current_month <= 8:  # Spring/Summer
        base_veg += 0.2
    
    # Adjust for drought (inverse relationship)
    veg_adjusted = base_veg * (1 - (drought / 7))
    
    # Add randomness (±0.15)
    vegetation = max(0.1, min(0.9, veg_adjusted + (np.random.random() * 0.3 - 0.15)))
    
    return drought, vegetation


def prepare_direction_features(
    elevation: float,
    wind_dir: float,
    wind_speed: float,
    temp_min: float,
    temp_max: float,
    humidity: float,
    drought: float,
    vegetation: float,
    brightness: float
) -> np.ndarray:
    """
    Prepare 23 features for the direction prediction model.
    
    The model expects features in this exact order:
    - Environmental stats (mean, max) for 8 variables: 16 features
    - Fire mask stats: 2 features
    - Wind vector components: 2 features
    - Interaction terms: 2 features
    - Fire shape ratio: 1 feature
    """
    features = []
    
    # Since we have single-point values, use same value for mean and max
    # 1. Elevation (mean, max)
    features.extend([elevation, elevation])
    
    # 2. Wind direction (mean, max)
    features.extend([wind_dir, wind_dir])
    
    # 3. Wind speed (mean, max)
    features.extend([wind_speed, wind_speed])
    
    # 4. Temp min (mean, max)
    features.extend([temp_min, temp_min])
    
    # 5. Temp max (mean, max)
    features.extend([temp_max, temp_max])
    
    # 6. Humidity (mean, max) - model was trained with specific humidity (0-1 scale)
    humidity_normalized = humidity / 100.0
    features.extend([humidity_normalized, humidity_normalized])
    
    # 7. Drought (mean, max)
    features.extend([drought, drought])
    
    # 8. Vegetation/NDVI (mean, max)
    features.extend([vegetation, vegetation])
    
    # 9. Fire mask stats (fire_sum, fire_mean) - approximate from brightness
    fire_sum = brightness / 10.0
    fire_mean = brightness / 3500.0
    features.extend([fire_sum, fire_mean])
    
    # 10. Wind vector components (wind_east, wind_north)
    wind_east = wind_speed * math.cos(math.radians(wind_dir))
    wind_north = wind_speed * math.sin(math.radians(wind_dir))
    features.extend([wind_east, wind_north])
    
    # 11. Interaction terms
    wind_elevation = wind_speed * elevation
    drought_vegetation = drought * vegetation
    features.extend([wind_elevation, drought_vegetation])
    
    # 12. Fire shape ratio (default 1.0 for active point fire)
    features.append(1.0)
    
    return np.array(features, dtype=float).reshape(1, -1)


def predict_direction(
    elevation: float,
    wind_dir: float,
    wind_speed: float,
    temp_min: float,
    temp_max: float,
    humidity: float,
    drought: float,
    vegetation: float,
    brightness: float
) -> float:
    """
    Predict fire spread direction using the trained ML model.
    
    Returns:
        Direction in degrees (0-360) where 0=North, 90=East, 180=South, 270=West
    """
    features = prepare_direction_features(
        elevation, wind_dir, wind_speed, temp_min, temp_max,
        humidity, drought, vegetation, brightness
    )
    
    direction_deg = direction_predictor.predict(features)[0]
    return float(direction_deg % 360)


def predict_fire_spread(lat: float, lng: float, brightness: float = 350.0) -> Dict:
    """
    Predict fire spread using both classifier and regressor models.
    
    Args:
        lat: Latitude of fire location
        lng: Longitude of fire location
        brightness: Fire brightness/intensity (default: 350)
    
    Returns:
        Dictionary containing prediction results and GeoJSON for visualization
    """
    # Get real weather data if possible
    weather = get_weather_data(lat, lng)
    
    # Use real data or reasonable defaults
    if weather:
        temp_max = weather['temp_max']
        humidity = weather['humidity']
        wind_speed = weather['wind_speed']
        wind_dir = weather['wind_direction']
        temp_min = temp_max - 10
        data_source = "weather_api"
    else:
        # Generate semi-random values that make sense together
        temp_factor = min(1.0, brightness / 400)
        temp_max = 20 + (temp_factor * 15) + (np.random.random() * 5)
        temp_min = temp_max - 10
        humidity = max(10, 60 - (temp_factor * 40) + (np.random.random() * 10))
        wind_speed = 5 + (np.random.random() * 20)
        wind_dir = np.random.randint(0, 360)
        data_source = "estimated"
    
    # Get elevation
    elevation = get_elevation_data(lat, lng)
    
    # Get drought and vegetation indices
    drought, vegetation = get_drought_vegetation(lat, lng)
    
    fire_intensity = brightness
    
    # Predict fire spread direction using ML model
    predicted_direction = predict_direction(
        elevation=elevation,
        wind_dir=wind_dir,
        wind_speed=wind_speed,
        temp_min=temp_min,
        temp_max=temp_max,
        humidity=humidity,
        drought=drought,
        vegetation=vegetation,
        brightness=brightness
    )
    
    # Build feature vector for classifier/regressor (must match training order)
    features = [
        elevation, elevation,
        wind_dir, wind_dir,
        wind_speed, wind_speed,
        temp_min, temp_min,
        temp_max, temp_max,
        humidity, humidity,
        drought, drought,
        vegetation, vegetation,
        fire_intensity, fire_intensity / 100.0,
        wind_speed * math.cos(math.radians(wind_dir)),
        wind_speed * math.sin(math.radians(wind_dir)),
        elevation * wind_speed,
        drought * vegetation,
        1.0  # default shape ratio placeholder
    ]
    X = np.array(features).reshape(1, -1)

    # Classifier prediction
    will_spread = bool(classifier.predict(X)[0])
    spread_prob = float(classifier.predict_proba(X)[0][1])

    # Regressor prediction (clipped)
    raw_ratio = float(regressor.predict(X)[0])
    spread_ratio = max(0.1, min(10.0, raw_ratio))

    # Visualization parameters - use ML-predicted direction
    spread_angle = math.radians((270 - predicted_direction) % 360)
    
    # Higher brightness and wind speed increase base distance
    brightness_factor = min(1.5, brightness / 350)
    wind_factor = min(1.5, wind_speed / 10)
    base_km = 1.0 * brightness_factor * wind_factor
    spread_km = base_km * spread_ratio

    # Generate polygon points
    pts = []
    for i in range(8):
        angle = spread_angle + (i * 2 * math.pi / 8)
        
        # Distance varies by direction (further in predicted spread direction)
        direction_factor = 0.5 + 0.5 * math.cos(angle - spread_angle)
        
        # Terrain and vegetation effects
        terrain_factor = 1.0
        veg_factor = 0.7 + (vegetation * 0.6)
        
        # Combined factors
        d = spread_km * direction_factor * terrain_factor * veg_factor
        
        lat_off = d / 111.32
        lng_off = d / (111.32 * math.cos(math.radians(lat)))
        pts.append({
            "lat": lat + lat_off * math.sin(angle),
            "lng": lng + lng_off * math.cos(angle),
            "probability": spread_prob * direction_factor
        })

    # Compose GeoJSON
    features_geo = []
    
    # Origin point
    features_geo.append({
        "type": "Feature",
        "properties": {"type": "origin", "intensity": fire_intensity},
        "geometry": {"type": "Point", "coordinates": [lng, lat]}
    })
    
    # Spread polygon
    coords = [[lng, lat]] + [[p['lng'], p['lat']] for p in pts] + [[lng, lat]]
    features_geo.append({
        "type": "Feature",
        "properties": {
            "type": "spread",
            "will_spread": int(will_spread),
            "probability": spread_prob,
            "spread_ratio": spread_ratio,
            "spread_distance_km": spread_km,
            "wind_direction": wind_dir,
            "predicted_direction": predicted_direction
        },
        "geometry": {"type": "Polygon", "coordinates": [coords]}
    })
    
    # Direction arrow (uses ML-predicted direction)
    arrow_end = [
        lng + math.cos(spread_angle) * spread_km / (111.32 * math.cos(math.radians(lat))),
        lat + math.sin(spread_angle) * spread_km / 111.32
    ]
    features_geo.append({
        "type": "Feature",
        "properties": {"type": "direction", "direction": predicted_direction, "probability": spread_prob},
        "geometry": {"type": "LineString", "coordinates": [[lng, lat], arrow_end]}
    })
    
    # Individual spread points
    for idx, p in enumerate(pts):
        features_geo.append({
            "type": "Feature",
            "properties": {"type": "spread_point", "probability": p['probability'], "index": idx},
            "geometry": {"type": "Point", "coordinates": [p['lng'], p['lat']]}  
        })

    # Environmental data for transparency
    env_data = {
        "elevation": elevation,
        "wind_direction": wind_dir,
        "wind_speed": wind_speed,
        "temperature": temp_max,
        "humidity": humidity,
        "drought": drought,
        "vegetation": vegetation,
        "brightness": brightness,
        "data_source": data_source
    }

    return {
        "will_spread": will_spread,
        "spread_probability": spread_prob,
        "spread_ratio": spread_ratio,
        "spread_direction": predicted_direction,
        "spread_distance_km": spread_km,
        "environmental_data": env_data,
        "geojson": {"type": "FeatureCollection", "features": features_geo}
    }

