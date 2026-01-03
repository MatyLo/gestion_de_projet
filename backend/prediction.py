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

# Load models from the ml directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "ml")

classifier = joblib.load(os.path.join(MODELS_DIR, "wildfire_spread_classifier_advanced.joblib"))
regressor = joblib.load(os.path.join(MODELS_DIR, "wildfire_spread_regressor_advanced.joblib"))


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
    
    # Build feature vector (must match training order)
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

    # Visualization parameters
    wind_angle = math.radians((270 - wind_dir) % 360)
    
    # Higher brightness and wind speed increase base distance
    brightness_factor = min(1.5, brightness / 350)
    wind_factor = min(1.5, wind_speed / 10)
    base_km = 1.0 * brightness_factor * wind_factor
    spread_km = base_km * spread_ratio

    # Generate polygon points
    pts = []
    for i in range(8):
        angle = wind_angle + (i * 2 * math.pi / 8)
        
        # Distance varies by direction (further in wind direction)
        direction_factor = 0.5 + 0.5 * math.cos(angle - wind_angle)
        
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
            "wind_direction": wind_dir
        },
        "geometry": {"type": "Polygon", "coordinates": [coords]}
    })
    
    # Direction arrow
    arrow_end = [
        lng + math.cos(wind_angle) * spread_km / (111.32 * math.cos(math.radians(lat))),
        lat + math.sin(wind_angle) * spread_km / 111.32
    ]
    features_geo.append({
        "type": "Feature",
        "properties": {"type": "direction", "direction": wind_dir, "probability": spread_prob},
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
        "spread_direction": wind_dir,
        "spread_distance_km": spread_km,
        "environmental_data": env_data,
        "geojson": {"type": "FeatureCollection", "features": features_geo}
    }


def generate_heatmap_grid(lat: float, lng: float, brightness: float = 350.0, grid_size: int = 50) -> list:
    """
    Generate a heatmap grid showing fire spread probability across an area.
    
    Args:
        lat: Center latitude
        lng: Center longitude
        brightness: Fire brightness/intensity
        grid_size: Number of grid points (grid_size x grid_size)
    
    Returns:
        List of [lat, lng, probability] tuples for heatmap visualization
    """
    # Get base prediction for the center point
    base_result = predict_fire_spread(lat, lng, brightness)
    base_prob = base_result['spread_probability']
    spread_km = base_result['spread_distance_km']
    wind_dir = base_result['spread_direction']
    wind_angle = math.radians((270 - wind_dir) % 360)
    
    # Calculate grid bounds (2x the spread distance)
    radius_km = spread_km * 2
    lat_range = radius_km / 111.32
    lng_range = radius_km / (111.32 * math.cos(math.radians(lat)))
    
    heatmap_data = []
    
    # Generate grid
    for i in range(grid_size):
        for j in range(grid_size):
            # Calculate position in grid
            lat_offset = (i / grid_size - 0.5) * 2 * lat_range
            lng_offset = (j / grid_size - 0.5) * 2 * lng_range
            
            grid_lat = lat + lat_offset
            grid_lng = lng + lng_offset
            
            # Calculate distance from center
            distance_km = math.sqrt(
                (lat_offset * 111.32) ** 2 + 
                (lng_offset * 111.32 * math.cos(math.radians(lat))) ** 2
            )
            
            # Calculate angle from center
            angle = math.atan2(lat_offset, lng_offset)
            angle_diff = abs(angle - wind_angle)
            if angle_diff > math.pi:
                angle_diff = 2 * math.pi - angle_diff
            
            # Probability decreases with distance
            distance_factor = max(0, 1 - (distance_km / (spread_km * 2)))
            
            # Higher probability in wind direction
            direction_factor = 0.3 + 0.7 * (1 - angle_diff / math.pi)
            
            # Combine factors
            probability = base_prob * distance_factor * direction_factor
            
            # Only add points with significant probability
            if probability > 0.1:
                heatmap_data.append([grid_lat, grid_lng, probability])
    
    return heatmap_data

