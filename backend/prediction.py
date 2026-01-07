"""
Fire Spread Prediction Module
Refactored from predict_spread.py for FastAPI integration
"""
import numpy as np
import joblib
import math
import os
import requests
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import ee
ee.Initialize(project='airy-galaxy-471607-b6')

# Load models from the ml directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "ml")

classifier = joblib.load(os.path.join(MODELS_DIR, "wildfire_spread_classifier_advanced.joblib"))
regressor = joblib.load(os.path.join(MODELS_DIR, "wildfire_spread_regressor_advanced.joblib"))
direction = joblib.load(os.path.join(MODELS_DIR, "wildfire_direction_predictor.joblib"))


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


def extract_features_from_dict(features_dict):
    """
    Extrait les features ML depuis un dict de tenseurs TensorFlow.
    DOIT être identique à votre fonction dans collect_gee_data !
    """
    
    # Convertir tensors en numpy
    data = {k: v.numpy() if hasattr(v, 'numpy') else v 
            for k, v in features_dict.items()}
    
    feature_list = []
    
    # 1. Environmental stats (mean, max)
    env_vars = ['elevation', 'th', 'vs', 'tmmn', 'tmmx', 'sph', 'pdsi', 'NDVI']
    
    for var_name in env_vars:
        arr = np.asarray(data[var_name], dtype=float).flatten()
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        feature_list.extend([np.mean(arr), np.max(arr)])
    
    # 2. Fire mask stats
    prev_fire = np.asarray(data['PrevFireMask'], dtype=float).flatten()
    prev_fire = np.nan_to_num(prev_fire, nan=0.0)
    feature_list.extend([np.sum(prev_fire), np.mean(prev_fire)])
    
    # 3. Wind vector components
    wind_dir = np.asarray(data['th'], dtype=float).flatten()
    wind_speed = np.asarray(data['vs'], dtype=float).flatten()
    wind_dir = np.nan_to_num(wind_dir, nan=0.0)
    wind_speed = np.nan_to_num(wind_speed, nan=0.0)
    
    wind_dir_mean = np.mean(wind_dir)
    wind_speed_mean = np.mean(wind_speed)
    
    wind_east = wind_speed_mean * np.cos(np.radians(wind_dir_mean))
    wind_north = wind_speed_mean * np.sin(np.radians(wind_dir_mean))
    
    feature_list.append(wind_east)
    feature_list.append(wind_north)
    
    # 4. Interactions
    elev_mean = np.mean(np.asarray(data['elevation'], dtype=float).flatten())
    drought_mean = np.mean(np.asarray(data['pdsi'], dtype=float).flatten())
    ndvi_mean = np.mean(np.asarray(data['NDVI'], dtype=float).flatten())
    
    feature_list.append(elev_mean * wind_speed_mean)
    feature_list.append(drought_mean * ndvi_mean)
    
    # 5. Fire shape ratio
    try:
        fire_mask_2d = prev_fire.reshape(64, 64)
        if np.sum(fire_mask_2d) > 0:
            y_idx, x_idx = np.where(fire_mask_2d > 0)
            fire_width = np.max(x_idx) - np.min(x_idx)
            fire_height = np.max(y_idx) - np.min(y_idx)
            shape_ratio = fire_width / (fire_height + 1e-6)
        else:
            shape_ratio = 1.0
    except:
        shape_ratio = 1.0
    
    feature_list.append(shape_ratio)
    
    # Nettoyage final
    features = np.array(feature_list, dtype=float)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    return features


def collect_gee_data(lat, lon, date):
    """
    Collecte les données depuis Google Earth Engine.
    
    ⚠️ IMPORTANT : 
    - Les données GEE ne sont pas en temps réel
    - Utiliser une date d'au moins 3-5 jours dans le passé
    - GRIDMET : délai de ~2 jours
    - VIIRS NDVI : délai de ~8 jours
    - PDSI : mis à jour tous les 5 jours
    
    Args:
        lat, lon: Coordonnées (WGS84)
        date: datetime object (doit être dans le passé)
    
    Returns:
        dict avec arrays 64×64 pour chaque variable
    """
    
    print(f"\n🔥 Collecte pour ({lat}, {lon}) le {date.strftime('%Y-%m-%d')}")
    
    # ==========================================
    # 1. VALIDATION ET AJUSTEMENT DE LA DATE
    # ==========================================
    
    today = datetime.now()
    
    # Vérifier si la date est dans le futur
    if date > today:
        print(f"⚠️  Date dans le futur ! ({date.strftime('%Y-%m-%d')})")
        date = today - timedelta(days=7)
        print(f"   → Utilisation du {date.strftime('%Y-%m-%d')} (7 jours avant)")
    else:
        days_ago = (today - date).days
        if days_ago < 3:
            print(f"⚠️  Date trop récente ({days_ago} jours)")
            date = today - timedelta(days=7)
            print(f"   → Utilisation du {date.strftime('%Y-%m-%d')} (7 jours avant)")
    
    # ==========================================
    # 2. DÉFINIR LA RÉGION
    # ==========================================
    
    point = ee.Geometry.Point([lon, lat])
    # Buffer de 32 km = zone de 64×64 km
    region = point.buffer(32000).bounds()
    
    date_str = date.strftime('%Y-%m-%d')
    date_end = (date + timedelta(days=1)).strftime('%Y-%m-%d')
    
    # ==========================================
    # 3. COLLECTER CHAQUE SOURCE SÉPARÉMENT
    # ==========================================
    
    data_dict = {}
    
    # --- ELEVATION (statique, toujours disponible) ---
    print("📍 Élévation...")
    try:
        elevation = ee.Image('USGS/SRTMGL1_003').select('elevation')
        elev_sample = elevation.sample(
            region=region,
            scale=1000,
            numPixels=4096,
            seed=42
        )
        elev_features = elev_sample.getInfo()['features']
        elev_values = [f['properties']['elevation'] for f in elev_features]
        
        # Compléter si nécessaire
        while len(elev_values) < 4096:
            elev_values.append(np.mean(elev_values) if elev_values else 500)
        
        data_dict['elevation'] = np.array(elev_values[:4096]).reshape(64, 64)
        print(f"   ✓ {len(elev_features)} points (min: {np.min(data_dict['elevation']):.0f}m, max: {np.max(data_dict['elevation']):.0f}m)")
    except Exception as e:
        print(f"   ✗ Erreur: {e}")
        data_dict['elevation'] = np.ones((64, 64)) * 500
    
    # --- GRIDMET (météo) ---
    print("🌤️  Météo GRIDMET...")
    try:
        gridmet = ee.ImageCollection('IDAHO_EPSCOR/GRIDMET') \
                    .filterDate(date_str, date_end) \
                    .filterBounds(point)
        
        # Vérifier qu'il y a des images
        count = gridmet.size().getInfo()
        print(f"   → {count} image(s) trouvée(s)")
        
        if count == 0:
            # Essayer avec une plage plus large
            date_start_wide = (date - timedelta(days=3)).strftime('%Y-%m-%d')
            gridmet = ee.ImageCollection('IDAHO_EPSCOR/GRIDMET') \
                        .filterDate(date_start_wide, date_end) \
                        .filterBounds(point) \
                        .sort('system:time_start', False)
            count = gridmet.size().getInfo()
            print(f"   → Recherche élargie: {count} image(s)")
        
        if count > 0:
            img = gridmet.first()
            
            for var in ['th', 'vs', 'tmmn', 'tmmx', 'sph']:
                try:
                    var_sample = img.select(var).sample(
                        region=region,
                        scale=1000,
                        numPixels=4096,
                        seed=42
                    )
                    var_features = var_sample.getInfo()['features']
                    var_values = [f['properties'][var] for f in var_features if var in f['properties']]
                    
                    # Valeur par défaut si vide
                    default_values = {
                        'th': 180,    # direction vent
                        'vs': 5,      # vitesse vent
                        'tmmn': 15,   # temp min
                        'tmmx': 25,   # temp max
                        'sph': 0.01   # humidité
                    }
                    
                    while len(var_values) < 4096:
                        var_values.append(default_values[var])
                    
                    data_dict[var] = np.array(var_values[:4096]).reshape(64, 64)
                    print(f"   ✓ {var}: {len(var_features)} points (moy: {np.mean(data_dict[var]):.2f})")
                    
                except Exception as e:
                    print(f"   ✗ {var}: {e}")
                    data_dict[var] = np.ones((64, 64)) * default_values[var]
        else:
            print(f"   ⚠️  Aucune donnée GRIDMET, utilisation de valeurs par défaut")
            data_dict['th'] = np.ones((64, 64)) * 180
            data_dict['vs'] = np.ones((64, 64)) * 5
            data_dict['tmmn'] = np.ones((64, 64)) * 15
            data_dict['tmmx'] = np.ones((64, 64)) * 25
            data_dict['sph'] = np.ones((64, 64)) * 0.01
            
    except Exception as e:
        print(f"   ✗ Erreur GRIDMET: {e}")
        data_dict['th'] = np.ones((64, 64)) * 180
        data_dict['vs'] = np.ones((64, 64)) * 5
        data_dict['tmmn'] = np.ones((64, 64)) * 15
        data_dict['tmmx'] = np.ones((64, 64)) * 25
        data_dict['sph'] = np.ones((64, 64)) * 0.01
    
    # --- PDSI (sécheresse) ---
    print("💧 Sécheresse PDSI...")
    try:
        # PDSI mis à jour tous les 5 jours
        date_start_pdsi = (date - timedelta(days=10)).strftime('%Y-%m-%d')
        
        drought = ee.ImageCollection('GRIDMET/DROUGHT') \
                    .filterDate(date_start_pdsi, date_end) \
                    .filterBounds(point) \
                    .sort('system:time_start', False)
        
        count = drought.size().getInfo()
        print(f"   → {count} image(s) trouvée(s)")
        
        if count > 0:
            img = drought.first()
            pdsi_sample = img.select('pdsi').sample(
                region=region,
                scale=1000,
                numPixels=4096,
                seed=42
            )
            pdsi_features = pdsi_sample.getInfo()['features']
            pdsi_values = [f['properties']['pdsi'] for f in pdsi_features if 'pdsi' in f['properties']]
            
            while len(pdsi_values) < 4096:
                pdsi_values.append(0)
            
            data_dict['pdsi'] = np.array(pdsi_values[:4096]).reshape(64, 64)
            print(f"   ✓ {len(pdsi_features)} points (moy: {np.mean(data_dict['pdsi']):.2f})")
        else:
            data_dict['pdsi'] = np.zeros((64, 64))
            print(f"   ⚠️  Pas de données PDSI récentes")
            
    except Exception as e:
        print(f"   ✗ Erreur: {e}")
        data_dict['pdsi'] = np.zeros((64, 64))
    
    # --- NDVI (végétation) ---
    print("🌿 Végétation NDVI...")
    try:
        # NDVI: délai de traitement ~15-20 jours, chercher sur 60 jours
        date_start_ndvi = (date - timedelta(days=60)).strftime('%Y-%m-%d')
        
        print(f"   → Recherche VIIRS du {date_start_ndvi} au {date_end}")
        viirs = ee.ImageCollection('NOAA/VIIRS/001/VNP13A1') \
                  .filterDate(date_start_ndvi, date_end) \
                  .filterBounds(point) \
                  .sort('system:time_start', False)
        
        count = viirs.size().getInfo()
        print(f"   → {count} image(s) VIIRS trouvée(s)")
        
        if count > 0:
            img = viirs.first()
            
            # Obtenir la date de l'image
            img_date = datetime.fromtimestamp(img.get('system:time_start').getInfo() / 1000)
            days_old = (date - img_date).days
            print(f"   → Image du {img_date.strftime('%Y-%m-%d')} ({days_old} jours avant)")
            
            ndvi_sample = img.select('NDVI').sample(
                region=region,
                scale=500,  # VIIRS à 500m
                numPixels=4096,
                seed=42
            )
            ndvi_features = ndvi_sample.getInfo()['features']
            # NDVI est en scale 0-10000, normaliser à 0-1
            ndvi_values = [f['properties']['NDVI'] / 10000.0 for f in ndvi_features if 'NDVI' in f['properties']]
            
            while len(ndvi_values) < 4096:
                ndvi_values.append(0.5)
            
            data_dict['NDVI'] = np.array(ndvi_values[:4096]).reshape(64, 64)
            print(f"   ✓ {len(ndvi_features)} points (min: {np.min(data_dict['NDVI']):.3f}, moy: {np.mean(data_dict['NDVI']):.3f}, max: {np.max(data_dict['NDVI']):.3f})")
        else:
            # Essayer MODIS NDVI (plus disponible, 250m)
            print(f"   → Tentative avec MODIS...")
            modis = ee.ImageCollection('MODIS/061/MOD13Q1') \
                      .filterDate(date_start_ndvi, date_end) \
                      .filterBounds(point) \
                      .sort('system:time_start', False)
            
            count_modis = modis.size().getInfo()
            print(f"   → {count_modis} image(s) MODIS trouvée(s)")
            
            if count_modis > 0:
                img = modis.first()
                img_date = datetime.fromtimestamp(img.get('system:time_start').getInfo() / 1000)
                days_old = (date - img_date).days
                print(f"   → Image du {img_date.strftime('%Y-%m-%d')} ({days_old} jours avant)")
                
                ndvi_sample = img.select('NDVI').sample(
                    region=region,
                    scale=250,
                    numPixels=4096,
                    seed=42
                )
                ndvi_features = ndvi_sample.getInfo()['features']
                # MODIS NDVI aussi en scale 0-10000
                ndvi_values = [f['properties']['NDVI'] / 10000.0 for f in ndvi_features if 'NDVI' in f['properties']]
                
                while len(ndvi_values) < 4096:
                    ndvi_values.append(0.5)
                
                data_dict['NDVI'] = np.array(ndvi_values[:4096]).reshape(64, 64)
                print(f"   ✓ {len(ndvi_features)} points MODIS (min: {np.min(data_dict['NDVI']):.3f}, moy: {np.mean(data_dict['NDVI']):.3f}, max: {np.max(data_dict['NDVI']):.3f})")
            else:
                data_dict['NDVI'] = np.ones((64, 64)) * 0.5
                print(f"   ⚠️  Pas de données NDVI récentes (ni VIIRS ni MODIS)")
            
    except Exception as e:
        print(f"   ✗ Erreur: {e}")
        data_dict['NDVI'] = np.ones((64, 64)) * 0.5
    
    # --- PrevFireMask (à implémenter avec FIRMS) ---
    data_dict['PrevFireMask'] = np.zeros((64, 64))
    
    # ==========================================
    # 4. RÉSUMÉ
    # ==========================================
    
    print(f"\n✅ Collecte terminée:")
    print(f"   Variables: {list(data_dict.keys())}")
    print(f"   Forme: {data_dict['elevation'].shape}")
    print(f"   Qualité des données:")
    
    # Évaluer la qualité
    quality_score = 0
    if np.std(data_dict['elevation']) > 10:
        quality_score += 1
    if np.std(data_dict['vs']) > 0.1:
        quality_score += 1
    if np.std(data_dict['NDVI']) > 0.01:
        quality_score += 1
    
    quality = ["Faible", "Moyenne", "Bonne", "Excellente"][min(quality_score, 3)]
    print(f"   → {quality}")
    
    return data_dict

def predict_direction(lat: float, lng: float, date: datetime) -> float:
    # Collecter des données
    data = collect_gee_data(lat, lng, date)

    # Extraire features
    features = extract_features_from_dict(data)
    features = np.array(features).reshape(1, -1)

    # Prédire
    prediction = direction.predict(features)[0]
    #print(f"Direction prédite : {prediction:.1f}°")

    return prediction