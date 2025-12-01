"""
Script pour utiliser les données collectées avec ton modèle de prédiction
"""

import numpy as np
#import tensorflow as tf

# ============================================
# 1. CHARGER LES DONNÉES COLLECTÉES
# ============================================

def load_collected_data(filepath="backend/data/processed/collected_data.npy"):
    """Charge les données depuis le fichier .npy"""
    data = np.load(filepath, allow_pickle=True).item()
    return data

# ============================================
# 2. EXTRAIRE LES FEATURES (comme dans process_data.py)
# ============================================

def extract_features_from_collected_data(data):
    """
    Extrait les features dans le même format que process_data.py attend
    
    Ton process_data.py calcule:
    - Pour chaque variable: [mean, max]
    - Vecteur de vent: [wind_east, wind_north]
    - Interactions: [elevation*wind_speed, drought*vegetation]
    - Fire shape ratio
    """
    
    features = []
    
    # 1. Environmental stats (mean, max pour chaque variable)
    # Ordre: elevation, th, vs, tmmn, tmmx, sph, pdsi, NDVI
    for var_name in ['elevation', 'th', 'vs', 'tmmn', 'tmmx', 'sph', 'pdsi', 'NDVI']:
        arr = data[var_name]
        features.extend([np.mean(arr), np.max(arr)])
    
    # 2. Fire mask stats (sum, mean de PrevFireMask)
    prev_fire = data['PrevFireMask']
    features.extend([np.sum(prev_fire), np.mean(prev_fire)])
    
    # 3. Wind vector components
    wind_dir_mean = np.mean(data['th'])
    wind_speed_mean = np.mean(data['vs'])
    wind_east = wind_speed_mean * np.cos(np.radians(wind_dir_mean))
    wind_north = wind_speed_mean * np.sin(np.radians(wind_dir_mean))
    features.append(wind_east)
    features.append(wind_north)
    
    # 4. Interactions
    elevation_mean = np.mean(data['elevation'])
    drought_mean = np.mean(data['pdsi'])
    vegetation_mean = np.mean(data['NDVI'])
    
    features.append(elevation_mean * wind_speed_mean)
    features.append(drought_mean * vegetation_mean)
    
    # 5. Fire shape ratio
    try:
        fire_mask_2d = prev_fire.reshape(64, 64)
        if np.sum(fire_mask_2d) > 0:
            y_indices, x_indices = np.where(fire_mask_2d > 0)
            fire_width = np.max(x_indices) - np.min(x_indices)
            fire_height = np.max(y_indices) - np.min(y_indices)
            shape_ratio = fire_width / (fire_height + 1e-6)
        else:
            shape_ratio = 1.0
    except Exception:
        shape_ratio = 1.0
    
    features.append(shape_ratio)
    
    return np.array(features)

