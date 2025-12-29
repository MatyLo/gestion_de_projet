import pandas as pd
import requests
from io import StringIO
from datetime import datetime
import numpy as np
from datetime import timedelta
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import tensorflow as tf
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from backend.recup_data import collect_all_data
from backend.process_feature import extract_features_from_collected_data

"""
Google Earth Engine Data Collector - Version robuste
Corrige tous les problèmes de dates et données null
"""

import ee
import numpy as np
from datetime import datetime, timedelta

# Initialiser GEE
ee.Initialize(project='airy-galaxy-471607-b6')


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

"""
Training Pipeline for Wildfire Direction Prediction Model
Loads Google's Next Day Wildfire Spread dataset and trains a direction predictor
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Configuration
TFRECORD_DIR = "/Users/matylo/Documents/projet_incendie2/backend/data_nextfire"  # À MODIFIER
MODEL_SAVE_PATH = "backend/ml/wildfire_direction_predictor.joblib"
GRID_SIZE = 64


# ==========================================
# 📂 PARTIE 1 : CHARGEMENT DU DATASET
# ==========================================

def parse_tfrecord(example_proto):
    """Parse un exemple TFRecord du dataset Google"""
    
    INPUT_FEATURES = ['elevation', 'th', 'vs', 'tmmn', 'tmmx', 'sph', 'pdsi', 'NDVI']
    
    feature_description = {}
    
    # Features d'entrée
    for feature_name in INPUT_FEATURES:
        feature_description[feature_name] = tf.io.FixedLenFeature(
            [GRID_SIZE * GRID_SIZE], tf.float32
        )
    
    # Masques de feu
    feature_description['PrevFireMask'] = tf.io.FixedLenFeature(
        [GRID_SIZE * GRID_SIZE], tf.float32
    )
    feature_description['FireMask'] = tf.io.FixedLenFeature(
        [GRID_SIZE * GRID_SIZE], tf.float32
    )
    
    # Parse
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    
    # Reshape en 64×64
    features = {}
    for key in parsed.keys():
        features[key] = tf.reshape(parsed[key], [GRID_SIZE, GRID_SIZE])
    
    return features



def load_tfrecord_dataset(tfrecord_dir, batch_size=1):
    """Charge les fichiers TFRecord avec détection automatique de compression"""
    
    tfrecord_files = list(Path(tfrecord_dir).glob("*.tfrecord*"))
    tfrecord_files = [str(f) for f in tfrecord_files]
    
    print(f"📁 {len(tfrecord_files)} fichiers TFRecord trouvés")
    
    if len(tfrecord_files) == 0:
        raise FileNotFoundError(f"Aucun fichier TFRecord dans {tfrecord_dir}")
    
    # Afficher les fichiers
    print(f"\n📄 Fichiers détectés:")
    for i, f in enumerate(tfrecord_files[:3], 1):
        print(f"   {i}. {Path(f).name}")
    if len(tfrecord_files) > 3:
        print(f"   ... et {len(tfrecord_files) - 3} autres")
    
    # Tester différents types de compression
    print(f"\n🔍 Détection du type de compression...")
    
    compression_types = [
        ('GZIP', 'GZIP'),
        ('ZLIB', 'ZLIB'),
        ('None', ''),
        ('Auto', None)
    ]
    
    for comp_name, comp_type in compression_types:
        try:
            print(f"   Tentative avec compression={comp_name}...")
            test_dataset = tf.data.TFRecordDataset(
                tfrecord_files[:1], 
                compression_type=comp_type
            )
            test_dataset = test_dataset.map(parse_tfrecord)
            
            # Essayer de lire un exemple
            for example in test_dataset.take(1):
                print(f"   ✓ {comp_name} fonctionne!")
                compression_found = comp_type
                break
            
            # Si ça marche, utiliser ce type
            dataset = tf.data.TFRecordDataset(tfrecord_files, compression_type=comp_type)
            dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            
            print(f"\n✅ Utilisation de compression: {comp_name}")
            return dataset
            
        except Exception as e:
            print(f"   ✗ {comp_name} échoue: {str(e)[:60]}")
            continue
    
    # Si aucune compression ne fonctionne
    raise ValueError("Impossible de lire les fichiers TFRecord avec aucun type de compression")


# ==========================================
# 🎯 PARTIE 2 : EXTRACTION DES FEATURES
# ==========================================

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

# ==========================================
# 📐 PARTIE 3 : CALCUL DE LA DIRECTION
# ==========================================

def compute_fire_direction(prev_fire_mask, next_fire_mask):
    """
    Calcule la direction de propagation entre deux masques.
    Retourne un angle en degrés (0-360) où 0=Nord, 90=Est, 180=Sud, 270=Ouest
    """
    
    prev_fire = prev_fire_mask.numpy() if hasattr(prev_fire_mask, 'numpy') else prev_fire_mask
    next_fire = next_fire_mask.numpy() if hasattr(next_fire_mask, 'numpy') else next_fire_mask
    
    # Nouveaux pixels brûlés
    new_fire = (next_fire > 0) & (prev_fire == 0)
    
    # Vérifications
    if np.sum(new_fire) == 0:
        return None  # Pas de nouvelle propagation
    
    if np.sum(prev_fire > 0) == 0:
        return None  # Pas de feu précédent
    
    # Centroïde du feu précédent
    prev_y, prev_x = np.where(prev_fire > 0)
    prev_center_y = np.mean(prev_y)
    prev_center_x = np.mean(prev_x)
    
    # Centroïde de la nouvelle zone brûlée
    new_y, new_x = np.where(new_fire)
    new_center_y = np.mean(new_y)
    new_center_x = np.mean(new_x)
    
    # Calcul de l'angle
    dy = new_center_y - prev_center_y
    dx = new_center_x - prev_center_x
    
    # Convertir en angle géographique (0=Nord, sens horaire)
    angle = np.degrees(np.arctan2(dx, -dy)) % 360
    
    return angle


def calculate_circular_mean(angles):
    """Moyenne circulaire pour les angles"""
    rad = np.radians(angles)
    sin_mean = np.mean(np.sin(rad))
    cos_mean = np.mean(np.cos(rad))
    mean_angle = np.degrees(np.arctan2(sin_mean, cos_mean)) % 360
    return mean_angle


def calculate_circular_std(angles):
    """Écart-type circulaire pour les angles"""
    rad = np.radians(angles)
    sin_mean = np.mean(np.sin(rad))
    cos_mean = np.mean(np.cos(rad))
    r = np.sqrt(sin_mean**2 + cos_mean**2)
    
    if r < 0.001:  # Données trop dispersées
        return 180.0
    
    circular_std = np.degrees(np.sqrt(-2 * np.log(r)))
    return circular_std


def angular_error(y_true, y_pred):
    """Erreur angulaire (prend en compte que 359° et 1° sont proches)"""
    diff = np.abs(y_true - y_pred)
    return np.minimum(diff, 360 - diff)


"""
Advanced Circular Regression Model for Fire Direction Prediction
Traite correctement les angles circulaires (0° = 360°)
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
import joblib


class CircularRegressor(BaseEstimator, RegressorMixin):
    """
    Régression pour angles circulaires.
    
    Au lieu de prédire directement l'angle (0-360°), prédit:
    - sin(angle) et cos(angle)
    
    Avantages:
    - Pas de discontinuité à 0°/360°
    - Meilleure gestion de la circularité
    """
    
    def __init__(self, base_estimator=None, n_estimators=300, max_depth=25):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model_sin = None
        self.model_cos = None
    
    def fit(self, X, y):
        """
        Entraîne deux modèles: un pour sin(angle), un pour cos(angle)
        
        Args:
            X: Features (n_samples, n_features)
            y: Angles en degrés (n_samples,)
        """
        # Convertir angles en composantes sin/cos
        y_rad = np.radians(y)
        y_sin = np.sin(y_rad)
        y_cos = np.cos(y_rad)
        
        # Créer les modèles de base
        if self.base_estimator is None:
            self.model_sin = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=3,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1
            )
            self.model_cos = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=3,
                min_samples_leaf=1,
                random_state=43,  # Seed différent
                n_jobs=-1
            )
        else:
            from copy import deepcopy
            self.model_sin = deepcopy(self.base_estimator)
            self.model_cos = deepcopy(self.base_estimator)
        
        print("   Entraînement modèle sin...")
        self.model_sin.fit(X, y_sin)
        
        print("   Entraînement modèle cos...")
        self.model_cos.fit(X, y_cos)
        
        return self
    
    def predict(self, X):
        """
        Prédit les angles
        
        Args:
            X: Features (n_samples, n_features)
        
        Returns:
            Angles en degrés (n_samples,)
        """
        # Prédire sin et cos
        y_sin_pred = self.model_sin.predict(X)
        y_cos_pred = self.model_cos.predict(X)
        
        # Normaliser (sin²+cos²=1)
        magnitude = np.sqrt(y_sin_pred**2 + y_cos_pred**2)
        y_sin_pred = y_sin_pred / (magnitude + 1e-8)
        y_cos_pred = y_cos_pred / (magnitude + 1e-8)
        
        # Convertir en angle
        angles_rad = np.arctan2(y_sin_pred, y_cos_pred)
        angles_deg = np.degrees(angles_rad) % 360
        
        return angles_deg
    
    def get_confidence(self, X):
        """
        Estime la confiance de la prédiction basée sur la magnitude
        
        Returns:
            Confiance (0-1) pour chaque prédiction
        """
        y_sin_pred = self.model_sin.predict(X)
        y_cos_pred = self.model_cos.predict(X)
        
        # Magnitude proche de 1 = haute confiance
        magnitude = np.sqrt(y_sin_pred**2 + y_cos_pred**2)
        
        return magnitude


class WeightedCircularRegressor(BaseEstimator, RegressorMixin):
    """
    Combine régression directe + régression circulaire avec pondération
    """
    
    def __init__(self, alpha=0.6):
        """
        Args:
            alpha: Poids de la régression circulaire (0-1)
                   1.0 = uniquement circulaire
                   0.0 = uniquement directe
        """
        self.alpha = alpha
        self.circular_model = None
        self.direct_model = None
    
    def fit(self, X, y):
        print(f"   Alpha (circulaire): {self.alpha:.2f}")
        
        # Modèle circulaire
        self.circular_model = CircularRegressor(n_estimators=200, max_depth=25)
        self.circular_model.fit(X, y)
        
        # Modèle direct
        self.direct_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=25,
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )
        print("   Entraînement modèle direct...")
        self.direct_model.fit(X, y)
        
        return self
    
    def predict(self, X):
        # Prédictions des deux modèles
        pred_circular = self.circular_model.predict(X)
        pred_direct = self.direct_model.predict(X)
        
        # Moyenne circulaire pondérée
        pred_circ_rad = np.radians(pred_circular)
        pred_direct_rad = np.radians(pred_direct)
        
        # Convertir en vecteurs
        sin_circ = np.sin(pred_circ_rad) * self.alpha
        cos_circ = np.cos(pred_circ_rad) * self.alpha
        
        sin_direct = np.sin(pred_direct_rad) * (1 - self.alpha)
        cos_direct = np.cos(pred_direct_rad) * (1 - self.alpha)
        
        # Combiner
        sin_combined = sin_circ + sin_direct
        cos_combined = cos_circ + cos_direct
        
        # Convertir en angle
        angles_rad = np.arctan2(sin_combined, cos_combined)
        angles_deg = np.degrees(angles_rad) % 360
        
        return angles_deg


def train_circular_models(X_train, y_train, X_val, y_val):
    """
    Entraîne et évalue les modèles circulaires
    
    Returns:
        dict avec les résultats de chaque modèle
    """
    print("\n" + "="*60)
    print("🔄 MODÈLES CIRCULAIRES SPÉCIALISÉS")
    print("="*60)
    
    from sklearn.metrics import mean_absolute_error
    
    def angular_error(y_true, y_pred):
        diff = np.abs(y_true - y_pred)
        return np.minimum(diff, 360 - diff)
    
    results = {}
    
    # 1. Régression circulaire pure
    print("\n🎯 CircularRegressor (sin/cos)...")
    circular_model = CircularRegressor(n_estimators=300, max_depth=30)
    circular_model.fit(X_train, y_train)
    
    y_pred = circular_model.predict(X_val)
    errors = angular_error(y_val, y_pred)
    
    results['Circular'] = {
        'model': circular_model,
        'mean_angular_error': np.mean(errors),
        'median_angular_error': np.median(errors),
        'pct_within_30': np.mean(errors <= 30) * 100,
        'pct_within_45': np.mean(errors <= 45) * 100,
        'pct_within_90': np.mean(errors <= 90) * 100
    }
    
    print(f"   Erreur angulaire moyenne: {results['Circular']['mean_angular_error']:.2f}°")
    print(f"   Dans ±30°: {results['Circular']['pct_within_30']:.1f}%")
    print(f"   Dans ±45°: {results['Circular']['pct_within_45']:.1f}%")
    
    # 2. Régression pondérée (tester différents alphas)
    best_alpha = 0.6
    best_error = float('inf')
    
    print("\n🔀 WeightedCircularRegressor (optimisation alpha)...")
    for alpha in [0.3, 0.5, 0.6, 0.7, 0.8]:
        print(f"\n   Test alpha={alpha}...")
        weighted_model = WeightedCircularRegressor(alpha=alpha)
        weighted_model.fit(X_train, y_train)
        
        y_pred = weighted_model.predict(X_val)
        errors = angular_error(y_val, y_pred)
        mean_error = np.mean(errors)
        
        print(f"   Erreur: {mean_error:.2f}°")
        
        if mean_error < best_error:
            best_error = mean_error
            best_alpha = alpha
            best_weighted_model = weighted_model
            
            results['Weighted'] = {
                'model': best_weighted_model,
                'mean_angular_error': mean_error,
                'median_angular_error': np.median(errors),
                'pct_within_30': np.mean(errors <= 30) * 100,
                'pct_within_45': np.mean(errors <= 45) * 100,
                'pct_within_90': np.mean(errors <= 90) * 100,
                'best_alpha': best_alpha
            }
    
    print(f"\n   Meilleur alpha: {best_alpha}")
    print(f"   Erreur angulaire moyenne: {results['Weighted']['mean_angular_error']:.2f}°")
    print(f"   Dans ±45°: {results['Weighted']['pct_within_45']:.1f}%")
    
    return results


# ==========================================
# 🚀 INTÉGRATION AU PIPELINE PRINCIPAL
# ==========================================

def add_circular_models_to_training(X_train, y_train, X_val, y_val, existing_results):
    """
    Ajoute les modèles circulaires aux résultats existants
    
    Args:
        X_train, y_train: Données d'entraînement
        X_val, y_val: Données de validation
        existing_results: Dict avec résultats des modèles standards
    
    Returns:
        Dict combiné avec tous les modèles
    """
    
    # Entraîner les modèles circulaires
    circular_results = train_circular_models(X_train, y_train, X_val, y_val)
    
    # Combiner avec les résultats existants
    all_results = {**existing_results, **circular_results}
    
    # Afficher le comparatif
    print("\n" + "="*60)
    print("📊 COMPARATIF TOUS MODÈLES")
    print("="*60)
    
    sorted_models = sorted(all_results.items(), 
                          key=lambda x: x[1]['mean_angular_error'])
    
    for i, (name, metrics) in enumerate(sorted_models, 1):
        print(f"\n{i}. {name}")
        print(f"   Erreur: {metrics['mean_angular_error']:.2f}°")
        print(f"   ±30°: {metrics['pct_within_30']:.1f}%")
        print(f"   ±45°: {metrics['pct_within_45']:.1f}%")
        print(f"   ±90°: {metrics['pct_within_90']:.1f}%")
    
    return all_results

# ==========================================
# 🤖 PARTIE 4 : ENTRAÎNEMENT DU MODÈLE
# ==========================================

def prepare_training_data(tfrecord_dir, max_samples=None):
    """
    Charge et prépare les données d'entraînement.
    
    Args:
        tfrecord_dir: Chemin vers les fichiers TFRecord
        max_samples: Limite de samples (None = tous)
    
    Returns:
        X, y, metadata
    """
    
    print("=" * 60)
    print("📊 PRÉPARATION DES DONNÉES D'ENTRAÎNEMENT")
    print("=" * 60)
    
    dataset = load_tfrecord_dataset(tfrecord_dir, batch_size=1)
    
    X_list = []
    y_list = []
    metadata_list = []
    
    count_total = 0
    count_valid = 0
    count_no_spread = 0
    
    print("\n🔄 Traitement des échantillons...")
    
    for batch in dataset:
        if max_samples and count_total >= max_samples:
            break
        
        count_total += 1
        
        # Extraire le premier élément du batch
        features_dict = {k: v[0] for k, v in batch.items()}
        
        # Extraire les features ML
        features = extract_features_from_dict(features_dict)
        
        # Calculer la direction (label)
        direction = compute_fire_direction(
            features_dict['PrevFireMask'],
            features_dict['FireMask']
        )
        
        if direction is not None:
            X_list.append(features)
            y_list.append(direction)
            
            # Métadonnées pour analyse
            metadata_list.append({
                'wind_direction': np.mean(features_dict['th'].numpy()),
                'wind_speed': np.mean(features_dict['vs'].numpy()),
                'elevation': np.mean(features_dict['elevation'].numpy()),
                'ndvi': np.mean(features_dict['NDVI'].numpy())
            })
            
            count_valid += 1
        else:
            count_no_spread += 1
        
        # Progression
        if count_total % 1000 == 0:
            print(f"  Traité: {count_total:,} | Valides: {count_valid:,} | Sans propagation: {count_no_spread:,}")
    
    X = np.array(X_list)
    y = np.array(y_list)
    metadata = pd.DataFrame(metadata_list)
    
    print(f"\n✅ Données préparées:")
    print(f"   Total échantillons: {count_total:,}")
    print(f"   Avec propagation: {count_valid:,} ({count_valid/count_total*100:.1f}%)")
    print(f"   Sans propagation: {count_no_spread:,}")
    print(f"   Forme X: {X.shape}")
    print(f"   Forme y: {y.shape}")
    
    np.save("backend/ml/X.npy", X)
    np.save("backend/ml/y.npy", y)
    np.save("backend/ml/metadata.npy", metadata)
    return X, y, metadata


def train_direction_model2(X, y, metadata):
    """
    Entraîne le modèle de prédiction de direction.
    
    Args:
        X: Features (n_samples, n_features)
        y: Directions (n_samples,) en degrés 0-360
        metadata: DataFrame avec infos contextuelles
    
    Returns:
        model, metrics
    """
    
    print("\n" + "=" * 60)
    print("🤖 ENTRAÎNEMENT DU MODÈLE")
    print("=" * 60)
    
    # Split train/validation/test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42  # 0.176 * 0.85 ≈ 0.15
    )
    
    # Vérifier le % de NaN dans vos features
    nan_pct = np.isnan(X_train).sum(axis=0) / len(X_train) * 100
    print("% NaN par feature:", nan_pct)

    print(f"\n📊 Splits:")
    print(f"   Train: {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   Val:   {len(X_val):,} ({len(X_val)/len(X)*100:.1f}%)")
    print(f"   Test:  {len(X_test):,} ({len(X_test)/len(X)*100:.1f}%)")
    
    # Distribution des directions
    print(f"\n📐 Distribution des directions (train):")
    print(f"   Moyenne circulaire: {calculate_circular_mean(y_train):.1f}°")
    print(f"   Std circulaire: {calculate_circular_std(y_train):.1f}°")
    print(f"   Min: {np.min(y_train):.1f}°, Max: {np.max(y_train):.1f}°")
    
    # Entraîner plusieurs modèles et choisir le meilleur
    models = {
        'RandomForest': RandomForestRegressor(
            n_estimators=200,
            max_depth=25,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            verbose=1
        ),
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.1,
            random_state=42,
            verbose=1
        )
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n🔧 Entraînement {name}...")
        model.fit(X_train, y_train)
        
        # Prédictions
        y_pred_val = model.predict(X_val)
        
        # Métriques
        mae = mean_absolute_error(y_val, y_pred_val)
        angular_errors = angular_error(y_val, y_pred_val)
        mean_angular_error = np.mean(angular_errors)
        median_angular_error = np.median(angular_errors)
        
        # Pourcentage dans X degrés
        pct_within_30 = np.mean(angular_errors <= 30) * 100
        pct_within_45 = np.mean(angular_errors <= 45) * 100
        pct_within_90 = np.mean(angular_errors <= 90) * 100
        
        results[name] = {
            'model': model,
            'mae': mae,
            'mean_angular_error': mean_angular_error,
            'median_angular_error': median_angular_error,
            'pct_within_30': pct_within_30,
            'pct_within_45': pct_within_45,
            'pct_within_90': pct_within_90
        }
        
        print(f"   MAE: {mae:.2f}°")
        print(f"   Erreur angulaire moyenne: {mean_angular_error:.2f}°")
        print(f"   Erreur angulaire médiane: {median_angular_error:.2f}°")
        print(f"   Dans ±30°: {pct_within_30:.1f}%")
        print(f"   Dans ±45°: {pct_within_45:.1f}%")
        print(f"   Dans ±90°: {pct_within_90:.1f}%")
    
    # Choisir le meilleur modèle (basé sur erreur angulaire moyenne)
    best_name = min(results, key=lambda x: results[x]['mean_angular_error'])
    best_model = results[best_name]['model']
    best_metrics = results[best_name]
    
    print(f"\n🏆 Meilleur modèle: {best_name}")
    print(f"   Erreur angulaire moyenne: {best_metrics['mean_angular_error']:.2f}°")
    
    # Évaluation finale sur test set
    print(f"\n📊 ÉVALUATION FINALE (Test Set):")
    y_pred_test = best_model.predict(X_test)
    
    test_angular_errors = angular_error(y_test, y_pred_test)
    test_mean_error = np.mean(test_angular_errors)
    test_median_error = np.median(test_angular_errors)
    test_pct_30 = np.mean(test_angular_errors <= 30) * 100
    test_pct_45 = np.mean(test_angular_errors <= 45) * 100
    test_pct_90 = np.mean(test_angular_errors <= 90) * 100
    
    print(f"   Erreur angulaire moyenne: {test_mean_error:.2f}°")
    print(f"   Erreur angulaire médiane: {test_median_error:.2f}°")
    print(f"   Dans ±30°: {test_pct_30:.1f}%")
    print(f"   Dans ±45°: {test_pct_45:.1f}%")
    print(f"   Dans ±90°: {test_pct_90:.1f}%")
    
    # Feature importance (si Random Forest)
    if best_name == 'RandomForest':
        feature_names = [
            'elev_mean', 'elev_max',
            'wind_dir_mean', 'wind_dir_max',
            'wind_speed_mean', 'wind_speed_max',
            'temp_min_mean', 'temp_min_max',
            'temp_max_mean', 'temp_max_max',
            'humidity_mean', 'humidity_max',
            'drought_mean', 'drought_max',
            'ndvi_mean', 'ndvi_max',
            'fire_sum', 'fire_mean',
            'wind_east', 'wind_north',
            'elev_wind_interaction', 'drought_ndvi_interaction',
            'fire_shape_ratio'
        ]
        
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        
        print(f"\n🔝 Top 10 features importantes:")
        for i, idx in enumerate(indices, 1):
            print(f"   {i}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    final_metrics = {
        'model_name': best_name,
        'test_mean_angular_error': test_mean_error,
        'test_median_angular_error': test_median_error,
        'test_pct_within_30': test_pct_30,
        'test_pct_within_45': test_pct_45,
        'test_pct_within_90': test_pct_90,
        'n_train': len(X_train),
        'n_test': len(X_test)
    }
    
    return best_model, final_metrics


def train_direction_model(X, y, metadata):
    """
    Entraîne le modèle de prédiction de direction.
    
    Args:
        X: Features (n_samples, n_features)
        y: Directions (n_samples,) en degrés 0-360
        metadata: DataFrame avec infos contextuelles
    
    Returns:
        model, metrics
    """
    
    print("\n" + "=" * 60)
    print("🤖 ENTRAÎNEMENT DU MODÈLE")
    print("=" * 60)
    
    # Split train/validation/test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42  # 0.176 * 0.85 ≈ 0.15
    )
    
    print(f"\n📊 Splits:")
    print(f"   Train: {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   Val:   {len(X_val):,} ({len(X_val)/len(X)*100:.1f}%)")
    print(f"   Test:  {len(X_test):,} ({len(X_test)/len(X)*100:.1f}%)")
    
    # Distribution des directions
    print(f"\n📐 Distribution des directions (train):")
    print(f"   Moyenne circulaire: {calculate_circular_mean(y_train):.1f}°")
    print(f"   Std circulaire: {calculate_circular_std(y_train):.1f}°")
    print(f"   Min: {np.min(y_train):.1f}°, Max: {np.max(y_train):.1f}°")
    
    # Entraîner plusieurs modèles et choisir le meilleur
    models = {
        'RandomForest': RandomForestRegressor(
            n_estimators=300,  # Augmenté de 200 à 300
            max_depth=30,      # Augmenté de 25 à 30
            min_samples_split=3,  # Réduit de 5 à 3
            min_samples_leaf=1,   # Réduit de 2 à 1
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            verbose=1
        ),
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=300,  # Augmenté de 200 à 300
            max_depth=12,      # Augmenté de 10 à 12
            learning_rate=0.05,  # Réduit de 0.1 à 0.05
            subsample=0.8,
            random_state=42,
            verbose=1
        ),
        'XGBoost': None,  # Sera initialisé si disponible
        'LightGBM': None,  # Sera initialisé si disponible
        'ExtraTrees': None,  # Sera initialisé si disponible
        'HistGradientBoosting': None,  # Sera initialisé si disponible
    }
    
    # Essayer d'importer et d'initialiser les modèles avancés
    try:
        import xgboost as xgb
        models['XGBoost'] = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=12,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=1
        )
        print("✓ XGBoost disponible")
    except ImportError:
        print("⚠️  XGBoost non disponible (pip install xgboost)")
        del models['XGBoost']
    
    try:
        import lightgbm as lgb
        models['LightGBM'] = lgb.LGBMRegressor(
            n_estimators=300,
            max_depth=12,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        print("✓ LightGBM disponible")
    except ImportError:
        print("⚠️  LightGBM non disponible (pip install lightgbm)")
        del models['LightGBM']
    
    try:
        from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
        
        models['ExtraTrees'] = ExtraTreesRegressor(
            n_estimators=300,
            max_depth=30,
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        models['HistGradientBoosting'] = HistGradientBoostingRegressor(
            max_iter=300,
            max_depth=12,
            learning_rate=0.05,
            random_state=42,
            verbose=1
        )
        print("✓ ExtraTrees et HistGradientBoosting disponibles")
    except ImportError:
        print("⚠️  ExtraTrees/HistGradientBoosting non disponibles")
        if 'ExtraTrees' in models:
            del models['ExtraTrees']
        if 'HistGradientBoosting' in models:
            del models['HistGradientBoosting']
    
    results = {}
    
    for name, model in models.items():
        print(f"\n🔧 Entraînement {name}...")
        model.fit(X_train, y_train)
        
        # Prédictions
        y_pred_val = model.predict(X_val)
        
        # Métriques
        mae = mean_absolute_error(y_val, y_pred_val)
        angular_errors = angular_error(y_val, y_pred_val)
        mean_angular_error = np.mean(angular_errors)
        median_angular_error = np.median(angular_errors)
        
        # Pourcentage dans X degrés
        pct_within_30 = np.mean(angular_errors <= 30) * 100
        pct_within_45 = np.mean(angular_errors <= 45) * 100
        pct_within_90 = np.mean(angular_errors <= 90) * 100
        
        results[name] = {
            'model': model,
            'mae': mae,
            'mean_angular_error': mean_angular_error,
            'median_angular_error': median_angular_error,
            'pct_within_30': pct_within_30,
            'pct_within_45': pct_within_45,
            'pct_within_90': pct_within_90
        }
        
        print(f"   MAE: {mae:.2f}°")
        print(f"   Erreur angulaire moyenne: {mean_angular_error:.2f}°")
        print(f"   Erreur angulaire médiane: {median_angular_error:.2f}°")
        print(f"   Dans ±30°: {pct_within_30:.1f}%")
        print(f"   Dans ±45°: {pct_within_45:.1f}%")
        print(f"   Dans ±90°: {pct_within_90:.1f}%")
    
    # Choisir le meilleur modèle (basé sur erreur angulaire moyenne)
    best_name_stand = min(results, key=lambda x: results[x]['mean_angular_error'])
    best_model_stand = results[best_name_stand]['model']
    best_metrics_stand = results[best_name_stand]
    
    print(f"\n🏆 Meilleur modèle standard: {best_name_stand}")
    print(f"   Erreur angulaire moyenne: {best_metrics_stand['mean_angular_error']:.2f}°")

    # ==========================================
    # 🔀 MODÈLES CIRCULAIRES (NOUVEAU)
    # ==========================================
    
    try:
        print("\n" + "="*60)
        print("🔄 TEST DES MODÈLES CIRCULAIRES")
        print("="*60)
        
        
        # Ajouter les modèles circulaires aux résultats
        all_results = add_circular_models_to_training(
            X_train, y_train, X_val, y_val, results
        )
        
        # Rechoisir le meilleur parmi TOUS les modèles (standards + circulaires)
        best_name = min(all_results, key=lambda x: all_results[x]['mean_angular_error'])
        best_model = all_results[best_name]['model']
        best_metrics = all_results[best_name]
        
        print(f"\n🏆 MEILLEUR MODÈLE GLOBAL: {best_name}")
        print(f"   Erreur angulaire moyenne: {best_metrics['mean_angular_error']:.2f}°")
       
    except Exception as e:
        print(f"\n⚠️  Impossible de charger les modèles circulaires: {e}")
        print("   Utilisation du meilleur modèle standard")
        # On garde best_name, best_model, best_metrics des modèles standards
    
    
    # ==========================================
    # 🔀 ENSEMBLE (STACKING) - Optionnel
    # ==========================================
    
    print(f"\n🔀 Création d'un modèle d'ensemble...")
    
    # Sélectionner les 3 meilleurs modèles
    top_3_names = sorted(results, key=lambda x: results[x]['mean_angular_error'])[:3]
    print(f"   Top 3 modèles: {', '.join(top_3_names)}")
    
    if len(top_3_names) >= 2:
        try:
            from sklearn.ensemble import StackingRegressor
            from sklearn.linear_model import Ridge
            
            # Créer l'ensemble
            estimators = [(name, results[name]['model']) for name in top_3_names]
            
            ensemble = StackingRegressor(
                estimators=estimators,
                final_estimator=Ridge(alpha=1.0),
                cv=3,
                n_jobs=-1
            )
            
            print(f"   Entraînement de l'ensemble...")
            ensemble.fit(X_train, y_train)
            
            # Évaluer l'ensemble
            y_pred_ensemble = ensemble.predict(X_val)
            ensemble_angular_errors = angular_error(y_val, y_pred_ensemble)
            ensemble_mean_error = np.mean(ensemble_angular_errors)
            ensemble_pct_30 = np.mean(ensemble_angular_errors <= 30) * 100
            ensemble_pct_45 = np.mean(ensemble_angular_errors <= 45) * 100
            ensemble_pct_90 = np.mean(ensemble_angular_errors <= 90) * 100
            
            print(f"   Erreur angulaire moyenne: {ensemble_mean_error:.2f}°")
            print(f"   Dans ±30°: {ensemble_pct_30:.1f}%")
            print(f"   Dans ±45°: {ensemble_pct_45:.1f}%")
            
            # Comparer avec le meilleur modèle individuel
            if ensemble_mean_error < best_metrics['mean_angular_error']:
                print(f"   ✓ Ensemble meilleur que {best_name} ({best_metrics['mean_angular_error']:.2f}°)")
                best_model = ensemble
                best_name = "Ensemble"
                best_metrics['mean_angular_error'] = ensemble_mean_error
                best_metrics['pct_within_30'] = ensemble_pct_30
                best_metrics['pct_within_45'] = ensemble_pct_45
                best_metrics['pct_within_90'] = ensemble_pct_90
            else:
                print(f"   ✗ Ensemble moins bon que {best_name}")
                
        except ImportError:
            print(f"   ⚠️  StackingRegressor non disponible")
        except Exception as e:
            print(f"   ⚠️  Erreur lors de la création de l'ensemble: {e}")
    
    # Évaluation finale sur test set
    print(f"\n📊 ÉVALUATION FINALE (Test Set):")
    y_pred_test = best_model.predict(X_test)
    
    test_angular_errors = angular_error(y_test, y_pred_test)
    test_mean_error = np.mean(test_angular_errors)
    test_median_error = np.median(test_angular_errors)
    test_pct_30 = np.mean(test_angular_errors <= 30) * 100
    test_pct_45 = np.mean(test_angular_errors <= 45) * 100
    test_pct_90 = np.mean(test_angular_errors <= 90) * 100
    
    print(f"   Erreur angulaire moyenne: {test_mean_error:.2f}°")
    print(f"   Erreur angulaire médiane: {test_median_error:.2f}°")
    print(f"   Dans ±30°: {test_pct_30:.1f}%")
    print(f"   Dans ±45°: {test_pct_45:.1f}%")
    print(f"   Dans ±90°: {test_pct_90:.1f}%")
    
    # Feature importance (si Random Forest)
    if best_name == 'RandomForest':
        feature_names = [
            'elev_mean', 'elev_max',
            'wind_dir_mean', 'wind_dir_max',
            'wind_speed_mean', 'wind_speed_max',
            'temp_min_mean', 'temp_min_max',
            'temp_max_mean', 'temp_max_max',
            'humidity_mean', 'humidity_max',
            'drought_mean', 'drought_max',
            'ndvi_mean', 'ndvi_max',
            'fire_sum', 'fire_mean',
            'wind_east', 'wind_north',
            'elev_wind_interaction', 'drought_ndvi_interaction',
            'fire_shape_ratio'
        ]
        
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        
        print(f"\n🔝 Top 10 features importantes:")
        for i, idx in enumerate(indices, 1):
            print(f"   {i}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    final_metrics = {
        'model_name': best_name,
        'test_mean_angular_error': test_mean_error,
        'test_median_angular_error': test_median_error,
        'test_pct_within_30': test_pct_30,
        'test_pct_within_45': test_pct_45,
        'test_pct_within_90': test_pct_90,
        'n_train': len(X_train),
        'n_test': len(X_test)
    }
    
    return best_model, final_metrics

# ==========================================
# 💾 PARTIE 5 : SAUVEGARDE
# ==========================================

def save_model(model, metrics, save_path):
    """Sauvegarde le modèle et ses métriques"""
    
    print(f"\n💾 Sauvegarde du modèle...")
    
    # Créer le dossier si nécessaire
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Sauvegarder le modèle
    joblib.dump(model, save_path)
    
    # Sauvegarder les métriques
    metrics_path = save_path.replace('.joblib', '_metrics.json')
    import json
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"   ✓ Modèle: {save_path}")
    print(f"   ✓ Métriques: {metrics_path}")

# ==========================================
# 🚀 PIPELINE PRINCIPAL
# ==========================================

"""def main():
    #Pipeline complet d'entraînement
    
    print("\n" + "🔥" * 30)
    print("ENTRAÎNEMENT MODÈLE DE DIRECTION DE PROPAGATION")
    print("🔥" * 30)
    
    # 1. Préparer les données
    #X, y, metadata = prepare_training_data(
    #    tfrecord_dir=TFRECORD_DIR,
    #    max_samples=None  # Utiliser tous les échantillons
    #)
    X = np.load("backend/ml/X.npy")
    y = np.load("backend/ml/y.npy")
    metadata = np.load("backend/ml/metadata.npy")
    
    # 2. Entraîner le modèle
    model, metrics = train_direction_model(X, y, metadata)
    
    # 3. Sauvegarder
    save_model(model, metrics, MODEL_SAVE_PATH)
    
    print("\n✅ ENTRAÎNEMENT TERMINÉ !")
    print(f"   Modèle prêt à être utilisé pour la prédiction")
    print(f"   Erreur moyenne: {metrics['test_mean_angular_error']:.1f}°")
    print(f"   Précision ±45°: {metrics['test_pct_within_45']:.1f}%")
    
    return model, metrics


if __name__ == "__main__":
    # Vérifier que le chemin TFRecord est défini
    if TFRECORD_DIR == "/path/to/your/tfrecord/files":
        print("⚠️  ERREUR: Veuillez définir TFRECORD_DIR avec le chemin vers vos fichiers TFRecord")
        print("   Exemple: TFRECORD_DIR = '/home/user/wildfire_data/tfrecords'")
    else:
        model, metrics = main()

import joblib
#from collect_gee_data import collect_gee_data
from datetime import datetime

# Charger le modèle
model = joblib.load('backend/ml/wildfire_direction_predictor.joblib')
print(model)

# Collecter des données
data = collect_gee_data(34.05, -118.25, datetime(2024, 8, 15))

# Extraire features
features = extract_features_from_dict(data)

# Prédire
direction = model.predict(features.reshape(1, -1))[0]
print(f"Direction prédite : {direction:.1f}°")"""