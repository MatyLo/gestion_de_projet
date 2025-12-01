from fastapi import FastAPI
from pydantic import BaseModel
import requests
import pandas as pd
import joblib
import numpy as np
import json
from datetime import datetime
from process_feature import load_collected_data, extract_features_from_collected_data
from recup_data import collect_all_data

# Chemins des modèles
MODEL_CLASSIF_PATH = "/Users/matylo/Documents/projet_incendie2/backend/ml/wildfire_spread_classifier_advanced.joblib"
MODEL_REG_PATH = "/Users/matylo/Documents/projet_incendie2/backend/ml/wildfire_spread_regressor_advanced.joblib"


app = FastAPI()

# -----------------------------
# 📌 INPUT MODEL
# -----------------------------
class PointRequest(BaseModel):
    lat: float
    lon: float
    date: str  # format YYYY-MM-DD

# -----------------------------
# 🔧 FONCTIONS UTILES
# -----------------------------
def extract_features_from_point(lat, lon, date_str):
    """
    Extract features for a single point (lat, lon) and prepare
    input features compatible with the training pipeline.

    This version includes:
      - sanity checks
      - NaN cleaning
      - fallback values
      - coordinate-specific safety
    """

    # ------------------------------------------
    # 1) Convert date
    # ------------------------------------------
    date = datetime.strptime(date_str, "%Y-%m-%d")

    print(f"\n📍 Extraction des features pour ({lat}, {lon}) — date {date_str}")

    # ------------------------------------------
    # 2) Collect raw data (64×64 grids)
    # ------------------------------------------
    data = collect_all_data(lat, lon, date)

    # Les variables attendues dans data :
    # th, vs, tmmn, tmmx, sph, pdsi
    # elevation
    # PrevFireMask
    # NDVI
    # FireMask (toujours vide dans l'inférence)

    features = []

    # ------------------------------------------
    # 3) Helper pour nettoyer les tableaux
    # ------------------------------------------
    def safe_array(arr, name):
        """Clean an array: replace NaN, inf, or empty arrays."""
        arr = np.asarray(arr, dtype=float)

        if arr.size == 0:
            print(f"⚠️  {name} vide → tableau de zéros")
            return np.zeros(4096)

        if np.isnan(arr).any() or np.isinf(arr).any():
            print(f"⚠️  {name} contient NaN ou inf → remplacement")
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        return arr

    # liste des variables environnementales
    env_vars = ['elevation', 'th', 'vs', 'tmmn', 'tmmx', 'sph', 'pdsi', 'NDVI']

    # ------------------------------------------
    # 4) Environmental features (mean + max)
    # ------------------------------------------
    for var in env_vars:
        arr = safe_array(data[var], var)
        features.extend([np.mean(arr), np.max(arr)])

    # ------------------------------------------
    # 5) Fire history
    # ------------------------------------------
    prev_fire = safe_array(data['PrevFireMask'], "PrevFireMask")
    features.extend([np.sum(prev_fire), np.mean(prev_fire)])

    # ------------------------------------------
    # 6) Wind vector (magnitude + direction → east/north)
    # ------------------------------------------
    wind_dir = safe_array(data['th'], "th")
    wind_speed = safe_array(data['vs'], "vs")

    wind_dir_mean = np.mean(wind_dir)
    wind_speed_mean = np.mean(wind_speed)

    wind_east = wind_speed_mean * np.cos(np.radians(wind_dir_mean))
    wind_north = wind_speed_mean * np.sin(np.radians(wind_dir_mean))

    features.append(wind_east)
    features.append(wind_north)

    # ------------------------------------------
    # 7) Interactions
    # ------------------------------------------
    elev_mean = np.mean(safe_array(data['elevation'], "elevation"))
    drought_mean = np.mean(safe_array(data['pdsi'], "pdsi"))
    ndvi_mean = np.mean(safe_array(data['NDVI'], "NDVI"))

    features.append(elev_mean * wind_speed_mean)         # vent * relief
    features.append(drought_mean * ndvi_mean)            # sécheresse × végétation

    # ------------------------------------------
    # 8) Fire shape ratio (sécurité : si pas de feu → 1.0)
    # ------------------------------------------
    try:
        fire_mask_2d = prev_fire.reshape(64, 64)

        if np.sum(fire_mask_2d) > 0:
            y_idx, x_idx = np.where(fire_mask_2d > 0)
            fire_width = np.max(x_idx) - np.min(x_idx)
            fire_height = np.max(y_idx) - np.min(y_idx)
            shape_ratio = fire_width / (fire_height + 1e-6)
        else:
            shape_ratio = 1.0   # valeur par défaut
    except Exception as e:
        print(f"⚠️  Impossible de calculer shape_ratio : {e}")
        shape_ratio = 1.0

    features.append(shape_ratio)

    # ------------------------------------------
    # 9) Dernière sécurité : enlever NaN dans features
    # ------------------------------------------
    features = np.array(features, dtype=float)
    if np.isnan(features).any() or np.isinf(features).any():
        print("⚠️  Nettoyage final des NaN dans les features")
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    print("✅ Features générées :")
    print(features)

    return features



"""def extract_features_from_point(lat, lon, date_str):
    
    #Collecte les données pour un point et transforme en features ML.
    
    date = datetime.strptime(date_str, "%Y-%m-%d")

    # Collecte toutes les données nécessaires
    data = collect_all_data(lat, lon, date)

    # Sauvegarde temporaire (optionnel)
    #np.save("backend/data/processed/collected_data.npy", data)

    # Extraction des features (comme dans process_data.py)
    features = []

    # 1. Environmental stats (mean, max)
    for var_name in ['elevation', 'th', 'vs', 'tmmn', 'tmmx', 'sph', 'pdsi', 'NDVI']:
        arr = data[var_name]
        features.extend([np.mean(arr), np.max(arr)])

    # 2. Fire mask stats
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
    features.append(np.mean(data['elevation']) * wind_speed_mean)
    features.append(np.mean(data['pdsi']) * np.mean(data['NDVI']))

    # 5. Fire shape ratio
    try:
        fire_mask_2d = prev_fire.reshape(64, 64)
        if np.sum(fire_mask_2d) > 0:
            y_indices, x_indices = np.where(fire_mask_2d > 0)
            shape_ratio = (np.max(x_indices) - np.min(x_indices)) / (np.max(y_indices) - np.min(y_indices) + 1e-6)
        else:
            shape_ratio = 1.0
    except Exception:
        shape_ratio = 1.0

    features.append(shape_ratio)
    print(np.array(features))

    return np.array(features)
"""
# -----------------------------
# 🔮 Chargement des modèles
# -----------------------------
try:
    classifier = joblib.load(MODEL_CLASSIF_PATH)
    regressor = joblib.load(MODEL_REG_PATH)
    print("✅ Modèles chargés avec succès")
except Exception as e:
    print(f"❌ Erreur chargement modèles: {e}")
    classifier = None
    regressor = None

# -----------------------------
# 🚀 ROUTE PRINCIPALE : /predict_point
# -----------------------------
@app.post("/predict_point")
def predict_point(req: PointRequest):
    X = extract_features_from_point(req.lat, req.lon, req.date).reshape(1, -1)

    pred = classifier.predict(X)[0]
    prob = regressor.predict(X)[0]


    return {
        "latitude": req.lat,
        "longitude": req.lon,
        "date": req.date,
        "risk_prediction": int(pred),
        "risk_propagation": round(float(prob), 4)
    }
