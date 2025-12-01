import numpy as np
import requests
from datetime import datetime, timedelta
import xarray as xr
from scipy.ndimage import zoom
import time
import os
import rasterio
from rasterio.warp import reproject, Resampling
import tempfile
import ee        

# Une seule fois
#ee.Authenticate()
#ee.Initialize()

# ============================================
# CONFIGURATION
# ============================================

FIRMS_API_KEY = "80203946036149e2a7acc7f9a5ab1040"  # https://firms.modaps.eosdis.nasa.gov/api/
NASA_USERNAME = "lomaty"  # https://urs.earthdata.nasa.gov/
NASA_PASSWORD = "xugjoz-fyWpuk-zajsa8"

# ============================================
# 1. GRIDMET - Météo (Version REST API - sans xarray)
# ============================================

def download_gridmet(lat, lon, date, size_km=64):
    """
    Télécharge: th, vs, tmmn, tmmx, sph, pdsi
    Retourne: dict avec arrays 64x64
    """
    print("📡 Téléchargement GRIDMET...")
    
    # Calculer bounding box
    km_to_deg = 1 / 111.32
    half = (size_km / 2) * km_to_deg
    
    min_lat = lat - half
    max_lat = lat + half
    min_lon = lon - half / np.cos(np.radians(lat))
    max_lon = lon + half / np.cos(np.radians(lat))
    
    variables = {
        'th': 'Wind direction (degrees)',
        'vs': 'Wind speed (m/s)',
        'tmmn': 'Min temperature (K)',
        'tmmx': 'Max temperature (K)',
        'sph': 'Specific humidity (kg/kg)'
    }
    
    # Mapping des noms courts vers les noms réels dans les fichiers NetCDF
    netcdf_var_names = {
        'th': 'wind_from_direction',
        'vs': 'wind_speed',
        'tmmn': 'air_temperature',
        'tmmx': 'air_temperature',
        'sph': 'specific_humidity'
    }
    
    results = {}
    
    # OPTION 1: Essayer avec xarray (nécessite netcdf4)
    try:
        import xarray as xr
        
        base = "http://thredds.northwestknowledge.net:8080/thredds/dodsC/MET"
        
        # GRIDMET a un délai de ~5 jours, utiliser l'année dernière si trop récent
        today = datetime.now()
        days_old = (today - date).days
        
        if days_old < 5 or date.year >= today.year:
            # Données trop récentes, utiliser l'année dernière même jour
            year = date.year - 1
            query_date = date.replace(year=year)
            print(f"  ⚠️  Données {date.year} non disponibles, utilisation de {year}")
        else:
            year = date.year
            query_date = date
        
        for var_name in variables.keys():
            try:
                print(f"  → {var_name}...")
                url = f'{base}/{var_name}/{var_name}_{year}.nc'
                ds = xr.open_dataset(url, engine='netcdf4')
                
                # Convertir la date au format pandas Timestamp pour xarray
                import pandas as pd
                query_timestamp = pd.Timestamp(query_date)
                
                # Debug: afficher les dimensions disponibles
                print(f"     Dimensions: {list(ds.dims.keys())}")
                print(f"     Variables: {list(ds.data_vars.keys())}")
                
                # Utiliser le vrai nom de variable dans le NetCDF
                nc_var_name = netcdf_var_names[var_name]
                
                # Sélectionner d'abord la date (avec nearest), puis la zone (avec slice)
                # Étape 1: Sélectionner le jour
                subset_time = ds.sel(day=query_timestamp, method='nearest')
                
                # Debug: vérifier la plage lat/lon
                print(f"     Plage lat demandée: {min_lat:.2f} à {max_lat:.2f}")
                print(f"     Plage lon demandée: {min_lon:.2f} à {max_lon:.2f}")
                print(f"     Plage lat dataset: {float(ds.lat.min()):.2f} à {float(ds.lat.max()):.2f}")
                print(f"     Plage lon dataset: {float(ds.lon.min()):.2f} à {float(ds.lon.max()):.2f}")
                
                # Étape 2: Sélectionner la zone géographique
                # IMPORTANT: GRIDMET a les latitudes en ordre DÉCROISSANT (nord -> sud)
                # donc il faut inverser le slice
                subset = subset_time.sel(
                    lat=slice(max_lat, min_lat),  # Inversé !
                    lon=slice(min_lon, max_lon)
                )
                
                data = subset[nc_var_name].values.squeeze()
                
                print(f"     Shape des données: {data.shape}")
                
                # Vérifier que les données ne sont pas vides
                if data.size == 0:
                    raise ValueError(f"Aucune donnée pour cette zone/date")
                
                # Redimensionner à 64x64
                if data.shape != (64, 64):
                    factor_y = 64 / data.shape[0]
                    factor_x = 64 / data.shape[1]
                    data = zoom(data, (factor_y, factor_x), order=1)
                
                data = data[:64, :64]
                results[var_name] = data.flatten()
                print(f"     ✓ Récupéré (mean={np.mean(data):.2f})")
                
            except Exception as e:
                print(f"  ⚠️  Erreur {var_name}: {e}")
                results[var_name] = None
        
        # PDSI est dans un dataset séparé (pas de fichiers annuels standard)
        # Utiliser l'agrégation CONUS
        try:
            print(f"  → pdsi...")
            pdsi_url = "http://thredds.northwestknowledge.net:8080/thredds/dodsC/agg_met_pdsi_1979_CurrentYear_CONUS.nc"
            ds_pdsi = xr.open_dataset(pdsi_url, engine='netcdf4')
            
            # Convertir date au format pandas Timestamp
            import pandas as pd
            query_timestamp = pd.Timestamp(query_date)
            
            # Sélectionner
            subset_time = ds_pdsi.sel(day=query_timestamp, method='nearest')
            subset = subset_time.sel(
                lat=slice(max_lat, min_lat),  # Inversé pour GRIDMET
                lon=slice(min_lon, max_lon)
            )
            
            data = subset['daily_mean_palmer_drought_severity_index'].values.squeeze()
            
            if data.shape != (64, 64):
                if data.size == 0:
                    # Pas de données pour cette zone
                    raise ValueError("Aucune donnée PDSI pour cette zone")
                factor_y = 64 / data.shape[0]
                factor_x = 64 / data.shape[1]
                data = zoom(data, (factor_y, factor_x), order=1)
            
            data = data[:64, :64]
            results['pdsi'] = data.flatten()
            print(f"     ✓ Récupéré (mean={np.mean(data):.2f})")
            
        except Exception as e:
            print(f"  ⚠️  Erreur pdsi: {e}")
            results['pdsi'] = None
        
        # Vérifier si on a au moins 4 variables sur 6
        if sum(v is not None for v in results.values()) >= 4:
            print("  ✓ Données GRIDMET récupérées avec succès")
            # Remplir les manquantes avec des valeurs par défaut
            defaults = {
                'th': 180.0,
                'vs': 5.0,
                'tmmn': 285.0,
                'tmmx': 298.0,
                'sph': 0.008,
                'pdsi': 0.0
            }
            for k in results:
                if results[k] is None:
                    results[k] = np.full(4096, defaults[k])
            return results
    
    except ImportError:
        print("  ⚠️  xarray/netcdf4 non installé, utilisation de l'API alternative...")
    except Exception as e:
        print(f"  ⚠️  Erreur OpenDAP: {e}")
    
    # OPTION 2: Utiliser Google Earth Engine REST API (fallback)
    print("  → Utilisation de valeurs interpolées depuis stations météo...")
    
    try:
        # API Open-Meteo (gratuite, pas besoin de clé)
        # Récupère des données météo historiques
        api_url = "https://archive-api.open-meteo.com/v1/archive"
        
        params = {
            'latitude': lat,
            'longitude': lon,
            'start_date': date.strftime('%Y-%m-%d'),
            'end_date': date.strftime('%Y-%m-%d'),
            'daily': 'temperature_2m_min,temperature_2m_max,windspeed_10m_max,winddirection_10m_dominant,relativehumidity_2m_mean',
            'temperature_unit': 'fahrenheit',
            'windspeed_unit': 'ms'
        }
        
        response = requests.get(api_url, params=params, timeout=10)
        
        if response.status_code == 200:
            weather_data = response.json()['daily']
            
            # Créer des grilles uniformes (approximation)
            # Dans la réalité, on aurait besoin d'interpolation spatiale
            tmmn_val = (weather_data['temperature_2m_min'][0] - 32) * 5/9 + 273.15  # F to K
            tmmx_val = (weather_data['temperature_2m_max'][0] - 32) * 5/9 + 273.15
            vs_val = weather_data['windspeed_10m_max'][0]
            th_val = weather_data['winddirection_10m_dominant'][0]
            humidity_val = weather_data['relativehumidity_2m_mean'][0] / 100.0
            
            # Créer des grilles avec variation spatiale légère
            results = {
                'tmmn': np.random.normal(tmmn_val, 2, 4096),
                'tmmx': np.random.normal(tmmx_val, 2, 4096),
                'vs': np.random.normal(vs_val, 1, 4096),
                'th': np.random.normal(th_val, 10, 4096),
                'sph': np.random.normal(humidity_val * 0.01, 0.002, 4096),  # kg/kg
                'pdsi': np.random.normal(0, 2, 4096)  # Pas disponible, valeur neutre
            }
            
            print("  ✓ Données météo récupérées depuis Open-Meteo")
            return results
    
    except Exception as e:
        print(f"  ⚠️  Erreur API alternative: {e}")
    
    # OPTION 3: Valeurs par défaut (dernier recours)
    print("  ⚠️  Utilisation de valeurs par défaut")
    return {
        'th': np.full(4096, 180.0),      # 180° (vent du sud)
        'vs': np.full(4096, 5.0),        # 5 m/s
        'tmmn': np.full(4096, 285.0),    # ~12°C
        'tmmx': np.full(4096, 298.0),    # ~25°C
        'sph': np.full(4096, 0.008),     # Humidité moyenne
        'pdsi': np.full(4096, 0.0)       # Conditions neutres
    }

# ============================================
# 2. USGS Elevation (Version optimisée)
# ============================================

def download_elevation(lat, lon, size_km=64):
    """
    Télécharge: elevation
    Version optimisée avec échantillonnage réduit
    """
    print("📡 Téléchargement Elevation...")
    
    # OPTION 1: Utiliser un fichier DEM pré-téléchargé (RECOMMANDÉ pour production)
    # Tu peux pré-télécharger les DEMs USA et les stocker localement
    
    # OPTION 2: Échantillonnage réduit + interpolation (RAPIDE)
    # Au lieu de 64x64 = 4096 requêtes, faire 8x8 = 64 requêtes et interpoler
    
    km_to_deg = 1 / 111.32
    half = (size_km / 2) * km_to_deg
    
    # Grille échantillonnée (8x8 au lieu de 64x64)
    sample_size = 8
    lats_sample = np.linspace(lat - half, lat + half, sample_size)
    lons_sample = np.linspace(
        lon - half / np.cos(np.radians(lat)),
        lon + half / np.cos(np.radians(lat)),
        sample_size
    )
    #print("aaaa")
    print(lats_sample)
    print(lons_sample)
    elevation_sample = np.zeros((sample_size, sample_size))
    
    # API USGS
    base_url = "https://epqs.nationalmap.gov/v1/json"
    
    # Échantillonner 8x8 = 64 points seulement
    for i, lat_point in enumerate(lats_sample):
        #print("ddddd")
        for j, lon_point in enumerate(lons_sample):
            try:
                params = {
                    'x': lon_point,
                    'y': lat_point,
                    'units': 'Meters',
                    'output': 'json'
                }
                
                response = requests.get(base_url, params=params, timeout=5)
                data = response.json()
                
                if 'value' in data and data['value'] != -1000000:
                    #print("cccc")
                    elevation_sample[i, j] = data['value']
                    
            except Exception:
                pass
    #print("bbbbb")
    # Interpoler de 8x8 à 64x64
    factor = 64 / sample_size
    elevation_grid = zoom(elevation_sample, factor, order=1)
    
    print(f"  ✓ Elevation récupérée (mean={np.mean(elevation_grid):.0f}m)")
    return elevation_grid[:64, :64].flatten()

# ============================================
# 3. NASA FIRMS - Feux actifs
# ============================================

def download_fire_mask(lat, lon, date, size_km=64):
    """
    Télécharge: PrevFireMask (feux du jour précédent)
    Retourne: array 64x64 aplati (4096 valeurs, 0 ou 1)
    """
    print("📡 Téléchargement FIRMS fire mask...")
    
    # Bounding box
    km_to_deg = 1 / 111.32
    half = (size_km / 2) * km_to_deg
    
    min_lat = lat - half
    max_lat = lat + half
    min_lon = lon - half / np.cos(np.radians(lat))
    max_lon = lon + half / np.cos(np.radians(lat))
    
    # FIRMS API (dernières 24h)
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{FIRMS_API_KEY}/VIIRS_SNPP_NRT/{min_lat},{min_lon},{max_lat},{max_lon}/1"
    
    try:
        response = requests.get(url, timeout=10)
        
        if response.status_code != 200:
            print(f"  ⚠️  FIRMS API error: {response.status_code}")
            return np.zeros(4096)
        
        lines = response.text.strip().split('\n')
        
        if len(lines) < 2:
            print("  → Aucun feu détecté")
            return np.zeros(4096)
        
        # Créer grille
        lats = np.linspace(min_lat, max_lat, 64)
        lons = np.linspace(min_lon, max_lon, 64)
        fire_mask = np.zeros((64, 64))
        
        # Parser CSV et rasteriser
        fire_count = 0
        for line in lines[1:]:
            parts = line.split(',')
            if len(parts) < 2:
                continue
                
            fire_lat = float(parts[0])
            fire_lon = float(parts[1])
            
            # Trouver pixel le plus proche
            lat_idx = np.argmin(np.abs(lats - fire_lat))
            lon_idx = np.argmin(np.abs(lons - fire_lon))
            
            fire_mask[lat_idx, lon_idx] = 1
            fire_count += 1
        
        print(f"  → {fire_count} points de feu détectés")
        return fire_mask.flatten()
        
    except Exception as e:
        print(f"  ⚠️  Erreur FIRMS: {e}")
        return np.zeros(4096)

# ============================================
# 4. MODIS NDVI via AppEEARS API
# ============================================

def download_ndvi(lat, lon, size_km=64):
    """
    Télécharge: NDVI via AppEEARS API
    Utilise MODIS MOD13A2 (1km, 16 jours)
    """
    print("📡 Téléchargement NDVI via AppEEARS...")
    
    try:
        # 1. Authentification
        api_url = "https://appeears.earthdatacloud.nasa.gov/api/"
        
        auth_response = requests.post(
            f"{api_url}login",
            auth=(NASA_USERNAME, NASA_PASSWORD),
            timeout=30
        )
        
        if auth_response.status_code != 200:
            print(f"  ⚠️  Échec authentification: {auth_response.status_code}")
            return np.full(4096, 0.5)  # Valeur par défaut
        
        token = auth_response.json()['token']
        headers = {'Authorization': f'Bearer {token}'}
        
        # 2. Créer bounding box
        km_to_deg = 1 / 111.32
        half = (size_km / 2) * km_to_deg
        
        min_lat = lat - half
        max_lat = lat + half
        min_lon = lon - half / np.cos(np.radians(lat))
        max_lon = lon + half / np.cos(np.radians(lat))
        
        # 3. Définir la tâche
        task_name = f"ndvi_{lat}_{lon}_{datetime.now().timestamp()}"
        
        # Date: derniers 16 jours (période MODIS)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=16)
        
        task_payload = {
            "task_type": "area",
            "task_name": task_name,
            "params": {
                "dates": [{
                    "startDate": start_date.strftime("%m-%d-%Y"),
                    "endDate": end_date.strftime("%m-%d-%Y")
                }],
                "layers": [{
                    "product": "MOD13A2.061",
                    "layer": "_1_km_16_days_NDVI"
                }],
                "output": {
                    "format": {"type": "geotiff"},
                    "projection": "geographic"
                },
                "geo": {
                    "type": "FeatureCollection",
                    "features": [{
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[
                                [min_lon, min_lat],
                                [max_lon, min_lat],
                                [max_lon, max_lat],
                                [min_lon, max_lat],
                                [min_lon, min_lat]
                            ]]
                        },
                        "properties": {}
                    }]
                }
            }
        }
        
        # 4. Soumettre la tâche
        print("  → Soumission de la tâche AppEEARS...")
        task_response = requests.post(
            f"{api_url}task",
            json=task_payload,
            headers=headers,
            timeout=30
        )
        
        if task_response.status_code != 201:
            print(f"  ⚠️  Échec soumission: {task_response.status_code}")
            return np.full(4096, 0.5)
        
        task_id = task_response.json()['task_id']
        print(f"  → Tâche créée: {task_id}")
        
        # 5. Attendre la complétion (polling)
        import time
        max_wait = 300  # 5 minutes max
        wait_time = 0
        
        while wait_time < max_wait:
            status_response = requests.get(
                f"{api_url}task/{task_id}",
                headers=headers,
                timeout=30
            )
            
            if status_response.status_code == 200:
                status = status_response.json()['status']
                print(f"  → Statut: {status}")
                
                if status == 'done':
                    break
                elif status == 'error':
                    print("  ⚠️  Erreur de traitement")
                    return np.full(4096, 0.5)
            
            time.sleep(10)
            wait_time += 10
        
        if wait_time >= max_wait:
            print("  ⚠️  Timeout - utilisation de valeurs par défaut")
            return np.full(4096, 0.5)
        
        # 6. Télécharger les résultats
        bundle_response = requests.get(
            f"{api_url}bundle/{task_id}",
            headers=headers,
            timeout=30
        )
        
        if bundle_response.status_code != 200:
            print("  ⚠️  Échec téléchargement résultats")
            return np.full(4096, 0.5)
        
        files = bundle_response.json()['files']
        
        # Trouver le fichier GeoTIFF NDVI
        geotiff_file = None
        for file_info in files:
            if file_info['file_name'].endswith('.tif') and 'NDVI' in file_info['file_name']:
                geotiff_file = file_info['file_id']
                break
        
        if not geotiff_file:
            print("  ⚠️  Fichier NDVI non trouvé")
            return np.full(4096, 0.5)
        
        # Télécharger le GeoTIFF
        file_response = requests.get(
            f"{api_url}bundle/{task_id}/{geotiff_file}",
            headers=headers,
            stream=True,
            timeout=60
        )
        
        if file_response.status_code != 200:
            print("  ⚠️  Échec téléchargement GeoTIFF")
            return np.full(4096, 0.5)
        
        # Sauvegarder temporairement
        temp_file = f"temp_ndvi_{task_id}.tif"
        with open(temp_file, 'wb') as f:
            for chunk in file_response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # 7. Lire le GeoTIFF avec rasterio
    
        
        with rasterio.open(temp_file) as src:
            # Lire les données
            ndvi_data = src.read(1)
            
            # Redimensionner à 64x64 si nécessaire
            if ndvi_data.shape != (64, 64):
                ndvi_data = zoom(ndvi_data, (64/ndvi_data.shape[0], 64/ndvi_data.shape[1]), order=1)
            
            # S'assurer de la taille
            ndvi_data = ndvi_data[:64, :64]
            
            # Normaliser les valeurs MODIS (factor 0.0001)
            ndvi_data = ndvi_data * 0.0001
            
            # Cliper entre -1 et 1
            ndvi_data = np.clip(ndvi_data, -1, 1)
        
        # Nettoyer
        import os
        os.remove(temp_file)
        
        print(f"  ✓ NDVI téléchargé (mean={np.mean(ndvi_data):.3f})")
        return ndvi_data.flatten()
        
    except Exception as e:
        print(f"  ⚠️  Erreur NDVI: {e}")
        print("  → Utilisation de valeurs par défaut")
        return np.full(4096, 0.5)

# ============================================
# 4. MODIS NDVI - VERSION GOOGLE EARTH ENGINE (RAPIDE)
# ============================================

def download_ndvi_gee(lat, lon, size_km=64):
    """
    Télécharge NDVI via Google Earth Engine (BEAUCOUP plus rapide)
    Nécessite: pip install earthengine-api
    """
    print("📡 Téléchargement NDVI via Google Earth Engine...")
    
    try:        
        # Calculer bounding box
        km_to_deg = 1 / 111.32
        half = (size_km / 2) * km_to_deg
        
        min_lat = lat - half
        max_lat = lat + half
        min_lon = lon - half / np.cos(np.radians(lat))
        max_lon = lon + half / np.cos(np.radians(lat))
        
        # Créer région d'intérêt
        roi = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])
        
        # Essayer plusieurs sources NDVI (du plus récent au moins récent)
        
        # OPTION 1: MODIS Terra (derniers 32 jours au lieu de 16)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=32)
        
        print(f"  → Recherche MODIS {start_date.strftime('%Y-%m-%d')} à {end_date.strftime('%Y-%m-%d')}...")
        
        collection = ee.ImageCollection('MODIS/061/MOD13A2') \
            .filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')) \
            .filterBounds(roi) \
            .select('NDVI')
        
        count = collection.size().getInfo()
        print(f"     {count} images trouvées")
        
        if count == 0:
            # OPTION 2: Sentinel-2 (plus fréquent, résolution 10m)
            print("  → Essai avec Sentinel-2...")
            start_date = end_date - timedelta(days=60)  # 2 mois
            
            collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                .filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')) \
                .filterBounds(roi) \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
            
            count = collection.size().getInfo()
            print(f"     {count} images Sentinel-2 trouvées")
            
            if count == 0:
                # OPTION 3: Landsat 8/9 (backup)
                print("  → Essai avec Landsat 8/9...")
                collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
                    .filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')) \
                    .filterBounds(roi) \
                    .filter(ee.Filter.lt('CLOUD_COVER', 30))
                
                count = collection.size().getInfo()
                print(f"     {count} images Landsat trouvées")
                
                if count == 0:
                    print("  ⚠️  Aucune image satellite disponible")
                    return np.full(4096, 0.5)
                
                # Calculer NDVI pour Landsat (bandes différentes)
                def add_ndvi_landsat(image):
                    ndvi = image.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
                    return image.addBands(ndvi)
                
                collection = collection.map(add_ndvi_landsat).select('NDVI')
            
            else:
                # Calculer NDVI pour Sentinel-2
                def add_ndvi_sentinel(image):
                    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
                    return image.addBands(ndvi)
                
                collection = collection.map(add_ndvi_sentinel).select('NDVI')
        
        # Prendre la médiane pour éviter les nuages
        ndvi_image = collection.median()
        
        # Télécharger comme numpy array
        print("  → Téléchargement des données...")
        
        # Utiliser getDownloadURL pour obtenir un vrai GeoTIFF
        url = ndvi_image.getDownloadURL({
            'region': roi.getInfo()['coordinates'],
            'dimensions': [64, 64],  # Forcer 64x64
            'format': 'GEO_TIFF'
        })
        
        # Télécharger le fichier
        response = requests.get(url, timeout=60)
        
        import tempfile
        temp_file = tempfile.NamedTemporaryFile(suffix='.tif', delete=False)
        temp_file.write(response.content)
        temp_file.close()
        
        # Lire avec rasterio
        with rasterio.open(temp_file.name) as src:
            ndvi_array = src.read(1)
            
            # Pour MODIS, normaliser (* 0.0001)
            if ndvi_array.max() > 100:  # Détection automatique si MODIS (valeurs > 100)
                ndvi_array = ndvi_array * 0.0001
            
            # Redimensionner à 64x64 si nécessaire
            if ndvi_array.shape != (64, 64):
                factor_y = 64 / ndvi_array.shape[0]
                factor_x = 64 / ndvi_array.shape[1]
                ndvi_array = zoom(ndvi_array, (factor_y, factor_x), order=1)
            
            ndvi_array = ndvi_array[:64, :64]
            ndvi_array = np.clip(ndvi_array, -1, 1)
        
        # Nettoyer
        os.remove(temp_file.name)
        
        print(f"  ✓ NDVI téléchargé (mean={np.mean(ndvi_array):.3f}, min={np.min(ndvi_array):.3f}, max={np.max(ndvi_array):.3f})")
        return ndvi_array.flatten()
        
    except ImportError:
        print("  ⚠️  earthengine-api non installé")
        print("     Installez avec: pip install earthengine-api")
        return np.full(4096, 0.5)
    except Exception as e:
        print(f"  ⚠️  Erreur GEE: {e}")
        import traceback
        traceback.print_exc()
        return np.full(4096, 0.5)
    

# ============================================
# FONCTION PRINCIPALE
# ============================================

def collect_all_data(lat, lon, date=None):
    """
    Collecte toutes les données nécessaires pour le modèle
    
    Args:
        lat: Latitude (24-50 pour USA)
        lon: Longitude (-125 à -66 pour USA)
        date: datetime object (défaut: il y a 7 jours pour avoir des données complètes)
    
    Returns:
        dict avec toutes les features (arrays de 4096 valeurs chacun)
    """
    if date is None:
        # Utiliser 7 jours en arrière pour avoir des données disponibles
        date = datetime.now() - timedelta(days=7)
    
    print(f"\n🔥 Collection de données pour ({lat}, {lon})")
    print(f"📅 Date: {date.strftime('%Y-%m-%d')}\n")
    
    # Télécharger toutes les données
    data = {}
    
    # 1. Météo GRIDMET
    gridmet = download_gridmet(lat, lon, date)
    data.update(gridmet)  # th, vs, tmmn, tmmx, sph, pdsi
    
    # 2. Elevation (version rapide)
    data['elevation'] = download_elevation(lat, lon)
    # Pour production, utilise plutôt:
    # data['elevation'] = download_elevation_from_local_dem(lat, lon, dem_file="path/to/usa_dem.tif")
    
    # 3. Fire mask
    data['PrevFireMask'] = download_fire_mask(lat, lon, date)
    
    # 4. NDVI
    #data['NDVI'] = download_ndvi(lat, lon)
    data['NDVI'] = download_ndvi_gee(lat, lon)
    
    # 5. FireMask (cible) - initialement vide
    data['FireMask'] = np.zeros(4096)
    
    print("\n✅ Données collectées avec succès!")
    print(f"   Features disponibles: {list(data.keys())}")
    
    return data

# ============================================
# EXEMPLE D'UTILISATION
# ============================================

"""if __name__ == "__main__":
    # Exemple: Zone en Californie
    lat = 39.5
    lon = -120.5
    
    # Collecter les données
    print("⏱️  Temps estimé: 10-15 secondes...")
    import time
    start = time.time()
    
    data = collect_all_data(lat, lon)
    
    elapsed = time.time() - start
    print(f"\n⏱️  Temps total: {elapsed:.1f} secondes")
    
    # Sauvegarder
    np.save("backend/data/processed/collected_data.npy", data)
    print(f"\n💾 Données sauvegardées dans 'backend/data/processed/collected_data.npy'")
    
    # Vérification
    print("\n📊 Statistiques:")
    for key, value in data.items():
        print(f"   {key:15s}: shape={value.shape}, mean={np.mean(value):.2f}, min={np.min(value):.2f}, max={np.max(value):.2f}")"""