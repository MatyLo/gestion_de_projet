"""
NASA FIRMS API Integration
Récupère les feux actifs depuis la NASA FIRMS (Fire Information for Resource Management System)
"""
import requests
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import os

# NASA FIRMS API endpoint
FIRMS_API_BASE = "https://firms.modaps.eosdis.nasa.gov/api"

# NASA FIRMS API Key (MAP_KEY)
# Obtenez votre clé sur: https://firms.modaps.eosdis.nasa.gov/api/
NASA_FIRMS_API_KEY = os.getenv("NASA_FIRMS_API_KEY", "002d242100ffdb6932ac2f3d41135492")

def get_firms_fires(
    area: Optional[str] = None,
    country: Optional[str] = None,
    bbox: Optional[tuple] = None,
    days: int = 1,
    source: str = "VIIRS_NOAA20_NRT"
) -> List[Dict]:
    """
    Récupère les feux actifs depuis la NASA FIRMS.
    
    Args:
        area: Zone géographique (ex: "world", "usa", "canada")
        country: Code pays ISO (ex: "FRA", "USA")
        bbox: Bounding box (min_lng, min_lat, max_lng, max_lat)
        days: Nombre de jours de données à récupérer (1-10)
        source: Source de données ("MODIS_NRT", "VIIRS_NOAA20_NRT", "VIIRS_SNPP_NRT")
    
    Returns:
        Liste de dictionnaires contenant les données des feux
    """
    try:
        # Format correct de l'API FIRMS: area/csv/{MAP_KEY}/{SOURCE}/{bbox}/{days}
        # ou country/csv/{MAP_KEY}/{SOURCE}/{country}/{days}
        
        if country:
            # Endpoint pour un pays spécifique
            # Format: country/csv/{MAP_KEY}/{SOURCE}/{country}/{days}
            url = f"{FIRMS_API_BASE}/country/csv/{NASA_FIRMS_API_KEY}/{source}/{country}/{days}"
        elif bbox:
            # Endpoint pour une bounding box
            # Format: [west, south, east, north] = [min_lng, min_lat, max_lng, max_lat]
            min_lng, min_lat, max_lng, max_lat = bbox
            url = f"{FIRMS_API_BASE}/area/csv/{NASA_FIRMS_API_KEY}/{source}/{min_lng},{min_lat},{max_lng},{max_lat}/{days}"
        elif area:
            # Endpoint pour une zone prédéfinie
            url = f"{FIRMS_API_BASE}/area/csv/{NASA_FIRMS_API_KEY}/{source}/{area}/{days}"
        else:
            # Par défaut, récupérer les données mondiales récentes
            # Format: [west, south, east, north] = [-180, -90, 180, 90]
            url = f"{FIRMS_API_BASE}/area/csv/{NASA_FIRMS_API_KEY}/{source}/-180,-90,180,90/{days}"
        
        print(f"Requête FIRMS: {url[:150]}...")  # Debug
        
        response = requests.get(url, timeout=30)
        
        # Vérifier la réponse
        response.raise_for_status()
        response_text = response.text.strip()
        
        # Si l'API retourne une erreur dans le texte
        if "Invalid" in response_text or "Error" in response_text:
            error_msg = response_text[:200] if response_text else f"Status {response.status_code}"
            print(f"Erreur API FIRMS: {error_msg}")
            print(f"URL utilisée: {url[:150]}")
            # Essayer avec MODIS si c'était VIIRS
            if source == "VIIRS_NOAA20_NRT":
                print("Tentative avec MODIS_NRT...")
                return get_firms_fires(area=area, country=country, bbox=bbox, days=days, source="MODIS_NRT")
            return []
        
        # Vérifier si la réponse est vide
        if not response_text:
            print("Aucune donnée retournée par l'API FIRMS")
            return []
        
        # Parser le CSV
        lines = response_text.split('\n')
        if len(lines) < 2:
            print(f"Pas assez de lignes dans la réponse: {len(lines)}")
            if len(lines) > 0:
                print(f"Contenu: {lines[0][:200]}")
            return []
        
        headers = lines[0].split(',')
        fires = []
        
        for line in lines[1:]:
            if not line.strip():
                continue
            values = line.split(',')
            if len(values) < len(headers):
                continue
            
            fire_data = {}
            for i, header in enumerate(headers):
                header = header.strip().strip('"')
                value = values[i].strip().strip('"') if i < len(values) else ""
                
                # Convertir les valeurs numériques
                if header in ['latitude', 'longitude', 'brightness', 'bright_t31', 'bright_ti4', 'bright_ti5', 'frp', 'confidence']:
                    try:
                        fire_data[header] = float(value) if value else None
                    except ValueError:
                        fire_data[header] = None
                else:
                    fire_data[header] = value
            
            # Standardiser les noms de colonnes
            # bright_ti4 est utilisé par VIIRS, brightness par MODIS
            brightness_value = fire_data.get('brightness') or fire_data.get('bright_ti4') or fire_data.get('bright_ti5') or 350
            
            fire = {
                'lat': fire_data.get('latitude'),
                'lng': fire_data.get('longitude'),
                'brightness': float(brightness_value) if brightness_value else 350,
                'confidence': fire_data.get('confidence'),
                'acq_date': fire_data.get('acq_date', ''),
                'acq_time': fire_data.get('acq_time', ''),
                'satellite': fire_data.get('satellite', ''),
                'instrument': fire_data.get('instrument', ''),
                'frp': fire_data.get('frp'),  # Fire Radiative Power
                'daynight': fire_data.get('daynight', ''),
                'type': fire_data.get('type', '')
            }
            
            # Filtrer les feux invalides
            if fire['lat'] is not None and fire['lng'] is not None:
                fires.append(fire)
        
        return fires
    
    except requests.exceptions.RequestException as e:
        print(f"Erreur lors de la récupération des données FIRMS: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Status code: {e.response.status_code}")
            print(f"Response: {e.response.text[:500]}")
        return []
    except Exception as e:
        print(f"Erreur lors du parsing des données FIRMS: {e}")
        import traceback
        traceback.print_exc()
        return []


def get_firms_fires_by_bbox(
    min_lat: float,
    min_lng: float,
    max_lat: float,
    max_lng: float,
    days: int = 1
) -> List[Dict]:
    """
    Récupère les feux dans une bounding box spécifique.
    
    Args:
        min_lat: Latitude minimale
        min_lng: Longitude minimale
        max_lat: Latitude maximale
        max_lng: Longitude maximale
        days: Nombre de jours de données
    
    Returns:
        Liste des feux dans la zone
    """
    # Format API: [west, south, east, north] = [min_lng, min_lat, max_lng, max_lat]
    return get_firms_fires(bbox=(min_lng, min_lat, max_lng, max_lat), days=days, source="VIIRS_NOAA20_NRT")


def get_firms_fires_by_country(country_code: str, days: int = 1) -> List[Dict]:
    """
    Récupère les feux pour un pays spécifique.
    
    Args:
        country_code: Code pays ISO (ex: "FRA", "USA", "CAN")
        days: Nombre de jours de données
    
    Returns:
        Liste des feux dans le pays
    """
    return get_firms_fires(country=country_code, days=days)

