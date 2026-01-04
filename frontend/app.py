"""
Interface style NASA FIRMS avec prédiction de propagation
Panneau de contrôle à droite, carte à gauche
"""
import streamlit as st
import requests
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap, MarkerCluster, Fullscreen
import json
import os
from datetime import datetime, timedelta

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Page config
st.set_page_config(
    page_title="Fire Spread Prediction - FIRMS Style",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS - Style NASA FIRMS
st.markdown("""
    <style>
    .main {
        padding: 0;
    }
    .stButton>button {
        border-radius: 4px;
        font-weight: 500;
    }
    .control-panel {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    .section-header {
        font-weight: 600;
        font-size: 14px;
        color: #333;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .time-button {
        background-color: #e9ecef;
        border: 1px solid #dee2e6;
        padding: 8px 16px;
        border-radius: 4px;
        cursor: pointer;
    }
    .time-button.active {
        background-color: #0d6efd;
        color: white;
        border-color: #0d6efd;
    }
    .layer-item {
        display: flex;
        align-items: center;
        padding: 8px;
        margin: 4px 0;
        background-color: white;
        border-radius: 4px;
        border: 1px solid #dee2e6;
    }
    .alert-banner {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        padding: 12px;
        border-radius: 4px;
        margin: 10px 0;
        font-weight: 500;
        color: #856404;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'active_fires' not in st.session_state:
    st.session_state.active_fires = []
if 'selected_fire' not in st.session_state:
    st.session_state.selected_fire = None
if 'map_center' not in st.session_state:
    st.session_state.map_center = [35.0, -80.0]  # Carolinas
if 'map_zoom' not in st.session_state:
    st.session_state.map_zoom = 6
if 'time_range' not in st.session_state:
    st.session_state.time_range = "7DAYS"
if 'display_mode' not in st.session_state:
    st.session_state.display_mode = "simple"
if 'show_predictions' not in st.session_state:
    st.session_state.show_predictions = True
if 'active_layers' not in st.session_state:
    st.session_state.active_layers = {
        "viirs": True,
        "modis": True,
        "landsat": False
    }

# Load fires from NASA FIRMS
def load_nasa_fires(country=None, days=1, bbox=None):
    try:
        params = {"source": "nasa", "days": days}
        if country:
            params["country"] = country
        elif bbox:
            params["bbox_min_lat"] = bbox[0]
            params["bbox_min_lng"] = bbox[1]
            params["bbox_max_lat"] = bbox[2]
            params["bbox_max_lng"] = bbox[3]
        
        response = requests.get(f"{API_URL}/fires", params=params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            fires = data.get("fires", [])
            print(f"DEBUG: {len(fires)} feux chargés depuis l'API")
            if fires:
                print(f"DEBUG: Premier feu: {fires[0]}")
            return fires
        else:
            st.error(f"Erreur API: {response.status_code} - {response.text[:200]}")
            return []
    except Exception as e:
        st.error(f"Erreur: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return []

# Header
col_logo, col_title = st.columns([1, 4])
with col_logo:
    st.markdown("### 🔥")
with col_title:
    st.title("Fire Spread Prediction System")
    st.caption("Enhanced with AI Spread Prediction")

# Main layout: Map left, Control panel right
col_map, col_controls = st.columns([2.5, 1])

with col_controls:
    st.markdown("### 📊 Control Panel")
    
    # Forcer le chargement depuis le CSV local (pas d'options d'affichage)
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[1]
    CSV_PATH = os.getenv("NASA_FIRMS_CSV", str(repo_root / "data" / "MODIS_C6_1_Global_24h.csv"))
    days = 1
    MAX_FIRES = 200

    def load_csv_fires(csv_path, days=1, country=None, bbox=None, max_fires=200):
        try:
            params = {"source": "csv", "csv_path": csv_path, "days": days, "max_fires": max_fires}
            if country:
                params["country"] = country
            if bbox:
                params["bbox_min_lat"] = bbox[0]
                params["bbox_min_lng"] = bbox[1]
                params["bbox_max_lat"] = bbox[2]
                params["bbox_max_lng"] = bbox[3]
            response = requests.get(f"{API_URL}/fires", params=params, timeout=120)
            if response.status_code == 200:
                data = response.json()
                return data.get("fires", [])
            else:
                st.error(f"Erreur API: {response.status_code} - {response.text[:200]}")
                return []
        except Exception as e:
            st.error(f"Erreur: {str(e)}")
            return []

    # Auto-load CSV fires on first visit
    if len(st.session_state.active_fires) == 0:
        st.info(f"Chargement depuis: {CSV_PATH}")
        with st.spinner("Chargement des feux depuis le fichier CSV local..."):
            fires = load_csv_fires(CSV_PATH, days=days, max_fires=MAX_FIRES)
            st.session_state.active_fires = fires
            if len(fires) > 0:
                st.success(f"✅ {len(fires)} feux chargés depuis CSV")
    
    # Statistics
    st.divider()
    st.markdown("### 📈 Statistiques")
    if st.session_state.active_fires:
        total = len(st.session_state.active_fires)
        st.metric("Total Feux", total)
        fires_with_pred = sum(1 for f in st.session_state.active_fires if f.get('prediction'))
        st.metric("Avec Prédictions", fires_with_pred)
        if total > 0:
            avg_brightness = sum(f.get('brightness', 0) for f in st.session_state.active_fires) / total
            st.metric("Brightness Moyen", f"{avg_brightness:.0f}")
    else:
        st.info("👆 Cliquez sur 'Charger les Feux' pour commencer")

with col_map:
    st.markdown("### 🗺️ Fire Map")
    
    # Create map
    m = folium.Map(
        location=st.session_state.map_center,
        zoom_start=st.session_state.map_zoom,
        tiles='OpenStreetMap'
    )
    
    # Add tile layers
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satellite',
        overlay=False,
        control=True
    ).add_to(m)
    
    folium.TileLayer(
        tiles='https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
        attr='OpenStreetMap',
        name='Carte avec Villes',
        overlay=False,
        control=True
    ).add_to(m)
    
    # Add controls
    Fullscreen().add_to(m)
    marker_cluster = MarkerCluster().add_to(m)
    
    # Add fires to map
    if st.session_state.active_fires:
        
        for fire in st.session_state.active_fires:
            fire_lat = fire['lat']
            fire_lng = fire['lng']
            brightness = fire.get('brightness', 350)
            acq_date = fire.get('acq_date', '')
            satellite = fire.get('satellite', '')
            
            # Afficher tous les feux (pas de filtrage par couche pour simplifier)
            
            # Marker color based on brightness
            if brightness > 500:
                icon_color = 'red'
            elif brightness > 350:
                icon_color = 'orange'
            else:
                icon_color = 'yellow'
            
            # Popup content
            popup_html = f"""
            <div style="font-family: Arial; width: 280px;">
                <h4 style="color: {icon_color}; margin: 5px 0;">🔥 Fire Detection</h4>
                <table style="width: 100%; font-size: 12px;">
                    <tr><td><b>Location:</b></td><td>{fire_lat:.4f}, {fire_lng:.4f}</td></tr>
                    <tr><td><b>Brightness:</b></td><td>{brightness:.0f}</td></tr>
                    <tr><td><b>Date:</b></td><td>{acq_date} {fire.get('acq_time', '')}</td></tr>
                    <tr><td><b>Satellite:</b></td><td>{satellite}</td></tr>
                    <tr><td><b>Instrument:</b></td><td>{fire.get('instrument', 'N/A')}</td></tr>
                    {f'<tr><td><b>FRP:</b></td><td>{fire.get("frp", "N/A")} MW</td></tr>' if fire.get('frp') else ''}
            """
            
            # Add prediction info
            if fire.get('prediction') and st.session_state.show_predictions:
                pred = fire['prediction']
                popup_html += f"""
                    <tr><td colspan="2"><hr style="margin: 8px 0;"></td></tr>
                    <tr><td colspan="2"><b style="color: #0d6efd;">🤖 AI Prediction:</b></td></tr>
                    <tr><td><b>Will Spread:</b></td><td>{"✅ YES" if pred['will_spread'] else "❌ NO"}</td></tr>
                    <tr><td><b>Probability:</b></td><td>{pred['spread_probability']:.1%}</td></tr>
                    <tr><td><b>Distance:</b></td><td>{pred['spread_distance_km']:.2f} km</td></tr>
                    <tr><td><b>Direction:</b></td><td>{pred['spread_direction']:.0f}°</td></tr>
                """
            
            popup_html += "</table></div>"
            
            # Add marker
            marker = folium.Marker(
                [fire_lat, fire_lng],
                popup=folium.Popup(popup_html, max_width=300),
                icon=folium.Icon(color=icon_color, icon="fire", prefix="fa"),
                tooltip=f"Fire: {brightness:.0f}"
            )
            marker.add_to(marker_cluster)
            
            # Add spread prediction visualization
            if fire.get('prediction') and st.session_state.show_predictions:
                pred = fire['prediction']
                geojson = pred.get('geojson', {})
                
                if geojson and 'features' in geojson:
                    for feature in geojson['features']:
                        feature_type = feature['properties'].get('type')
                        
                        if feature_type == 'spread':
                            coords = feature['geometry']['coordinates'][0]
                            folium.Polygon(
                                locations=[(lat_coord, lng_coord) for lng_coord, lat_coord in coords],
                                color='red',
                                fill=True,
                                fillColor='red',
                                fillOpacity=0.15,
                                weight=2,
                                popup=f"Predicted Spread<br>Probability: {pred['spread_probability']:.1%}"
                            ).add_to(m)
                        
                        elif feature_type == 'direction':
                            coords = feature['geometry']['coordinates']
                            folium.PolyLine(
                                locations=[(lat_coord, lng_coord) for lng_coord, lat_coord in coords],
                                color='darkred',
                                weight=3,
                                opacity=0.6
                            ).add_to(m)
            
            
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Display map
    map_data = st_folium(
        m,
        width=None,
        height=700,
        key="fire_map"
    )

    # Handle marker clicks: call backend /predict_fire for clicked location
    clicked = None
    if isinstance(map_data, dict):
        clicked = map_data.get('last_clicked') or map_data.get('last_object_clicked')

    if clicked:
        # Extract lat/lng from click
        click_lat = None
        click_lng = None
        if isinstance(clicked, dict):
            click_lat = clicked.get('lat') or clicked.get('latitude')
            click_lng = clicked.get('lng') or clicked.get('lon') or clicked.get('longitude')

        if click_lat is not None and click_lng is not None:
            prev = st.session_state.get('last_click_pos')
            if prev is None or (prev[0] != click_lat or prev[1] != click_lng):
                st.session_state['last_click_pos'] = (click_lat, click_lng)
                # Find nearest fire to get brightness if available
                nearest = None
                min_dist = None
                for f in st.session_state.active_fires:
                    try:
                        dlat = f['lat'] - float(click_lat)
                        dlng = f['lng'] - float(click_lng)
                        dist2 = dlat * dlat + dlng * dlng
                        if min_dist is None or dist2 < min_dist:
                            min_dist = dist2
                            nearest = f
                    except Exception:
                        continue

                brightness = nearest.get('brightness', 350) if nearest else 350

                # Call backend predict_fire
                try:
                    resp = requests.get(f"{API_URL}/predict_fire", params={
                        'lat': float(click_lat), 'lng': float(click_lng), 'brightness': float(brightness)
                    }, timeout=30)
                    if resp.status_code == 200:
                        data = resp.json()
                        st.session_state['last_prediction'] = data.get('prediction')
                    else:
                        st.session_state['last_prediction'] = {'error': f"API {resp.status_code}"}
                except Exception as e:
                    st.session_state['last_prediction'] = {'error': str(e)}

    # Show last prediction in control panel
    if 'last_prediction' in st.session_state and st.session_state['last_prediction']:
        with st.expander('🔎 Dernière prédiction (clic)', expanded=True):
            pred = st.session_state['last_prediction']
            if pred.get('error'):
                st.error(pred.get('error'))
            else:
                st.write('Probabilité:', f"{pred['spread_probability']:.1%}")
                st.write('Distance (km):', f"{pred['spread_distance_km']:.2f}")
                st.write('Direction (°):', pred['spread_direction'])
                st.json(pred.get('environmental_data', {}))

# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: #666; font-size: 12px;'>
        <p>🔥 Fire Spread Prediction System | Enhanced with AI | Data: NASA FIRMS</p>
    </div>
""", unsafe_allow_html=True)

