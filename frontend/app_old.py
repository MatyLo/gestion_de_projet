"""
Streamlit Frontend for Fire Spread Prediction
A minimalistic interface for wildfire spread prediction
"""
import streamlit as st
import requests
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
import json
import os

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Page config
st.set_page_config(
    page_title="Fire Spread Prediction",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🔥 Système de Prédiction de Propagation des Feux")
st.markdown("**Feux en temps réel depuis NASA FIRMS** | Prédiction de propagation basée sur les données environnementales")

# Initialize session state
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'selected_lat' not in st.session_state:
    st.session_state.selected_lat = 34.05
if 'selected_lng' not in st.session_state:
    st.session_state.selected_lng = -118.25
if 'active_fires' not in st.session_state:
    st.session_state.active_fires = []
if 'heatmap_data' not in st.session_state:
    st.session_state.heatmap_data = None
if 'show_heatmap' not in st.session_state:
    st.session_state.show_heatmap = False

# Load active fires from NASA FIRMS
@st.cache_data(ttl=300)  # Cache for 5 minutes (NASA data updates every few hours)
def load_nasa_fires(country=None, days=1):
    try:
        params = {"source": "nasa", "days": days}
        if country:
            params["country"] = country
        response = requests.get(f"{API_URL}/fires", params=params, timeout=30)
        if response.status_code == 200:
            return response.json()["fires"]
    except Exception as e:
        st.error(f"Erreur lors du chargement des feux NASA: {str(e)}")
    return []

# Load manual fires
def load_manual_fires():
    try:
        response = requests.get(f"{API_URL}/fires", params={"source": "manual"}, timeout=5)
        if response.status_code == 200:
            return response.json()["fires"]
    except:
        pass
    return []

# Sidebar for inputs
with st.sidebar:
    st.header("🔥 Gestion des Feux")
    
    # Tab selection
    tab1, tab2 = st.tabs(["📍 Nouveau Feu", "📋 Feux Actifs"])
    
    with tab1:
        st.subheader("Ajouter un Feu")
        
        # Method selection
        input_method = st.radio(
            "Méthode de sélection:",
            ["Saisie Manuelle", "Clic sur Carte"],
            help="Choisissez comment spécifier l'emplacement du feu"
        )
        
        if input_method == "Saisie Manuelle":
            lat = st.number_input(
                "Latitude",
                min_value=-90.0,
                max_value=90.0,
                value=st.session_state.selected_lat,
                step=0.01,
                format="%.4f"
            )
            lng = st.number_input(
                "Longitude",
                min_value=-180.0,
                max_value=180.0,
                value=st.session_state.selected_lng,
                step=0.01,
                format="%.4f"
            )
            st.session_state.selected_lat = lat
            st.session_state.selected_lng = lng
        else:
            st.info("👆 Cliquez sur la carte pour sélectionner un emplacement")
            lat = st.session_state.selected_lat
            lng = st.session_state.selected_lng
            st.write(f"**Sélectionné:** ({lat:.4f}, {lng:.4f})")
        
        # Fire parameters
        fire_name = st.text_input("Nom du feu (optionnel)", value="")
        brightness = st.slider(
            "Intensité/Brightness",
            min_value=100,
            max_value=800,
            value=350,
            step=10,
            help="Intensité plus élevée = feu plus intense"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            add_fire_button = st.button("➕ Ajouter Feu", type="primary", use_container_width=True)
        with col2:
            predict_button = st.button("🔮 Prédire", use_container_width=True)
        
        # Handle add fire button (only for manual fires)
        if add_fire_button:
            with st.spinner("Ajout du feu..."):
                try:
                    response = requests.post(
                        f"{API_URL}/fires",
                        json={
                            "lat": st.session_state.selected_lat,
                            "lng": st.session_state.selected_lng,
                            "brightness": brightness,
                            "name": fire_name if fire_name else None
                        },
                        timeout=10
                    )
                    if response.status_code == 200:
                        st.success("✅ Feu manuel ajouté avec succès!")
                        st.session_state.active_fires = load_manual_fires()
                        st.rerun()
                    else:
                        st.error(f"❌ Erreur: {response.status_code}")
                except Exception as e:
                    st.error(f"❌ Erreur: {str(e)}")
        
        # Handle predict button
        if predict_button:
            with st.spinner("🔄 Analyse de la propagation..."):
                try:
                    response = requests.post(
                        f"{API_URL}/predict",
                        json={
                            "lat": st.session_state.selected_lat,
                            "lng": st.session_state.selected_lng,
                            "brightness": brightness
                        },
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        st.session_state.prediction_result = response.json()
                        st.success("✅ Prédiction terminée!")
                        st.rerun()
                    else:
                        st.error(f"❌ Erreur: {response.status_code}")
                        st.json(response.json())
                except requests.exceptions.ConnectionError:
                    st.error("❌ Impossible de se connecter à l'API backend. Assurez-vous qu'il fonctionne sur http://localhost:8000")
                except Exception as e:
                    st.error(f"❌ Erreur: {str(e)}")
        
        st.divider()
        
        # Heatmap toggle
        st.subheader("Carte de Chaleur")
        show_heatmap = st.checkbox("Afficher la carte de chaleur de propagation", value=st.session_state.show_heatmap)
        st.session_state.show_heatmap = show_heatmap
        
        if show_heatmap and st.button("🗺️ Générer Heatmap", use_container_width=True):
            with st.spinner("Génération de la carte de chaleur..."):
                try:
                    response = requests.post(
                        f"{API_URL}/heatmap",
                        json={
                            "lat": st.session_state.selected_lat,
                            "lng": st.session_state.selected_lng,
                            "brightness": brightness
                        },
                        timeout=15
                    )
                    if response.status_code == 200:
                        st.session_state.heatmap_data = response.json()["heatmap"]
                        st.success(f"✅ Heatmap générée ({len(st.session_state.heatmap_data)} points)")
                    else:
                        st.error(f"❌ Erreur: {response.status_code}")
                except Exception as e:
                    st.error(f"❌ Erreur: {str(e)}")
    
    with tab2:
        st.subheader("Feux Actifs")
        
        # Source selection
        fire_source = st.radio(
            "Source des feux:",
            ["NASA FIRMS (Temps réel)", "Feux manuels"],
            help="NASA FIRMS affiche les feux détectés par satellites en temps réel"
        )
        
        # Parameters for NASA fires
        if fire_source == "NASA FIRMS (Temps réel)":
            col1, col2 = st.columns(2)
            with col1:
                country_code = st.text_input(
                    "Code pays (optionnel)",
                    value="",
                    help="Code ISO (ex: FRA, USA, CAN). Laissez vide pour le monde entier",
                    placeholder="FRA"
                )
            with col2:
                days = st.number_input("Jours de données", min_value=1, max_value=10, value=1)
            
            if st.button("🔄 Charger feux NASA FIRMS", type="primary", use_container_width=True):
                with st.spinner("Chargement des feux depuis NASA FIRMS..."):
                    country = country_code.upper() if country_code else None
                    st.session_state.active_fires = load_nasa_fires(country=country, days=days)
                    st.success(f"✅ {len(st.session_state.active_fires)} feux chargés")
                    st.rerun()
        else:
            if st.button("🔄 Actualiser feux manuels", use_container_width=True):
                st.session_state.active_fires = load_manual_fires()
                st.rerun()
        
        # Auto-load NASA fires on first load
        if len(st.session_state.active_fires) == 0 and fire_source == "NASA FIRMS (Temps réel)":
            with st.spinner("Chargement initial des feux NASA..."):
                st.session_state.active_fires = load_nasa_fires(days=1)
        
        if st.session_state.active_fires:
            st.write(f"**{len(st.session_state.active_fires)} feux trouvés**")
            for fire in st.session_state.active_fires[:20]:  # Limiter à 20 pour la performance
                fire_name = fire.get('name', f"Feu #{fire['id']}")
                with st.expander(f"🔥 {fire_name}"):
                    st.write(f"**Position:** ({fire['lat']:.4f}, {fire['lng']:.4f})")
                    st.write(f"**Intensité (Brightness):** {fire['brightness']:.0f}")
                    
                    if fire.get('source') == 'nasa_firms':
                        if fire.get('confidence'):
                            st.write(f"**Confiance NASA:** {fire['confidence']}")
                        if fire.get('satellite'):
                            st.write(f"**Satellite:** {fire['satellite']}")
                        if fire.get('timestamp'):
                            st.write(f"**Détecté le:** {fire['timestamp']}")
                        if fire.get('frp'):
                            st.write(f"**Puissance radiative (FRP):** {fire['frp']:.2f} MW")
                    
                    if fire.get('prediction'):
                        pred = fire['prediction']
                        st.write(f"**Probabilité de propagation:** {pred['spread_probability']:.1%}")
                        st.write(f"**Distance estimée:** {pred['spread_distance_km']:.2f} km")
                        st.write(f"**Direction:** {pred['spread_direction']:.0f}°")
                    
                    # Bouton supprimer seulement pour les feux manuels
                    if fire.get('source') != 'nasa_firms':
                        if st.button(f"🗑️ Supprimer", key=f"delete_{fire['id']}"):
                            try:
                                response = requests.delete(f"{API_URL}/fires/{fire['id']}", timeout=5)
                                if response.status_code == 200:
                                    st.success("Feu supprimé")
                                    st.rerun()
                            except Exception as e:
                                st.error(f"Erreur: {str(e)}")
            
            if len(st.session_state.active_fires) > 20:
                st.info(f"⚠️ Affichage des 20 premiers feux sur {len(st.session_state.active_fires)}")
        else:
            if fire_source == "NASA FIRMS (Temps réel)":
                st.info("Aucun feu détecté par NASA FIRMS. Cliquez sur 'Charger feux NASA FIRMS' pour actualiser.")
            else:
                st.info("Aucun feu manuel. Ajoutez-en un dans l'onglet 'Nouveau Feu'")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("🗺️ Carte Interactive - Feux et Propagation")
    
    # Determine map center (use first active fire or selected location)
    if st.session_state.active_fires:
        center_lat = st.session_state.active_fires[0]['lat']
        center_lng = st.session_state.active_fires[0]['lng']
    else:
        center_lat = st.session_state.selected_lat
        center_lng = st.session_state.selected_lng
    
    # Create base map
    m = folium.Map(
        location=[center_lat, center_lng],
        zoom_start=10,
        tiles="OpenStreetMap"
    )
    
    # Add heatmap if enabled and data exists
    if st.session_state.show_heatmap and st.session_state.heatmap_data:
        heatmap_points = [[point[0], point[1], point[2]] for point in st.session_state.heatmap_data]
        HeatMap(
            heatmap_points,
            min_opacity=0.2,
            max_zoom=18,
            radius=15,
            blur=15,
            gradient={
                0.2: 'blue',
                0.4: 'cyan',
                0.6: 'lime',
                0.7: 'yellow',
                1: 'red'
            }
        ).add_to(m)
        st.info("🔥 Carte de chaleur activée - Zones rouges = propagation probable")
    
    # Add all active fires to the map
    for fire in st.session_state.active_fires:
        fire_lat = fire['lat']
        fire_lng = fire['lng']
        fire_name = fire.get('name', f"Feu #{fire['id']}")
        
        # Get fire prediction if available
        if fire.get('prediction'):
            pred = fire['prediction']
            popup_html = f"""
            <b>{fire_name}</b><br>
            Position: ({fire_lat:.4f}, {fire_lng:.4f})<br>
            Intensité: {fire['brightness']}<br>
            Probabilité: {pred['spread_probability']:.1%}<br>
            Distance: {pred['spread_distance_km']:.2f} km
            """
        else:
            popup_html = f"""
            <b>{fire_name}</b><br>
            Position: ({fire_lat:.4f}, {fire_lng:.4f})<br>
            Intensité: {fire['brightness']}
            """
        
        folium.Marker(
            [fire_lat, fire_lng],
            popup=folium.Popup(popup_html, max_width=200),
            icon=folium.Icon(color="red", icon="fire", prefix="fa"),
            tooltip=fire_name
        ).add_to(m)
        
        # Add spread visualization for active fires
        if fire.get('prediction'):
            pred = fire['prediction']
            geojson = pred.get('geojson', {})
            
            if geojson and 'features' in geojson:
                for feature in geojson['features']:
                    feature_type = feature['properties'].get('type')
                    
                    if feature_type == 'spread':
                        coords = feature['geometry']['coordinates'][0]
                        folium.Polygon(
                            locations=[(coord_lat, coord_lng) for coord_lng, coord_lat in coords],
                            color='orange',
                            fill=True,
                            fillColor='orange',
                            fillOpacity=0.2,
                            popup=f"Zone de propagation: {pred['spread_probability']:.1%}"
                        ).add_to(m)
                    
                    elif feature_type == 'direction':
                        coords = feature['geometry']['coordinates']
                        folium.PolyLine(
                            locations=[(coord_lat, coord_lng) for coord_lng, coord_lat in coords],
                            color='red',
                            weight=2,
                            opacity=0.6,
                            popup=f"Direction: {pred['spread_direction']:.0f}°"
                        ).add_to(m)
    
    # Add marker for selected location (if not already a fire)
    is_selected_a_fire = any(
        abs(f['lat'] - st.session_state.selected_lat) < 0.001 and 
        abs(f['lng'] - st.session_state.selected_lng) < 0.001
        for f in st.session_state.active_fires
    )
    
    if not is_selected_a_fire:
        folium.Marker(
            [st.session_state.selected_lat, st.session_state.selected_lng],
            popup=f"Emplacement sélectionné<br>Lat: {st.session_state.selected_lat:.4f}<br>Lng: {st.session_state.selected_lng:.4f}",
            icon=folium.Icon(color="blue", icon="map-marker", prefix="fa"),
            tooltip="Emplacement sélectionné"
        ).add_to(m)
    
    # If we have prediction results for selected location, add them
    if st.session_state.prediction_result:
        result = st.session_state.prediction_result
        geojson = result.get('geojson', {})
        
        if geojson and 'features' in geojson:
            for feature in geojson['features']:
                feature_type = feature['properties'].get('type')
                
                if feature_type == 'spread':
                    coords = feature['geometry']['coordinates'][0]
                    folium.Polygon(
                        locations=[(coord_lat, coord_lng) for coord_lng, coord_lat in coords],
                        color='darkorange',
                        fill=True,
                        fillColor='darkorange',
                        fillOpacity=0.4,
                        weight=3,
                        popup=f"Prédiction - Probabilité: {result['spread_probability']:.2%}"
                    ).add_to(m)
                
                elif feature_type == 'direction':
                    coords = feature['geometry']['coordinates']
                    folium.PolyLine(
                        locations=[(coord_lat, coord_lng) for coord_lng, coord_lat in coords],
                        color='darkred',
                        weight=4,
                        opacity=0.9,
                        popup=f"Direction du vent: {result['spread_direction']:.0f}°"
                    ).add_to(m)
    
    # Display map and capture clicks
    map_data = st_folium(
        m,
        width=None,
        height=600,
        key="map"
    )
    
    # Update selected location if map was clicked
    if map_data and map_data.get('last_clicked'):
        clicked = map_data['last_clicked']
        if clicked:
            st.session_state.selected_lat = clicked['lat']
            st.session_state.selected_lng = clicked['lng']
            # Note: Rerun is handled by button clicks in sidebar

with col2:
    st.header("📊 Résultats de Prédiction")
    
    # Display results
    if st.session_state.prediction_result:
        result = st.session_state.prediction_result
        
        # Main prediction metrics
        st.metric(
            "Will Spread?",
            "YES" if result['will_spread'] else "NO",
            delta=None
        )
        
        st.metric(
            "Spread Probability",
            f"{result['spread_probability']:.1%}"
        )
        
        st.metric(
            "Spread Distance",
            f"{result['spread_distance_km']:.2f} km"
        )
        
        st.metric(
            "Spread Direction",
            f"{result['spread_direction']:.0f}°"
        )
        
        # Environmental data
        with st.expander("🌍 Environmental Data", expanded=True):
            env = result['environmental_data']
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.write(f"**Temp:** {env['temperature']:.1f}°C")
                st.write(f"**Humidity:** {env['humidity']:.1f}%")
                st.write(f"**Wind:** {env['wind_speed']:.1f} km/h")
                st.write(f"**Elevation:** {env['elevation']:.0f} m")
            
            with col_b:
                st.write(f"**Drought:** {env['drought']:.2f}")
                st.write(f"**Vegetation:** {env['vegetation']:.2f}")
                st.write(f"**Brightness:** {env['brightness']:.0f}")
                st.write(f"**Source:** {env['data_source']}")
        
        # Download button for full results
        st.download_button(
            label="📥 Download Full Results (JSON)",
            data=json.dumps(result, indent=2),
            file_name="fire_prediction.json",
            mime="application/json"
        )
    else:
        st.info("👈 Set location and click 'Predict Fire Spread' to see results")

# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🔥 Système de Prédiction de Propagation des Feux | Propulsé par Machine Learning</p>
        <p><small>Données de feux: <a href="https://firms.modaps.eosdis.nasa.gov/" target="_blank">NASA FIRMS</a> | 
        Météo et élévation: <a href="https://open-meteo.com/" target="_blank">Open-Meteo API</a></small></p>
    </div>
""", unsafe_allow_html=True)

