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
@st.cache_data(ttl=300)
def load_nasa_fires(country=None, days=1, source="VIIRS_NOAA20_NRT"):
    try:
        params = {"source": "nasa", "days": days}
        if country:
            params["country"] = country
        response = requests.get(f"{API_URL}/fires", params=params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            return data.get("fires", [])
    except Exception as e:
        st.error(f"Erreur: {str(e)}")
    return []

# Header
col_logo, col_title, col_mode = st.columns([1, 3, 1])
with col_logo:
    st.markdown("### 🔥")
with col_title:
    st.title("Fire Spread Prediction System")
    st.caption("Enhanced with AI Spread Prediction")
with col_mode:
    mode = st.selectbox("Mode", ["BASIC MODE", "ADVANCED MODE"], label_visibility="collapsed")
    st.markdown(f"**{mode}**")

# Main layout: Map left, Control panel right
col_map, col_controls = st.columns([2.5, 1])

with col_controls:
    st.markdown("### 📊 Control Panel")
    
    # Time Range Selection
    st.markdown('<div class="control-panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">📅 Time Range</div>', unsafe_allow_html=True)
    
    col_today, col_24h, col_7d = st.columns(3)
    with col_today:
        today_active = st.session_state.time_range == "TODAY"
        if st.button("TODAY", use_container_width=True, disabled=today_active):
            st.session_state.time_range = "TODAY"
            st.rerun()
    with col_24h:
        h24_active = st.session_state.time_range == "24HRS"
        if st.button("24HRS", use_container_width=True, disabled=h24_active):
            st.session_state.time_range = "24HRS"
            st.rerun()
    with col_7d:
        d7_active = st.session_state.time_range == "7DAYS"
        if st.button("7DAYS", use_container_width=True, disabled=d7_active):
            st.session_state.time_range = "7DAYS"
            st.rerun()
    
    # Calculate days based on selection
    days_map = {"TODAY": 1, "24HRS": 1, "7DAYS": 7}
    days = days_map.get(st.session_state.time_range, 7)
    
    st.markdown(f"**Selected:** {datetime.now().strftime('%b %d %Y')} - {st.session_state.time_range}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Fires / Hotspots Section
    with st.expander("🔥 Fires / Hotspots", expanded=True):
        # Display Mode
        col_simple, col_time = st.columns(2)
        with col_simple:
            simple_active = st.session_state.display_mode == "simple"
            if st.button("Simple", use_container_width=True, type="primary" if simple_active else "secondary"):
                st.session_state.display_mode = "simple"
                st.rerun()
        with col_time:
            time_active = st.session_state.display_mode == "time"
            if st.button("Time Based", use_container_width=True, type="primary" if time_active else "secondary"):
                st.session_state.display_mode = "time"
                st.rerun()
        
        st.divider()
        
        # Data Layers
        st.markdown("**Data Layers:**")
        
        # VIIRS
        col_viirs1, col_viirs2 = st.columns([3, 1])
        with col_viirs1:
            st.session_state.active_layers["viirs"] = st.checkbox(
                "VIIRS (S-NPP, NOAA-20 & NOAA-21) [375m]",
                value=st.session_state.active_layers["viirs"],
                key="layer_viirs"
            )
        with col_viirs2:
            st.info("ℹ️", help="VIIRS provides high-resolution fire detection")
        
        # MODIS
        col_modis1, col_modis2 = st.columns([3, 1])
        with col_modis1:
            st.session_state.active_layers["modis"] = st.checkbox(
                "MODIS (Aqua & Terra) [1km]",
                value=st.session_state.active_layers["modis"],
                key="layer_modis"
            )
        with col_modis2:
            st.info("ℹ️", help="MODIS provides broad coverage fire detection")
        
        # Landsat
        col_landsat1, col_landsat2 = st.columns([3, 1])
        with col_landsat1:
            st.session_state.active_layers["landsat"] = st.checkbox(
                "Landsat [30m]",
                value=st.session_state.active_layers["landsat"],
                key="layer_landsat"
            )
        with col_landsat2:
            st.info("ℹ️", help="Landsat provides very high resolution")
        
        st.divider()
        
        # Prediction Options
        st.markdown("**AI Prediction:**")
        st.session_state.show_predictions = st.checkbox(
            "Show Spread Predictions",
            value=st.session_state.show_predictions,
            help="Display AI-predicted fire spread zones"
        )
        
        show_heatmap = st.checkbox(
            "Show Heatmap",
            value=False,
            help="Display heatmap of spread probability"
        )
    
    # Active Alerts Section
    with st.expander("⚠️ Active Alerts", expanded=True):
        st.markdown("""
            <div class="alert-banner">
                ⚠️ FIRES NOT DECLARED CONTAINED, CONTROLLED, NOR OUT.
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**Monitoring:**")
        
        # US Active Fires
        col_us1, col_us2 = st.columns([3, 1])
        with col_us1:
            st.checkbox("US Active Fires - IMSR", value=True, key="alert_us")
        with col_us2:
            st.info("ℹ️", help="US Interagency Fire Management")
        
        # Canada Active Fires
        col_can1, col_can2 = st.columns([3, 1])
        with col_can1:
            st.checkbox("Canada Active Fires - DIP", value=True, key="alert_can")
        with col_can2:
            st.info("ℹ️", help="Canadian Fire Information")
        
        # Fire Perimeter
        col_perim1, col_perim2 = st.columns([3, 1])
        with col_perim1:
            st.checkbox("US Fire Perimeter", value=True, key="alert_perim")
        with col_perim2:
            st.info("ℹ️", help="Fire perimeter boundaries")
    
    # Load Data Button
    st.divider()
    if st.button("🔄 Load Fire Data", type="primary", use_container_width=True):
        with st.spinner("Loading fire data..."):
            st.session_state.active_fires = load_nasa_fires(days=days)
            if len(st.session_state.active_fires) > 0:
                st.success(f"✅ {len(st.session_state.active_fires)} fires loaded")
            else:
                st.warning("⚠️ No fires found")
            st.rerun()
    
    # Statistics
    if st.session_state.active_fires:
        st.divider()
        st.markdown("### 📈 Statistics")
        st.metric("Total Fires", len(st.session_state.active_fires))
        fires_with_pred = sum(1 for f in st.session_state.active_fires if f.get('prediction'))
        st.metric("With Predictions", fires_with_pred)
        avg_brightness = sum(f.get('brightness', 0) for f in st.session_state.active_fires) / len(st.session_state.active_fires)
        st.metric("Avg Brightness", f"{avg_brightness:.0f}")

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
        heatmap_points = []
        
        for fire in st.session_state.active_fires:
            fire_lat = fire['lat']
            fire_lng = fire['lng']
            brightness = fire.get('brightness', 350)
            acq_date = fire.get('acq_date', '')
            satellite = fire.get('satellite', '')
            
            # Filter by active layers
            if satellite.startswith('N') and not st.session_state.active_layers["viirs"]:
                continue
            if satellite in ['Aqua', 'Terra'] and not st.session_state.active_layers["modis"]:
                continue
            
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
                    <tr><td><b>Date:</b></td><td>{acq_date}</td></tr>
                    <tr><td><b>Satellite:</b></td><td>{satellite}</td></tr>
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
            
            # Add to heatmap
            if show_heatmap and fire.get('prediction'):
                pred = fire['prediction']
                heatmap_points.append([
                    fire_lat,
                    fire_lng,
                    pred['spread_probability']
                ])
        
        # Add heatmap
        if show_heatmap and heatmap_points:
            HeatMap(
                heatmap_points,
                min_opacity=0.2,
                max_zoom=18,
                radius=20,
                blur=15,
                gradient={
                    0.2: 'blue',
                    0.4: 'cyan',
                    0.6: 'lime',
                    0.7: 'yellow',
                    1: 'red'
                }
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

# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: #666; font-size: 12px;'>
        <p>🔥 Fire Spread Prediction System | Enhanced with AI | Data: NASA FIRMS</p>
    </div>
""", unsafe_allow_html=True)


