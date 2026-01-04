"""
NASA FIRMS Style Interface with Fire Spread Prediction
Interface similaire à NASA FIRMS avec prédiction de propagation
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

# Page config - Style NASA FIRMS
st.set_page_config(
    page_title="Fire Spread Prediction - NASA FIRMS Style",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Style NASA FIRMS
st.markdown("""
    <style>
    .main {
        padding: 0.5rem;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        border-radius: 4px;
    }
    .stButton>button:hover {
        background-color: #0d5a8a;
    }
    .fire-marker {
        font-size: 20px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    h1 {
        color: #1f77b4;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'active_fires' not in st.session_state:
    st.session_state.active_fires = []
if 'selected_fire' not in st.session_state:
    st.session_state.selected_fire = None
if 'map_center' not in st.session_state:
    st.session_state.map_center = [34.05, -118.25]
if 'map_zoom' not in st.session_state:
    st.session_state.map_zoom = 6
if 'show_predictions' not in st.session_state:
    st.session_state.show_predictions = True
if 'show_heatmap' not in st.session_state:
    st.session_state.show_heatmap = False

# Load fires from NASA FIRMS
@st.cache_data(ttl=300)
def load_nasa_fires(country=None, days=1, source="MODIS_NRT"):
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

# Header - Style NASA FIRMS
col_header1, col_header2, col_header3 = st.columns([2, 1, 1])
with col_header1:
    st.title("🔥 NASA FIRMS - Fire Spread Prediction")
    st.markdown("**Fire Information for Resource Management System** | Enhanced with AI Spread Prediction")
with col_header2:
    st.metric("Active Fires", len(st.session_state.active_fires))
with col_header3:
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

# Sidebar - Filters (Style NASA FIRMS)
with st.sidebar:
    st.header("🔍 Filters & Controls")
    
    # Data Source
    st.subheader("Data Source")
    data_source = st.selectbox(
        "Satellite Source",
        ["MODIS_NRT", "VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT"],
        index=0,
        help="Select satellite data source"
    )
    
    # Geographic Filter
    st.subheader("Geographic Filter")
    filter_type = st.radio(
        "Filter by:",
        ["Worldwide", "Country", "Bounding Box"],
        index=0
    )
    
    country_code = None
    bbox = None
    
    if filter_type == "Country":
        country_code = st.text_input(
            "Country Code (ISO)",
            value="USA",
            help="ISO country code (e.g., USA, FRA, CAN)"
        )
    elif filter_type == "Bounding Box":
        col1, col2 = st.columns(2)
        with col1:
            min_lat = st.number_input("Min Latitude", value=-90.0, step=1.0)
            min_lng = st.number_input("Min Longitude", value=-180.0, step=1.0)
        with col2:
            max_lat = st.number_input("Max Latitude", value=90.0, step=1.0)
            max_lng = st.number_input("Max Longitude", value=180.0, step=1.0)
        bbox = (min_lng, min_lat, max_lng, max_lat)
    
    # Time Range
    st.subheader("Time Range")
    days = st.slider(
        "Days of Data",
        min_value=1,
        max_value=10,
        value=1,
        help="Number of days of fire data to retrieve"
    )
    
    # Load Button
    if st.button("📥 Load Fire Data", type="primary", use_container_width=True):
        with st.spinner("Loading fire data from NASA FIRMS..."):
            if filter_type == "Country" and country_code:
                st.session_state.active_fires = load_nasa_fires(
                    country=country_code.upper(), 
                    days=days
                )
            elif filter_type == "Bounding Box" and bbox:
                # TODO: Implement bbox loading
                st.session_state.active_fires = load_nasa_fires(days=days)
            else:
                st.session_state.active_fires = load_nasa_fires(days=days)
            st.success(f"✅ Loaded {len(st.session_state.active_fires)} fires")
            st.rerun()
    
    st.divider()
    
    # Display Options
    st.subheader("Display Options")
    st.session_state.show_predictions = st.checkbox(
        "Show Spread Predictions",
        value=True,
        help="Display predicted fire spread zones"
    )
    st.session_state.show_heatmap = st.checkbox(
        "Show Heatmap",
        value=False,
        help="Display heatmap of fire spread probability"
    )
    
    # Fire Details
    if st.session_state.selected_fire:
        st.divider()
        st.subheader("🔥 Selected Fire")
        fire = st.session_state.selected_fire
        st.write(f"**Location:** ({fire['lat']:.4f}, {fire['lng']:.4f})")
        st.write(f"**Brightness:** {fire.get('brightness', 'N/A')}")
        if fire.get('confidence'):
            st.write(f"**Confidence:** {fire['confidence']}")
        if fire.get('acq_date'):
            st.write(f"**Date:** {fire['acq_date']} {fire.get('acq_time', '')}")

# Main Content Area
col_map, col_info = st.columns([3, 1])

with col_map:
    st.subheader("🗺️ Fire Map")
    
    # Create map
    m = folium.Map(
        location=st.session_state.map_center,
        zoom_start=st.session_state.map_zoom,
        tiles="OpenStreetMap"
    )
    
    # Add tile layers (like NASA FIRMS)
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satellite',
        overlay=False,
        control=True
    ).add_to(m)
    
    folium.TileLayer('OpenStreetMap').add_to(m)
    
    # Add fullscreen control
    Fullscreen().add_to(m)
    
    # Add marker cluster for better performance
    marker_cluster = MarkerCluster().add_to(m)
    
    # Add fires to map
    if st.session_state.active_fires:
        for fire in st.session_state.active_fires:
            fire_lat = fire['lat']
            fire_lng = fire['lng']
            brightness = fire.get('brightness', 350)
            confidence = fire.get('confidence', 'N/A')
            acq_date = fire.get('acq_date', '')
            acq_time = fire.get('acq_time', '')
            satellite = fire.get('satellite', '')
            
            # Determine marker color based on brightness
            if brightness > 500:
                color = 'darkred'
                icon_color = 'red'
            elif brightness > 350:
                color = 'red'
                icon_color = 'orange'
            else:
                color = 'orange'
                icon_color = 'yellow'
            
            # Create popup HTML
            popup_html = f"""
            <div style="font-family: Arial; width: 250px;">
                <h4 style="color: {color}; margin: 5px 0;">🔥 Fire Detection</h4>
                <table style="width: 100%; font-size: 12px;">
                    <tr><td><b>Location:</b></td><td>{fire_lat:.4f}, {fire_lng:.4f}</td></tr>
                    <tr><td><b>Brightness:</b></td><td>{brightness:.0f}</td></tr>
                    <tr><td><b>Confidence:</b></td><td>{confidence}</td></tr>
                    <tr><td><b>Date/Time:</b></td><td>{acq_date} {acq_time}</td></tr>
                    <tr><td><b>Satellite:</b></td><td>{satellite}</td></tr>
            """
            
            # Add prediction info if available
            if fire.get('prediction') and st.session_state.show_predictions:
                pred = fire['prediction']
                popup_html += f"""
                    <tr><td colspan="2"><hr style="margin: 5px 0;"></td></tr>
                    <tr><td colspan="2"><b style="color: #1f77b4;">AI Prediction:</b></td></tr>
                    <tr><td><b>Will Spread:</b></td><td>{"YES" if pred['will_spread'] else "NO"}</td></tr>
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
                tooltip=f"Fire: {brightness:.0f} | {acq_date}"
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
                            # Spread polygon
                            coords = feature['geometry']['coordinates'][0]
                            folium.Polygon(
                                locations=[(coord_lat, coord_lng) for coord_lng, coord_lat in coords],
                                color='red',
                                fill=True,
                                fillColor='red',
                                fillOpacity=0.2,
                                weight=2,
                                popup=f"Predicted Spread Zone<br>Probability: {pred['spread_probability']:.1%}"
                            ).add_to(m)
                        
                        elif feature_type == 'direction':
                            # Direction arrow
                            coords = feature['geometry']['coordinates']
                            folium.PolyLine(
                                locations=[(coord_lat, coord_lng) for coord_lng, coord_lat in coords],
                                color='darkred',
                                weight=3,
                                opacity=0.7,
                                popup=f"Wind Direction: {pred['spread_direction']:.0f}°"
                            ).add_to(m)
        
        # Add heatmap if enabled
        if st.session_state.show_heatmap:
            heatmap_points = []
            for fire in st.session_state.active_fires:
                if fire.get('prediction'):
                    pred = fire['prediction']
                    # Add center point with probability
                    heatmap_points.append([
                        fire['lat'],
                        fire['lng'],
                        pred['spread_probability']
                    ])
            
            if heatmap_points:
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
    
    # Handle map clicks
    if map_data and map_data.get('last_clicked'):
        clicked = map_data['last_clicked']
        # Find nearest fire
        if st.session_state.active_fires:
            min_dist = float('inf')
            nearest_fire = None
            for fire in st.session_state.active_fires:
                dist = ((fire['lat'] - clicked['lat'])**2 + (fire['lng'] - clicked['lng'])**2)**0.5
                if dist < min_dist:
                    min_dist = dist
                    nearest_fire = fire
            
            if min_dist < 0.01:  # Within ~1km
                st.session_state.selected_fire = nearest_fire
                st.rerun()

with col_info:
    st.subheader("📊 Fire Statistics")
    
    if st.session_state.active_fires:
        # Statistics
        total_fires = len(st.session_state.active_fires)
        fires_with_pred = sum(1 for f in st.session_state.active_fires if f.get('prediction'))
        avg_brightness = sum(f.get('brightness', 0) for f in st.session_state.active_fires) / total_fires if total_fires > 0 else 0
        
        st.metric("Total Fires", total_fires)
        st.metric("With Predictions", fires_with_pred)
        st.metric("Avg Brightness", f"{avg_brightness:.0f}")
        
        st.divider()
        
        # Fire list
        st.subheader("🔥 Fire List")
        for idx, fire in enumerate(st.session_state.active_fires[:10]):  # Show first 10
            with st.expander(f"Fire #{idx+1} - {fire.get('acq_date', 'N/A')}"):
                st.write(f"**Location:** ({fire['lat']:.4f}, {fire['lng']:.4f})")
                st.write(f"**Brightness:** {fire.get('brightness', 'N/A')}")
                
                if fire.get('prediction'):
                    pred = fire['prediction']
                    st.write("**Prediction:**")
                    st.write(f"- Spread: {'YES' if pred['will_spread'] else 'NO'}")
                    st.write(f"- Probability: {pred['spread_probability']:.1%}")
                    st.write(f"- Distance: {pred['spread_distance_km']:.2f} km")
                    
                    # Environmental data
                    env = pred.get('environmental_data', {})
                    with st.expander("Environmental Data"):
                        st.write(f"Temp: {env.get('temperature', 'N/A')}°C")
                        st.write(f"Humidity: {env.get('humidity', 'N/A')}%")
                        st.write(f"Wind: {env.get('wind_speed', 'N/A')} km/h")
                        st.write(f"Elevation: {env.get('elevation', 'N/A')} m")
    else:
        st.info("No fires loaded. Use filters to load fire data.")

# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: #666; font-size: 12px;'>
        <p>🔥 NASA FIRMS Enhanced with AI Fire Spread Prediction</p>
        <p>Data: <a href="https://firms.modaps.eosdis.nasa.gov/" target="_blank">NASA FIRMS</a> | 
        Weather: <a href="https://open-meteo.com/" target="_blank">Open-Meteo</a> | 
        Powered by Machine Learning</p>
    </div>
""", unsafe_allow_html=True)


