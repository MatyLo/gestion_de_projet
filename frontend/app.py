"""
Streamlit Frontend for Fire Spread Prediction
A minimalistic interface for wildfire spread prediction
"""
import streamlit as st
import requests
import folium
from streamlit_folium import st_folium
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
st.title("🔥 Fire Spread Prediction System")
st.markdown("Predict wildfire spread based on location and environmental factors")

# Initialize session state
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'selected_lat' not in st.session_state:
    st.session_state.selected_lat = 34.05
if 'selected_lng' not in st.session_state:
    st.session_state.selected_lng = -118.25

# Sidebar for inputs
with st.sidebar:
    st.header("🎯 Location Input")
    
    # Method selection
    input_method = st.radio(
        "Select input method:",
        ["Manual Input", "Map Click"],
        help="Choose how to specify the fire location"
    )
    
    if input_method == "Manual Input":
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
        st.info("👆 Click on the map to select a location")
        lat = st.session_state.selected_lat
        lng = st.session_state.selected_lng
        st.write(f"**Selected:** ({lat:.4f}, {lng:.4f})")
    
    st.divider()
    
    # Fire parameters
    st.header("🔥 Fire Parameters")
    brightness = st.slider(
        "Fire Brightness/Intensity",
        min_value=100,
        max_value=800,
        value=350,
        step=10,
        help="Higher brightness indicates more intense fire"
    )
    
    st.divider()
    
    # Predict button
    predict_button = st.button("🚀 Predict Fire Spread", type="primary")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📍 Map View")
    
    # Create base map
    m = folium.Map(
        location=[st.session_state.selected_lat, st.session_state.selected_lng],
        zoom_start=10,
        tiles="OpenStreetMap"
    )
    
    # Add marker for selected location
    folium.Marker(
        [st.session_state.selected_lat, st.session_state.selected_lng],
        popup=f"Fire Location<br>Lat: {st.session_state.selected_lat:.4f}<br>Lng: {st.session_state.selected_lng:.4f}",
        icon=folium.Icon(color="red", icon="fire", prefix="fa")
    ).add_to(m)
    
    # If we have prediction results, add them to the map
    if st.session_state.prediction_result:
        result = st.session_state.prediction_result
        geojson = result.get('geojson', {})
        
        # Add GeoJSON to map
        if geojson and 'features' in geojson:
            for feature in geojson['features']:
                feature_type = feature['properties'].get('type')
                
                if feature_type == 'spread':
                    # Add spread polygon
                    coords = feature['geometry']['coordinates'][0]
                    folium.Polygon(
                        locations=[(lat, lng) for lng, lat in coords],
                        color='orange',
                        fill=True,
                        fillColor='orange',
                        fillOpacity=0.3,
                        popup=f"Spread Probability: {result['spread_probability']:.2%}"
                    ).add_to(m)
                
                elif feature_type == 'direction':
                    # Add direction arrow
                    coords = feature['geometry']['coordinates']
                    folium.PolyLine(
                        locations=[(lat, lng) for lng, lat in coords],
                        color='red',
                        weight=3,
                        opacity=0.8,
                        popup=f"Wind Direction: {result['spread_direction']:.0f}°"
                    ).add_to(m)
    
    # Display map and capture clicks
    map_data = st_folium(
        m,
        width=None,
        height=500,
        key="map"
    )
    
    # Update selected location if map was clicked
    if input_method == "Map Click" and map_data and map_data.get('last_clicked'):
        clicked = map_data['last_clicked']
        if clicked:
            st.session_state.selected_lat = clicked['lat']
            st.session_state.selected_lng = clicked['lng']
            st.rerun()

with col2:
    st.header("📊 Prediction Results")
    
    # Make prediction when button is clicked
    if predict_button:
        with st.spinner("🔄 Analyzing fire spread..."):
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
                    st.success("✅ Prediction completed!")
                    st.rerun()
                else:
                    st.error(f"❌ Error: {response.status_code}")
                    st.json(response.json())
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to backend API. Make sure it's running on http://localhost:8000")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
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
        <p>Fire Spread Prediction System | Powered by Machine Learning</p>
        <p><small>Data sources: Open-Meteo API for weather and elevation</small></p>
    </div>
""", unsafe_allow_html=True)

