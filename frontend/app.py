"""
Streamlit Frontend for Fire Spread Prediction
A minimalistic interface for wildfire spread prediction
"""
import streamlit as st
import requests
import folium
from folium import plugins
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
    .marker-cluster-small {
        background-color: rgba(255, 181, 71, 0.6);
    }
    .marker-cluster-small div {
        background-color: rgba(255, 140, 0, 0.8);
        color: white;
        font-weight: bold;
    }
    .marker-cluster-medium {
        background-color: rgba(255, 120, 50, 0.6);
    }
    .marker-cluster-medium div {
        background-color: rgba(255, 87, 34, 0.8);
        color: white;
        font-weight: bold;
    }
    .marker-cluster-large {
        background-color: rgba(220, 50, 50, 0.6);
    }
    .marker-cluster-large div {
        background-color: rgba(183, 28, 28, 0.8);
        color: white;
        font-weight: bold;
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
if 'show_fires' not in st.session_state:
    st.session_state.show_fires = False
if 'fires_data' not in st.session_state:
    st.session_state.fires_data = None
if 'selected_fire_id' not in st.session_state:
    st.session_state.selected_fire_id = None
if 'auto_predict' not in st.session_state:
    st.session_state.auto_predict = True
if 'pending_prediction' not in st.session_state:
    st.session_state.pending_prediction = False
if 'last_fire_clicked' not in st.session_state:
    st.session_state.last_fire_clicked = None
if 'map_refresh_counter' not in st.session_state:
    st.session_state.map_refresh_counter = 0
if 'user_map_center' not in st.session_state:
    st.session_state.user_map_center = None
if 'user_map_zoom' not in st.session_state:
    st.session_state.user_map_zoom = None
if 'fire_just_selected' not in st.session_state:
    st.session_state.fire_just_selected = False
if 'refresh_result' not in st.session_state:
    st.session_state.refresh_result = None

# Sidebar for inputs
with st.sidebar:
    st.header("🎯 Location Input")
    
    # Mode selection
    mode = st.radio(
        "Mode:",
        ["Manual Input", "Map Click", "Active Fires"],
        help="Choose input mode"
    )
    
    # Active Fires mode
    if mode == "Active Fires":
        st.subheader("🔥 NASA FIRMS Data")
        
        with st.expander("🌍 Refresh Fire Data from API", expanded=False):
            st.write("Fetch fresh fire data from NASA FIRMS")
            
            refresh_mode = st.radio(
                "Select mode:",
                ["Preset Region", "Custom Area"],
                horizontal=True
            )
            
            if refresh_mode == "Preset Region":
                region = st.selectbox(
                    "Region:",
                    ["world", "usa", "europe", "australia", "brazil", "canada"],
                    help="Select a preset region"
                )
                days = st.slider("Days of data:", 1, 10, 1)
                
                if st.button("🔄 Fetch from API", type="secondary"):
                    with st.spinner("Fetching fresh fire data from NASA FIRMS..."):
                        try:
                            response = requests.post(
                                f"{API_URL}/refresh-fires",
                                json={"region": region, "days": days},
                                timeout=30
                            )
                            if response.status_code == 200:
                                result = response.json()
                                if result.get('success'):
                                    st.session_state.refresh_result = result
                                    st.success(f"✅ {result['message']}")
                                    st.info("Click 'Load Active Fires' below to see the new data")
                                else:
                                    st.warning(f"⚠️ {result.get('message', 'No data found')}")
                            else:
                                st.error(f"❌ Error: {response.status_code}")
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
            else:
                col_lon1, col_lon2 = st.columns(2)
                with col_lon1:
                    min_lon = st.number_input("Min Longitude", -180.0, 180.0, -125.0)
                with col_lon2:
                    max_lon = st.number_input("Max Longitude", -180.0, 180.0, -66.0)
                
                col_lat1, col_lat2 = st.columns(2)
                with col_lat1:
                    min_lat = st.number_input("Min Latitude", -90.0, 90.0, 24.0)
                with col_lat2:
                    max_lat = st.number_input("Max Latitude", -90.0, 90.0, 49.0)
                
                days = st.slider("Days of data:", 1, 10, 1, key="custom_days")
                
                if st.button("🔄 Fetch from API", type="secondary", key="custom_fetch"):
                    with st.spinner("Fetching fresh fire data from NASA FIRMS..."):
                        try:
                            response = requests.post(
                                f"{API_URL}/refresh-fires",
                                json={
                                    "min_lon": min_lon,
                                    "max_lon": max_lon,
                                    "min_lat": min_lat,
                                    "max_lat": max_lat,
                                    "days": days
                                },
                                timeout=30
                            )
                            if response.status_code == 200:
                                result = response.json()
                                if result.get('success'):
                                    st.session_state.refresh_result = result
                                    st.success(f"✅ {result['message']}")
                                    st.info("Click 'Load Active Fires' below to see the new data")
                                else:
                                    st.warning(f"⚠️ {result.get('message', 'No data found')}")
                            else:
                                st.error(f"❌ Error: {response.status_code}")
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
            
            if st.session_state.refresh_result:
                st.divider()
                st.write("**Last Refresh:**")
                result = st.session_state.refresh_result
                st.write(f"Fires fetched: {result['count']}")
                st.write(f"Region: {result.get('region', 'custom')}")
                st.write(f"Days: {result['days']}")
        
        st.divider()
        
        st.session_state.auto_predict = st.checkbox(
            "Auto-predict on fire click",
            value=st.session_state.auto_predict,
            help="Automatically predict spread when clicking a fire"
        )
        
        col_a, col_b = st.columns(2)
        with col_a:
            confidence_min = st.number_input(
                "Min Confidence",
                min_value=0,
                max_value=100,
                value=50,
                step=10
            )
        with col_b:
            brightness_min = st.number_input(
                "Min Brightness",
                min_value=0.0,
                max_value=500.0,
                value=300.0,
                step=10.0
            )
        
        fire_limit = st.number_input(
            "Max Fires to Display",
            min_value=10,
            max_value=1000,
            value=100,
            step=10
        )
        
        if st.button("🔄 Load Active Fires", type="primary"):
            with st.spinner("Loading fires..."):
                try:
                    response = requests.get(
                        f"{API_URL}/fires",
                        params={
                            "confidence_min": confidence_min,
                            "brightness_min": brightness_min,
                            "limit": fire_limit
                        },
                        timeout=10
                    )
                    if response.status_code == 200:
                        st.session_state.fires_data = response.json()
                        st.session_state.show_fires = True
                        st.session_state.map_refresh_counter += 1
                        st.session_state.user_map_center = None
                        st.session_state.user_map_zoom = None
                        st.success(f"✅ Loaded {st.session_state.fires_data['count']} fires")
                        st.rerun()
                    else:
                        st.error(f"❌ Error: {response.status_code}")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        if st.session_state.fires_data:
            stats = st.session_state.fires_data['statistics']
            st.metric("Total Fires", stats['count'])
            st.metric("Avg Brightness", f"{stats['avg_brightness']:.1f}")
            st.metric("Max Brightness", f"{stats['max_brightness']:.1f}")
            
            if st.button("❌ Clear Fires"):
                st.session_state.fires_data = None
                st.session_state.show_fires = False
                st.session_state.selected_fire_id = None
                st.session_state.prediction_result = None
                st.session_state.pending_prediction = False
                st.session_state.last_fire_clicked = None
                st.session_state.map_refresh_counter += 1
                st.session_state.user_map_center = None
                st.session_state.user_map_zoom = None
                st.rerun()
        
        st.divider()
        input_method = "Active Fires"
    else:
        input_method = mode
    
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
    elif input_method == "Map Click":
        st.info("👆 Click on the map to select a location")
        lat = st.session_state.selected_lat
        lng = st.session_state.selected_lng
        st.write(f"**Selected:** ({lat:.4f}, {lng:.4f})")
    else:
        if st.session_state.selected_fire_id is not None:
            fire = next((f for f in st.session_state.fires_data['fires'] if f['id'] == st.session_state.selected_fire_id), None)
            if fire:
                st.write(f"**Selected Fire:**")
                st.write(f"Lat: {fire['latitude']:.4f}")
                st.write(f"Lng: {fire['longitude']:.4f}")
                st.write(f"Brightness: {fire['brightness']:.1f}")
                st.write(f"Confidence: {fire['confidence']}%")
                lat = fire['latitude']
                lng = fire['longitude']
                brightness = fire['brightness']
            else:
                st.info("👆 Click on a fire marker to select")
                lat = st.session_state.selected_lat
                lng = st.session_state.selected_lng
        else:
            st.info("👆 Click on a fire marker to select")
            lat = st.session_state.selected_lat
            lng = st.session_state.selected_lng
    
    st.divider()
    
    # Fire parameters
    if input_method != "Active Fires" or st.session_state.selected_fire_id is None:
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
    if input_method == "Active Fires" and st.session_state.fires_data:
        predict_button = st.button("🚀 Predict Fire Spread", type="primary", disabled=st.session_state.selected_fire_id is None)
    else:
        predict_button = st.button("🚀 Predict Fire Spread", type="primary")

# Function to make prediction
def make_prediction(lat, lng, brightness):
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={
                "lat": lat,
                "lng": lng,
                "brightness": brightness
            },
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        return None

# Handle auto-prediction BEFORE rendering the map
if st.session_state.pending_prediction and mode == "Active Fires":
    if st.session_state.fires_data:
        fire = next((f for f in st.session_state.fires_data['fires'] if f['id'] == st.session_state.selected_fire_id), None)
        if fire:
            result = make_prediction(fire['latitude'], fire['longitude'], fire['brightness'])
            if result:
                st.session_state.prediction_result = result
    st.session_state.pending_prediction = False

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📍 Map View")
    
    if st.session_state.fires_data and st.session_state.show_fires:
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        with col_stat1:
            st.metric("Active Fires", st.session_state.fires_data['count'])
        with col_stat2:
            if st.session_state.selected_fire_id is not None:
                st.metric("Selected Fire", f"#{st.session_state.selected_fire_id}")
            else:
                st.metric("Selected Fire", "None")
        with col_stat3:
            if st.session_state.prediction_result:
                st.metric("Spread Risk", 
                         "HIGH" if st.session_state.prediction_result.get('will_spread') else "LOW",
                         delta=f"{st.session_state.prediction_result.get('spread_probability', 0):.0%}")
            else:
                st.metric("Spread Risk", "N/A")
    
    # Determine map center
    if st.session_state.fire_just_selected:
        map_center = [st.session_state.selected_lat, st.session_state.selected_lng]
        zoom_start = 10
    elif st.session_state.user_map_center is not None and st.session_state.user_map_zoom is not None:
        map_center = st.session_state.user_map_center
        zoom_start = st.session_state.user_map_zoom
    elif st.session_state.selected_fire_id is not None and st.session_state.fires_data:
        map_center = [st.session_state.selected_lat, st.session_state.selected_lng]
        zoom_start = 10
    elif st.session_state.fires_data and st.session_state.show_fires:
        fires = st.session_state.fires_data['fires']
        if fires:
            avg_lat = sum(f['latitude'] for f in fires) / len(fires)
            avg_lng = sum(f['longitude'] for f in fires) / len(fires)
            map_center = [avg_lat, avg_lng]
            zoom_start = 4
        else:
            map_center = [st.session_state.selected_lat, st.session_state.selected_lng]
            zoom_start = 10
    else:
        map_center = [st.session_state.selected_lat, st.session_state.selected_lng]
        zoom_start = 10
    
    # Create base map
    m = folium.Map(
        location=map_center,
        zoom_start=zoom_start,
        tiles="OpenStreetMap"
    )
    
    # Add legend
    if st.session_state.fires_data and st.session_state.show_fires:
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; right: 50px; width: 200px; height: auto; 
                    background-color: white; z-index:9999; font-size:14px;
                    border:2px solid grey; border-radius: 5px; padding: 10px">
        <p style="margin:0; font-weight:bold; text-align:center;">Fire Confidence</p>
        <p style="margin:5px 0;"><span style="color:red;">●</span> High (70%+)</p>
        <p style="margin:5px 0;"><span style="color:orange;">●</span> Medium (50-70%)</p>
        <p style="margin:5px 0;"><span style="color:lightcoral;">●</span> Low (&lt;50%)</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
    
    # Add active fires if loaded
    if st.session_state.fires_data and st.session_state.show_fires:
        fires = st.session_state.fires_data['fires']
        
        marker_cluster = plugins.MarkerCluster(
            name='Active Fires',
            overlay=True,
            control=True,
            icon_create_function="""
                function(cluster) {
                    var childCount = cluster.getChildCount();
                    var c = ' marker-cluster-';
                    if (childCount < 10) {
                        c += 'small';
                    } else if (childCount < 50) {
                        c += 'medium';
                    } else {
                        c += 'large';
                    }
                    return new L.DivIcon({ 
                        html: '<div><span>' + childCount + '</span></div>', 
                        className: 'marker-cluster' + c, 
                        iconSize: new L.Point(40, 40) 
                    });
                }
            """
        )
        
        for fire in fires:
            is_selected = fire['id'] == st.session_state.selected_fire_id
            
            if fire['confidence'] >= 70:
                color = 'darkred' if is_selected else 'red'
            elif fire['confidence'] >= 50:
                color = 'orange' if is_selected else 'orange'
            else:
                color = 'lightred' if is_selected else 'lightred'
            
            radius = min(10, max(3, fire['brightness'] / 50))
            if is_selected:
                radius *= 1.5
            
            popup_html = f"""
                <b>Fire #{fire['id']}</b><br>
                Lat: {fire['latitude']:.4f}<br>
                Lng: {fire['longitude']:.4f}<br>
                Brightness: {fire['brightness']:.1f}<br>
                Confidence: {fire['confidence']}%<br>
                FRP: {fire['frp']:.2f}<br>
                Date: {fire['acq_date']}<br>
                Time: {fire['acq_time']}<br>
                Satellite: {fire['satellite']}
            """
            
            folium.CircleMarker(
                location=[fire['latitude'], fire['longitude']],
                radius=radius,
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"Fire #{fire['id']} - Brightness: {fire['brightness']:.1f}",
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.7,
                weight=2 if is_selected else 1
            ).add_to(marker_cluster)
        
        marker_cluster.add_to(m)
    
    # Add marker for manually selected location (non-fire mode)
    if input_method in ["Manual Input", "Map Click"]:
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
                    coords = feature['geometry']['coordinates'][0]
                    
                    will_spread = result.get('will_spread', False)
                    spread_prob = result.get('spread_probability', 0)
                    
                    if will_spread and spread_prob > 0.7:
                        spread_color = '#ff4444'
                        fill_opacity = 0.4
                    elif will_spread and spread_prob > 0.5:
                        spread_color = '#ff8800'
                        fill_opacity = 0.35
                    else:
                        spread_color = '#ffaa00'
                        fill_opacity = 0.25
                    
                    folium.Polygon(
                        locations=[(lat, lng) for lng, lat in coords],
                        color=spread_color,
                        fill=True,
                        fillColor=spread_color,
                        fillOpacity=fill_opacity,
                        weight=3,
                        popup=folium.Popup(
                            f"""
                            <b>Predicted Spread Zone</b><br>
                            Probability: {spread_prob:.1%}<br>
                            Distance: {result.get('spread_distance_km', 0):.2f} km<br>
                            Will Spread: {'YES' if will_spread else 'NO'}
                            """,
                            max_width=250
                        ),
                        tooltip=f"Spread Probability: {spread_prob:.1%}"
                    ).add_to(m)
                
                elif feature_type == 'direction':
                    coords = feature['geometry']['coordinates']
                    
                    folium.PolyLine(
                        locations=[(lat, lng) for lng, lat in coords],
                        color='#cc0000',
                        weight=4,
                        opacity=0.9,
                        popup=folium.Popup(
                            f"""
                            <b>Spread Direction</b><br>
                            Direction: {result.get('spread_direction', 0):.0f}°<br>
                            Wind Speed: {result.get('environmental_data', {}).get('wind_speed', 0):.1f} km/h
                            """,
                            max_width=250
                        ),
                        tooltip=f"Direction: {result.get('spread_direction', 0):.0f}°"
                    ).add_to(m)
                    
                    if len(coords) >= 2:
                        end_point = coords[-1]
                        folium.CircleMarker(
                            location=[end_point[1], end_point[0]],
                            radius=6,
                            color='#cc0000',
                            fill=True,
                            fillColor='#cc0000',
                            fillOpacity=1.0,
                            weight=2
                        ).add_to(m)
    
    # Display map and capture clicks
    map_data = st_folium(
        m,
        width=None,
        height=500,
        returned_objects=["last_object_clicked", "last_clicked"],
        key=f"map_{st.session_state.map_refresh_counter}"
    )
    
    # Capture user's current map position
    if map_data:
        if map_data.get('center'):
            st.session_state.user_map_center = [map_data['center']['lat'], map_data['center']['lng']]
        if map_data.get('zoom'):
            st.session_state.user_map_zoom = map_data['zoom']
    
    # Clear the fire_just_selected flag after map renders
    if st.session_state.fire_just_selected:
        st.session_state.fire_just_selected = False
    
    # Update selected location if map was clicked
    if input_method == "Map Click" and map_data and map_data.get('last_clicked'):
        clicked = map_data['last_clicked']
        if clicked:
            st.session_state.selected_lat = clicked['lat']
            st.session_state.selected_lng = clicked['lng']
            st.rerun()
    
    # Handle fire selection
    if input_method == "Active Fires" and map_data and map_data.get('last_object_clicked'):
        clicked = map_data['last_object_clicked']
        if clicked and st.session_state.fires_data:
            clicked_lat = clicked.get('lat')
            clicked_lng = clicked.get('lng')
            if clicked_lat and clicked_lng:
                clicked_key = f"{clicked_lat:.4f}_{clicked_lng:.4f}"
                
                if st.session_state.last_fire_clicked != clicked_key:
                    for fire in st.session_state.fires_data['fires']:
                        if abs(fire['latitude'] - clicked_lat) < 0.01 and abs(fire['longitude'] - clicked_lng) < 0.01:
                            previous_fire_id = st.session_state.selected_fire_id
                            st.session_state.selected_fire_id = fire['id']
                            st.session_state.selected_lat = fire['latitude']
                            st.session_state.selected_lng = fire['longitude']
                            st.session_state.last_fire_clicked = clicked_key
                            st.session_state.fire_just_selected = True
                            
                            if st.session_state.auto_predict and previous_fire_id != fire['id']:
                                st.session_state.pending_prediction = True
                            
                            st.rerun()
                            break

with col2:
    st.header("📊 Prediction Results")
    
    # Make prediction when button is clicked
    if predict_button:
        with st.spinner("🔄 Analyzing fire spread..."):
            if input_method == "Active Fires" and st.session_state.selected_fire_id is not None:
                fire = next((f for f in st.session_state.fires_data['fires'] if f['id'] == st.session_state.selected_fire_id), None)
                if fire:
                    result = make_prediction(fire['latitude'], fire['longitude'], fire['brightness'])
                    if result:
                        st.session_state.prediction_result = result
                        st.success("✅ Prediction completed!")
                        st.rerun()
                    else:
                        st.error("❌ Prediction failed")
                else:
                    st.error("❌ Selected fire not found")
            else:
                result = make_prediction(st.session_state.selected_lat, st.session_state.selected_lng, brightness)
                if result:
                    st.session_state.prediction_result = result
                    st.success("✅ Prediction completed!")
                    st.rerun()
                else:
                    st.error("❌ Prediction failed")
    
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
        col_download, col_clear = st.columns([3, 1])
        with col_download:
            st.download_button(
                label="📥 Download Results",
                data=json.dumps(result, indent=2),
                file_name="fire_prediction.json",
                mime="application/json"
            )
        with col_clear:
            if st.button("🗑️", help="Clear prediction"):
                st.session_state.prediction_result = None
                st.rerun()
    else:
        st.info("👈 Set location and click 'Predict Fire Spread' to see results")
        
        if input_method == "Active Fires" and st.session_state.auto_predict:
            st.info("💡 Auto-predict is ON - click any fire marker to see predictions")

# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: #667;'>
        <p>Fire Spread Prediction System | Powered by Machine Learning</p>
        <p><small>Data sources: Open-Meteo API for weather and elevation</small></p>
    </div>
""", unsafe_allow_html=True)

