"""
FastAPI Backend for Fire Spread Prediction
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import uvicorn
from datetime import datetime

from prediction import predict_fire_spread, generate_heatmap_grid
from nasa_firms import get_firms_fires, get_firms_fires_by_bbox, get_firms_fires_by_country

# Create FastAPI app
app = FastAPI(
    title="Fire Spread Prediction API",
    description="Predict wildfire spread based on location and environmental factors",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Streamlit app's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request models
class PredictionRequest(BaseModel):
    lat: float = Field(..., description="Latitude of fire location", ge=-90, le=90)
    lng: float = Field(..., description="Longitude of fire location", ge=-180, le=180)
    brightness: Optional[float] = Field(350.0, description="Fire brightness/intensity", ge=0, le=1000)

class FireRequest(BaseModel):
    lat: float = Field(..., description="Latitude of fire location", ge=-90, le=90)
    lng: float = Field(..., description="Longitude of fire location", ge=-180, le=180)
    brightness: Optional[float] = Field(350.0, description="Fire brightness/intensity", ge=0, le=1000)
    name: Optional[str] = Field(None, description="Optional fire name/identifier")

# In-memory storage for active fires (in production, use a database)
active_fires = []


# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Fire Spread Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Predict fire spread",
            "/fires": "GET/POST - Get NASA FIRMS fires or manage manual fires",
            "/heatmap": "POST - Generate heatmap grid",
            "/health": "GET - Health check"
        }
    }


# Health check endpoint
@app.get("/health")
async def health():
    return {"status": "healthy"}


# Prediction endpoint
@app.post("/predict")
async def predict(request: PredictionRequest):
    """
    Predict fire spread based on location and brightness.
    
    - **lat**: Latitude of the fire location (-90 to 90)
    - **lng**: Longitude of the fire location (-180 to 180)
    - **brightness**: Fire brightness/intensity (optional, default: 350)
    
    Returns prediction results including:
    - Spread probability
    - Spread direction and distance
    - Environmental data used
    - GeoJSON for visualization
    """
    try:
        result = predict_fire_spread(
            lat=request.lat,
            lng=request.lng,
            brightness=request.brightness
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# Active fires management
@app.get("/fires")
async def get_fires(
    source: str = "nasa",  # "nasa" or "manual"
    country: Optional[str] = None,
    bbox_min_lat: Optional[float] = None,
    bbox_min_lng: Optional[float] = None,
    bbox_max_lat: Optional[float] = None,
    bbox_max_lng: Optional[float] = None,
    days: int = 1
):
    """
    Get active fires from NASA FIRMS or manual fires.
    
    - **source**: "nasa" (default) or "manual"
    - **country**: ISO country code (e.g., "FRA", "USA") - only for NASA source
    - **bbox_***: Bounding box coordinates - only for NASA source
    - **days**: Number of days of data to retrieve (1-10) - only for NASA source
    """
    if source == "nasa":
        # Récupérer les feux depuis NASA FIRMS
        if country:
            fires_data = get_firms_fires_by_country(country, days=days)
        elif bbox_min_lat and bbox_min_lng and bbox_max_lat and bbox_max_lng:
            fires_data = get_firms_fires_by_bbox(
                bbox_min_lat, bbox_min_lng, bbox_max_lat, bbox_max_lng, days=days
            )
        else:
            # Par défaut, récupérer les feux récents dans une zone large
            fires_data = get_firms_fires(days=days)
        
        # Convertir au format attendu par le frontend
        formatted_fires = []
        for idx, fire in enumerate(fires_data):
            # Générer une prédiction pour chaque feu
            try:
                prediction = predict_fire_spread(
                    lat=fire['lat'],
                    lng=fire['lng'],
                    brightness=fire.get('brightness', 350)
                )
            except Exception as e:
                print(f"Erreur lors de la prédiction pour le feu {idx}: {e}")
                prediction = None
            
            formatted_fire = {
                "id": idx + 1,
                "lat": fire['lat'],
                "lng": fire['lng'],
                "brightness": fire.get('brightness', 350),
                "name": f"Feu NASA {fire.get('acq_date', '')} {fire.get('acq_time', '')}",
                "timestamp": f"{fire.get('acq_date', '')} {fire.get('acq_time', '')}",
                "source": "nasa_firms",
                "confidence": fire.get('confidence'),
                "satellite": fire.get('satellite', ''),
                "frp": fire.get('frp'),
                "prediction": prediction
            }
            formatted_fires.append(formatted_fire)
        
        return {"fires": formatted_fires, "count": len(formatted_fires), "source": "nasa"}
    else:
        # Retourner les feux manuels
        return {"fires": active_fires, "count": len(active_fires), "source": "manual"}


@app.post("/fires")
async def add_fire(fire: FireRequest):
    """Add a new active fire"""
    try:
        # Get prediction for this fire
        prediction = predict_fire_spread(
            lat=fire.lat,
            lng=fire.lng,
            brightness=fire.brightness
        )
        
        fire_data = {
            "id": len(active_fires) + 1,
            "lat": fire.lat,
            "lng": fire.lng,
            "brightness": fire.brightness,
            "name": fire.name or f"Fire #{len(active_fires) + 1}",
            "timestamp": datetime.now().isoformat(),
            "prediction": prediction
        }
        active_fires.append(fire_data)
        return fire_data
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to add fire: {str(e)}"
        )


@app.delete("/fires/{fire_id}")
async def delete_fire(fire_id: int):
    """Delete an active fire by ID"""
    global active_fires
    original_count = len(active_fires)
    active_fires = [f for f in active_fires if f["id"] != fire_id]
    
    if len(active_fires) == original_count:
        raise HTTPException(status_code=404, detail="Fire not found")
    
    return {"message": "Fire deleted", "remaining": len(active_fires)}


@app.post("/heatmap")
async def get_heatmap(request: PredictionRequest):
    """
    Generate heatmap data for fire spread visualization.
    
    Returns a grid of [lat, lng, probability] points for heatmap display.
    """
    try:
        heatmap_data = generate_heatmap_grid(
            lat=request.lat,
            lng=request.lng,
            brightness=request.brightness,
            grid_size=50
        )
        return {
            "heatmap": heatmap_data,
            "center": {"lat": request.lat, "lng": request.lng},
            "point_count": len(heatmap_data)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Heatmap generation failed: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

