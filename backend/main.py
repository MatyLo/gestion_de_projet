"""
FastAPI Backend for Fire Spread Prediction
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import uvicorn
import os
from datetime import datetime

from prediction import predict_fire_spread, predict_direction
from fires import load_fires_from_csv, get_fire_statistics

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
    date: datetime = Field(...,description="Date of the fire observation (ISO 8601 format)")

class PredictResponse(BaseModel):
    direction: float

class BatchPredictionRequest(BaseModel):
    fire_ids: List[int] = Field(..., description="List of fire IDs to predict")


class RefreshFiresRequest(BaseModel):
    region: Optional[str] = Field(None, description="Region name (usa, europe, australia, brazil, canada, world)")
    min_lon: Optional[float] = Field(None, ge=-180, le=180)
    max_lon: Optional[float] = Field(None, ge=-180, le=180)
    min_lat: Optional[float] = Field(None, ge=-90, le=90)
    max_lat: Optional[float] = Field(None, ge=-90, le=90)
    days: int = Field(1, description="Number of days of data to fetch", ge=1, le=10)


# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Fire Spread Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Predict fire spread",
            "/fires": "GET - Get active fires from NASA FIRMS data",
            "/predict-batch": "POST - Predict spread for multiple fires",
            "/refresh-fires": "POST - Refresh fire data from NASA FIRMS API",
            "/regions": "GET - Get available regions for fire data",
            "/health": "GET - Health check"
        }
    }


# Health check endpoint
@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/regions")
async def get_regions():
    """
    Get available preset regions for fire data fetching
    """
    from api_request import REGIONS
    return {
        "regions": list(REGIONS.keys()),
        "region_details": REGIONS
    }


# Fires endpoint
@app.get("/fires")
async def get_fires(
    confidence_min: int = 0,
    brightness_min: float = 0.0,
    limit: Optional[int] = None
):
    """
    Get active fires from NASA FIRMS CSV data.
    
    - **confidence_min**: Minimum confidence level (0-100)
    - **brightness_min**: Minimum brightness value
    - **limit**: Maximum number of fires to return
    
    Returns list of active fires with metadata
    """
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, "..", "data", "world_hotspots.csv")
        if not os.path.exists(csv_path):
            csv_path = os.path.join(script_dir, "data", "world_hotspots.csv")
        
        fires = load_fires_from_csv(
            filepath=csv_path,
            confidence_min=confidence_min,
            brightness_min=brightness_min,
            limit=limit
        )
        
        stats = get_fire_statistics(fires)
        
        return {
            "count": len(fires),
            "statistics": stats,
            "fires": fires
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load fires: {str(e)}"
        )


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


# Direction Endpoint
@app.post(
    "/predict/direction",
    summary="Predict fire spread direction",
    description="Predicts the fire spread direction using location and date"
)
def predict_direction_endpoint(payload: PredictionRequest):
    try:
        prediction = predict_direction(
            lat=payload.lat,
            lng=payload.lng,
            date=payload.date
        )

        return {
            "direction": prediction
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

# Batch prediction endpoint
@app.post("/predict-batch")
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict fire spread for multiple fires.
    
    - **fire_ids**: List of fire IDs from the fires dataset
    
    Returns prediction results for each fire
    """
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, "..", "data", "MODIS_C6_1_Global_24h.csv")
        if not os.path.exists(csv_path):
            csv_path = os.path.join(script_dir, "data", "MODIS_C6_1_Global_24h.csv")
        
        all_fires = load_fires_from_csv(filepath=csv_path)
        fires_dict = {f['id']: f for f in all_fires}
        
        results = []
        for fire_id in request.fire_ids:
            if fire_id not in fires_dict:
                continue
            
            fire = fires_dict[fire_id]
            prediction = predict_fire_spread(
                lat=fire['latitude'],
                lng=fire['longitude'],
                brightness=fire['brightness']
            )
            prediction['fire_id'] = fire_id
            prediction['fire_data'] = fire
            results.append(prediction)
        
        return {
            "count": len(results),
            "predictions": results
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.post("/refresh-fires")
async def refresh_fires(request: RefreshFiresRequest):
    """
    Fetch fresh fire data from NASA FIRMS API and update the CSV file.
    
    You can either:
    - Use a preset region name (usa, europe, australia, brazil, canada, world)
    - Specify custom bounding box coordinates (min_lon, max_lon, min_lat, max_lat)
    
    - **region**: Preset region name
    - **min_lon, max_lon, min_lat, max_lat**: Custom bounding box
    - **days**: Number of days of data to fetch (1-10)
    
    Returns updated fire data and saves to CSV file
    """
    import traceback
    from api_request import get_firms_hotspots, REGIONS
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, "..", "data", "world_hotspots.csv")
        if not os.path.exists(os.path.dirname(csv_path)):
            csv_path = os.path.join(script_dir, "data", "world_hotspots.csv")
        
        if request.region:
            if request.region not in REGIONS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid region. Available: {list(REGIONS.keys())}"
                )
            region_coords = REGIONS[request.region]
            min_lon = region_coords["min_lon"]
            max_lon = region_coords["max_lon"]
            min_lat = region_coords["min_lat"]
            max_lat = region_coords["max_lat"]
        elif all(v is not None for v in [request.min_lon, request.max_lon, request.min_lat, request.max_lat]):
            min_lon = request.min_lon
            max_lon = request.max_lon
            min_lat = request.min_lat
            max_lat = request.max_lat
        else:
            min_lon, max_lon, min_lat, max_lat = -180, 180, -90, 90
        
        df = get_firms_hotspots(min_lon, max_lon, min_lat, max_lat, request.days)
        
        print(f"Columns returned from API: {list(df.columns)}")
        print(f"First row sample: {df.head(1).to_dict('records') if not df.empty else 'Empty'}")
        
        if df.empty:
            return {
                "success": False,
                "message": "No fire data found for the specified region/coordinates",
                "count": 0
            }
        
        df.to_csv(csv_path, index=False)
        
        fires = load_fires_from_csv(filepath=csv_path)
        stats = get_fire_statistics(fires)
        
        return {
            "success": True,
            "message": f"Successfully fetched and saved {len(fires)} fires",
            "count": len(fires),
            "statistics": stats,
            "region": request.region if request.region else "custom",
            "days": request.days,
            "csv_path": csv_path
        }
        
    except Exception as e:
        error_detail = f"Failed to refresh fires: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)
        raise HTTPException(
            status_code=500,
            detail=error_detail
        )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

