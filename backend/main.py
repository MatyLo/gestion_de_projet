"""
FastAPI Backend for Fire Spread Prediction
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn

from prediction import predict_fire_spread

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


# Request model
class PredictionRequest(BaseModel):
    lat: float = Field(..., description="Latitude of fire location", ge=-90, le=90)
    lng: float = Field(..., description="Longitude of fire location", ge=-180, le=180)
    brightness: Optional[float] = Field(350.0, description="Fire brightness/intensity", ge=0, le=1000)


# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Fire Spread Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Predict fire spread",
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


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

