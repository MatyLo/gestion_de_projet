"""
Fire Data Loading Module
Handles loading and filtering fire data from NASA FIRMS CSV
"""
import pandas as pd
import os
from typing import List, Dict, Optional


def load_fires_from_csv(
    filepath: str,
    confidence_min: int = 0,
    brightness_min: float = 0.0,
    limit: Optional[int] = None
) -> List[Dict]:
    """
    Load fire data from NASA FIRMS CSV file
    
    Args:
        filepath: Path to CSV file
        confidence_min: Minimum confidence level (0-100)
        brightness_min: Minimum brightness value
        limit: Maximum number of fires to return
    
    Returns:
        List of fire dictionaries with lat, lng, brightness, etc.
    """
    if not os.path.exists(filepath):
        return []
    
    df = pd.read_csv(filepath)
    
    if confidence_min > 0:
        df = df[df['confidence'] >= confidence_min]
    
    if brightness_min > 0:
        df = df[df['brightness'] >= brightness_min]
    
    if limit:
        df = df.head(limit)
    
    df = df.sort_values('brightness', ascending=False)
    
    fires = []
    for idx, row in df.iterrows():
        fires.append({
            'id': int(idx),
            'latitude': float(row['latitude']),
            'longitude': float(row['longitude']),
            'brightness': float(row['brightness']),
            'confidence': int(row['confidence']),
            'frp': float(row['frp']) if pd.notna(row['frp']) else 0.0,
            'acq_date': str(row['acq_date']),
            'acq_time': str(row['acq_time']),
            'satellite': str(row['satellite']),
            'daynight': str(row['daynight'])
        })
    
    return fires


def get_fire_statistics(fires: List[Dict]) -> Dict:
    """Get statistics about the fire data"""
    if not fires:
        return {
            'count': 0,
            'avg_brightness': 0,
            'avg_confidence': 0,
            'max_brightness': 0
        }
    
    return {
        'count': len(fires),
        'avg_brightness': sum(f['brightness'] for f in fires) / len(fires),
        'avg_confidence': sum(f['confidence'] for f in fires) / len(fires),
        'max_brightness': max(f['brightness'] for f in fires),
        'total_frp': sum(f['frp'] for f in fires)
    }

