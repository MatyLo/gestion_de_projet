#!/bin/bash

# Fire Prediction Frontend Startup Script
echo "🔥 Starting Fire Prediction Frontend..."

cd frontend

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Start the frontend
echo "🚀 Starting Streamlit app..."
echo "🌐 App will open at http://localhost:8501"
streamlit run app.py

