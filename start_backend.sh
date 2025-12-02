#!/bin/bash

# Fire Prediction Backend Startup Script
echo "🔥 Starting Fire Prediction Backend..."

cd backend

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Start the backend
echo "🚀 Starting FastAPI backend on http://localhost:8000"
echo "📖 API Docs available at http://localhost:8000/docs"
python main.py

