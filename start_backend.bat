@echo off
REM Fire Prediction Backend Startup Script (Windows)
echo 🔥 Starting Fire Prediction Backend...

cd backend

REM Check if virtual environment exists
if not exist "venv\" (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo 🔄 Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo 📥 Installing dependencies...
pip install -r requirements.txt

REM Start the backend
echo 🚀 Starting FastAPI backend on http://localhost:8000
echo 📖 API Docs available at http://localhost:8000/docs
python main.py

