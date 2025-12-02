@echo off
REM Fire Prediction Frontend Startup Script (Windows)
echo 🔥 Starting Fire Prediction Frontend...

cd frontend

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

REM Start the frontend
echo 🚀 Starting Streamlit app...
echo 🌐 App will open at http://localhost:8501
streamlit run app.py

