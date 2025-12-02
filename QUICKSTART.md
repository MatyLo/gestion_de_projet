# 🚀 Quick Start Guide

Get the Fire Prediction System up and running in 2 minutes!

## Prerequisites

- **Option 1 (Docker):** Docker & Docker Compose
- **Option 2/3 (Manual):** Python 3.11+ & pip

## Option 1: Using Docker (Easiest! 🐳)

```bash
docker-compose up --build
```

That's it! Wait for the containers to build and start.

**Access the app:**
- Frontend: http://localhost:8501
- Backend API: http://localhost:8000/docs

**Stop the app:**
```bash
docker-compose down
```

See [DOCKER.md](DOCKER.md) for more Docker commands.

---

## Option 2: Using Startup Scripts

### On macOS/Linux:

**Terminal 1 - Backend:**
```bash
./start_backend.sh
```

**Terminal 2 - Frontend:**
```bash
./start_frontend.sh
```

### On Windows:

**Terminal 1 - Backend:**
```cmd
start_backend.bat
```

**Terminal 2 - Frontend:**
```cmd
start_frontend.bat
```

## Option 3: Manual Setup

### Backend Setup

```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

### Frontend Setup

Open a new terminal:

```bash
cd frontend
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Access the Application

- **Frontend (Streamlit):** http://localhost:8501
- **Backend API:** http://localhost:8000
- **API Documentation:** http://localhost:8000/docs

## First Steps

1. **Open the Streamlit app** at http://localhost:8501
2. **Choose input method:**
   - Click on the map to select a location, OR
   - Enter latitude/longitude manually
3. **Adjust fire brightness** (optional, default: 350)
4. **Click "🚀 Predict Fire Spread"**
5. **View results** in the right panel and on the map

## Example Coordinates to Try

- **California (Wildfire Prone):** Lat: 34.05, Lng: -118.25
- **Australia Outback:** Lat: -25.27, Lng: 133.77
- **Mediterranean Region:** Lat: 37.98, Lng: 23.73
- **Western US:** Lat: 45.52, Lng: -122.68

## Troubleshooting

### Backend won't start?
- Check Python version: `python --version` (needs 3.11+)
- Ensure models exist in `backend/ml/` directory
- Check port 8000 is not in use

### Frontend can't connect to backend?
- Ensure backend is running at http://localhost:8000
- Check API health: http://localhost:8000/health
- Verify no firewall blocking localhost connections

### API requests timing out?
- First request may take longer (loading models)
- External weather API calls may take 2-5 seconds
- Check internet connection for weather data

## Need Help?

- Check the main [README.md](README.md) for detailed documentation
- Review API docs at http://localhost:8000/docs
- Check backend logs in the terminal where backend is running

---

**Happy Fire Prediction! 🔥**

