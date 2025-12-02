# 🐳 Docker Deployment Guide

## Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+

Check your versions:
```bash
docker --version
docker-compose --version
```

## Quick Start with Docker

### 🚀 One Command Deployment

```bash
docker-compose up --build
```

This will:
1. Build both backend and frontend Docker images
2. Start the backend on port 8000
3. Start the frontend on port 8501
4. Configure networking between services
5. Set up health checks

### 🌐 Access the Application

- **Frontend (Streamlit):** http://localhost:8501
- **Backend API:** http://localhost:8000
- **API Documentation:** http://localhost:8000/docs

## Docker Commands

### Build and Start

```bash
# Build and start in foreground (see logs)
docker-compose up --build

# Build and start in background (detached mode)
docker-compose up -d --build

# Start without rebuilding
docker-compose up -d
```

### Stop and Remove

```bash
# Stop services
docker-compose stop

# Stop and remove containers
docker-compose down

# Stop, remove containers, and remove volumes
docker-compose down -v

# Stop, remove everything including images
docker-compose down --rmi all
```

### View Logs

```bash
# View all logs
docker-compose logs

# View logs in real-time
docker-compose logs -f

# View specific service logs
docker-compose logs backend
docker-compose logs frontend

# Follow specific service logs
docker-compose logs -f backend
```

### Check Status

```bash
# List running containers
docker-compose ps

# Check container health
docker ps
```

### Restart Services

```bash
# Restart all services
docker-compose restart

# Restart specific service
docker-compose restart backend
docker-compose restart frontend
```

## Architecture

### Container Structure

```
┌─────────────────────────────────────┐
│  Docker Host                        │
│                                     │
│  ┌──────────────────────────────┐  │
│  │  Frontend Container           │  │
│  │  - Streamlit app              │  │
│  │  - Port: 8501                 │  │
│  │  - Connects to: backend:8000  │  │
│  └──────────────────────────────┘  │
│              │                      │
│              │ HTTP                 │
│              ▼                      │
│  ┌──────────────────────────────┐  │
│  │  Backend Container            │  │
│  │  - FastAPI app                │  │
│  │  - Port: 8000                 │  │
│  │  - ML Models loaded           │  │
│  └──────────────────────────────┘  │
│                                     │
│  Network: fire-prediction-network   │
└─────────────────────────────────────┘
```

### Environment Variables

The `docker-compose.yml` configures:
- `API_URL=http://backend:8000` - Frontend connects to backend using Docker service name

## Dockerfile Details

### Backend Dockerfile

```dockerfile
FROM python:3.11-slim
- Installs Python dependencies
- Copies FastAPI app and ML models
- Exposes port 8000
- Health check on /health endpoint
```

### Frontend Dockerfile

```dockerfile
FROM python:3.11-slim
- Installs Python dependencies
- Copies Streamlit app
- Configures Streamlit for Docker
- Exposes port 8501
- Health check on Streamlit health endpoint
```

## Health Checks

Both services include health checks:

**Backend:**
- Endpoint: `http://localhost:8000/health`
- Interval: 30 seconds
- Start period: 10 seconds

**Frontend:**
- Endpoint: `http://localhost:8501/_stcore/health`
- Interval: 30 seconds
- Depends on backend being healthy

## Troubleshooting

### Port Already in Use

If ports 8000 or 8501 are already in use:

```bash
# Find process using the port
lsof -i :8000
lsof -i :8501

# Or modify docker-compose.yml
ports:
  - "8080:8000"  # Use port 8080 instead of 8000
  - "8502:8501"  # Use port 8502 instead of 8501
```

### Container Won't Start

```bash
# Check logs
docker-compose logs backend
docker-compose logs frontend

# Rebuild from scratch
docker-compose down
docker-compose build --no-cache
docker-compose up
```

### Frontend Can't Connect to Backend

1. Check backend is healthy:
   ```bash
   docker-compose ps
   curl http://localhost:8000/health
   ```

2. Check network connectivity:
   ```bash
   docker-compose exec frontend ping backend
   ```

3. Verify environment variable:
   ```bash
   docker-compose exec frontend env | grep API_URL
   ```

### Models Not Loading

Ensure the ML models are present:
```bash
ls backend/ml/*.joblib
```

You should see:
- `wildfire_spread_classifier_advanced.joblib`
- `wildfire_spread_regressor_advanced.joblib`

## Production Deployment

### Using Docker Hub

```bash
# Tag images
docker tag fire-prediction-system-backend:latest yourusername/fire-prediction-backend:latest
docker tag fire-prediction-system-frontend:latest yourusername/fire-prediction-frontend:latest

# Push to Docker Hub
docker push yourusername/fire-prediction-backend:latest
docker push yourusername/fire-prediction-frontend:latest
```

### Using a Cloud Provider

#### AWS ECS
```bash
# Install AWS CLI and ECS CLI
# Configure AWS credentials
ecs-cli compose up
```

#### Google Cloud Run
```bash
# Build and push
gcloud builds submit --tag gcr.io/PROJECT-ID/fire-prediction-backend backend/
gcloud builds submit --tag gcr.io/PROJECT-ID/fire-prediction-frontend frontend/

# Deploy
gcloud run deploy backend --image gcr.io/PROJECT-ID/fire-prediction-backend
gcloud run deploy frontend --image gcr.io/PROJECT-ID/fire-prediction-frontend
```

#### Azure Container Instances
```bash
# Build and push to ACR
az acr build --registry myregistry --image fire-prediction-backend backend/
az acr build --registry myregistry --image fire-prediction-frontend frontend/

# Deploy
az container create --resource-group mygroup --file docker-compose.yml
```

## Resource Management

### Limit Resources

Add to `docker-compose.yml`:

```yaml
services:
  backend:
    deploy:
      resources:
        limits:
          cpus: '1'
          memory: 1G
        reservations:
          memory: 512M
  
  frontend:
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 512M
        reservations:
          memory: 256M
```

### View Resource Usage

```bash
docker stats
```

## Security Best Practices

1. **Don't run as root** - Add to Dockerfiles:
   ```dockerfile
   RUN adduser --disabled-password --gecos '' appuser
   USER appuser
   ```

2. **Scan for vulnerabilities:**
   ```bash
   docker scan fire-prediction-system-backend
   docker scan fire-prediction-system-frontend
   ```

3. **Use specific versions** - Already done in requirements.txt

4. **Keep images updated:**
   ```bash
   docker-compose pull
   docker-compose up -d
   ```

## Backup and Data Persistence

Since this application is stateless (no database), no data persistence is needed. However, if you add data storage:

```yaml
services:
  backend:
    volumes:
      - ./data:/app/data
```

## Performance Optimization

### Multi-stage Builds

For smaller images, use multi-stage builds in Dockerfiles.

### Layer Caching

- Requirements are copied first (changes less frequently)
- Application code copied last (changes more frequently)
- This optimizes Docker layer caching

### Image Size

Current images use `python:3.11-slim` for smaller size:
- Base image: ~50MB
- With dependencies: ~150-200MB per service

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Docker Build and Push

on:
  push:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Build images
        run: docker-compose build
      - name: Run tests
        run: docker-compose up -d && sleep 10 && curl http://localhost:8000/health
```

---

## Summary

**Start:** `docker-compose up -d --build`
**Stop:** `docker-compose down`
**Logs:** `docker-compose logs -f`
**Access:** http://localhost:8501

🐳 **Happy Dockerizing!**

