# Real-Time Predictive System (RPS) - MLOps Pipeline

A comprehensive, production-ready MLOps pipeline for real-time weather forecasting with automated data ingestion, model training, deployment, and monitoring.

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/huda515k/MLOps-Pipeline)
[![Python](https://img.shields.io/badge/Python-3.9+-green)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://www.docker.com/)

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Detailed Setup](#detailed-setup)
- [Project Structure](#project-structure)
- [Workflow](#workflow)
- [CI/CD Pipeline](#cicd-pipeline)
- [Monitoring & Observability](#monitoring--observability)
- [API Documentation](#api-documentation)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Project Overview

This project implements a complete, end-to-end MLOps pipeline that demonstrates industry best practices for:

- **Automated Data Ingestion**: Daily extraction from OpenWeatherMap API with quality gates
- **Feature Engineering**: Time-series feature creation (lags, rolling means, cyclical encodings)
- **Model Training**: Automated training with MLflow experiment tracking
- **Model Deployment**: Containerized REST API service with Docker
- **Continuous Monitoring**: Real-time metrics, data drift detection, and alerting
- **CI/CD Automation**: GitHub Actions workflows with CML for model comparison

### Predictive Task

**Domain**: Environmental (Weather Forecasting)  
**API**: OpenWeatherMap  
**Goal**: Predict temperature 4-6 hours ahead for a specific city (London, UK)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        MLOps Pipeline                            │
└─────────────────────────────────────────────────────────────────┘

┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Airflow    │────▶│  MinIO/S3    │────▶│   MLflow    │
│   (ETL)     │     │  (DVC Store) │     │ (Tracking)  │
│  Daily DAG  │     │              │     │  Dagshub    │
└─────────────┘     └──────────────┘     └─────────────┘
       │                    │                     │
       │                    │                     │
       ▼                    ▼                     ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  FastAPI   │────▶│  Prometheus  │────▶│   Grafana   │
│  (Service) │     │  (Metrics)   │     │ (Dashboard) │
│  Port 8000 │     │  Port 9090   │     │  Port 3000  │
└─────────────┘     └──────────────┘     └─────────────┘
```

## ✨ Features

### Phase I: Data Ingestion
- ✅ Automated daily data extraction from OpenWeatherMap API
- ✅ Strict data quality gates (>1% null check, schema validation, range checks)
- ✅ Comprehensive Pandas Profiling reports logged to MLflow/Dagshub
- ✅ Data versioning with DVC (MinIO/S3 storage)
- ✅ Feature engineering: lag features, rolling statistics, time encodings

### Phase II: Model Management
- ✅ MLflow experiment tracking (hyperparameters, metrics, artifacts)
- ✅ Dagshub integration (centralized Git + DVC + MLflow)
- ✅ Model registry for production deployments
- ✅ Automated model comparison with CML

### Phase III: CI/CD
- ✅ Strict branching model (dev → test → master)
- ✅ PR approval requirements for test and master branches
- ✅ Automated code quality checks (Black, Flake8)
- ✅ Unit test framework
- ✅ Model retraining tests with CML reporting
- ✅ Automated Docker image building and deployment
- ✅ Health check verification in CD pipeline

### Phase IV: Monitoring
- ✅ Prometheus metrics collection
- ✅ Real-time Grafana dashboards
- ✅ Service metrics (latency, request count)
- ✅ Data drift detection and alerting
- ✅ 5 configured alerts (latency, drift, errors, service down, no predictions)

## 🛠️ Tech Stack

| Category | Tools | Purpose |
|----------|-------|---------|
| **Orchestration** | Apache Airflow 2.7.3 | ETL and training workflow automation |
| **Data Versioning** | DVC 3.38+ | Version control for large datasets |
| **Storage** | MinIO | S3-compatible object storage for DVC |
| **Experiment Tracking** | MLflow 2.9+ | Model experiment tracking and registry |
| **Centralized Hub** | Dagshub | Git + DVC + MLflow integration |
| **API Framework** | FastAPI | REST API for model serving |
| **Monitoring** | Prometheus | Metrics collection |
| **Visualization** | Grafana | Dashboards and alerting |
| **CI/CD** | GitHub Actions | Automated testing and deployment |
| **ML Reporting** | CML | Model comparison in PRs |
| **Containerization** | Docker | Service containerization |

## 📦 Prerequisites

- **Python**: 3.9 or higher
- **Docker**: 20.10+ and Docker Compose 2.0+
- **Git**: For version control
- **OpenWeatherMap API Key**: [Get free API key](https://openweathermap.org/api)
- **Dagshub Account**: [Sign up for free](https://dagshub.com/)
- **Docker Hub Account**: For container registry (optional, for CD pipeline)

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/huda515k/MLOps-Pipeline.git
cd MLOps-Pipeline
```

### 2. Set Up Environment Variables

Create a `.env` file in the root directory:

```bash
# OpenWeatherMap API
OPENWEATHER_API_KEY=your_api_key_here

# Dagshub Configuration
DAGSHUB_USERNAME=your_username
DAGSHUB_REPO=your_repo_name
DAGSHUB_TOKEN=your_token

# MLflow (optional - defaults to local)
MLFLOW_TRACKING_URI=http://mlflow:5000

# Docker Hub (for CD pipeline)
DOCKER_USERNAME=your_dockerhub_username
DOCKER_PASSWORD=your_dockerhub_token
```

### 3. Start All Services

```bash
docker-compose up -d
```

This will start:
- **Apache Airflow** (Web UI + Scheduler)
- **FastAPI Service** (Model serving)
- **Prometheus** (Metrics collection)
- **Grafana** (Dashboards)
- **MLflow** (Experiment tracking)
- **MinIO** (Object storage)
- **PostgreSQL** (Airflow and MLflow databases)

### 4. Access Services

| Service | URL | Credentials |
|---------|-----|------------|
| **Airflow UI** | http://localhost:8080 | admin / admin |
| **FastAPI Docs** | http://localhost:8000/docs | - |
| **Prometheus** | http://localhost:9090 | - |
| **Grafana** | http://localhost:3000 | admin / admin |
| **MLflow UI** | http://localhost:5001 | - |
| **MinIO Console** | http://localhost:9001 | minioadmin / minioadmin |

## 📖 Detailed Setup

### Initialize DVC

```bash
# Initialize DVC
dvc init

# Configure MinIO as remote storage
dvc remote add storage s3://mlops-weather-data
dvc remote modify storage endpointurl http://localhost:9000
dvc remote modify storage access_key_id minioadmin
dvc remote modify storage secret_access_key minioadmin

# Set as default remote
dvc remote default storage
```

### Install Local Dependencies (Optional)

For local development without Docker:

```bash
pip install -r requirements-optimized.txt
```

### Trigger Airflow DAG

1. Open Airflow UI: http://localhost:8080
2. Login with `admin/admin`
3. Find `weather_forecast_pipeline` DAG
4. Click "Play" button to trigger manually
5. Monitor execution in the Graph View

## 📁 Project Structure

```
MLOps-Pipeline/
├── dags/                          # Airflow DAGs
│   └── weather_forecast_pipeline.py
├── scripts/                       # Data processing scripts
│   ├── extract.py                # OpenWeatherMap API extraction
│   ├── transform.py              # Feature engineering
│   ├── train.py                  # Model training with MLflow
│   ├── quality_check.py          # Data quality validation
│   └── test_dagshub_logging.py   # Dagshub integration test
├── src/                           # Application source
│   └── api/
│       └── main.py               # FastAPI service with Prometheus
├── docker/                        # Docker configurations
│   └── Dockerfile.fastapi.optimized
├── monitoring/                    # Monitoring configs
│   ├── prometheus/
│   │   └── prometheus.yml
│   └── grafana/
│       ├── provisioning/
│       │   ├── datasources/
│       │   ├── dashboards/
│       │   └── alerting/
│       └── dashboards/
│           └── weather-forecast-dashboard.json
├── .github/workflows/            # CI/CD pipelines
│   ├── ci-dev.yml               # Feature → dev
│   ├── ci-test.yml              # dev → test (with CML)
│   └── cd-master.yml            # test → master (deployment)
├── data/                         # Data storage (DVC versioned)
│   ├── raw/                     # Raw API data
│   └── processed/               # Processed datasets
├── models/                       # Trained models
├── docker-compose.yml            # All services configuration
├── Dockerfile.airflow.optimized  # Airflow container
├── requirements-optimized.txt    # Full dependencies (Airflow)
├── requirements-minimal.txt     # Minimal dependencies (FastAPI)
└── README.md                     # This file
```

## 🔄 Workflow

### Daily Pipeline Execution

1. **Extraction** (`extract_data`)
   - Fetches current weather from OpenWeatherMap API
   - Saves raw data with timestamp to `data/raw/`

2. **Quality Check** (`check_quality`)
   - Validates null values (<1% threshold)
   - Checks schema (required columns, data types)
   - Validates value ranges (temp, humidity, pressure, wind)
   - **Fails DAG if quality check fails**

3. **Transformation** (`transform_data`)
   - Cleans and engineers features:
     - Lag features (1h, 2h, 3h, 6h, 12h)
     - Rolling statistics (mean, std, min, max)
     - Time encodings (hour, day, month, cyclical)
     - Derived features (heat index, wind chill)
   - Saves processed data to `data/processed/`

4. **Profiling Report** (`generate_profiling_report`)
   - Generates comprehensive HTML report using ydata-profiling
   - Logs report as artifact to MLflow/Dagshub
   - Logs metadata (shape, quality score, collection date)

5. **Model Training** (`train_model`)
   - Trains Random Forest or Gradient Boosting model
   - Logs hyperparameters, metrics (RMSE, MAE, R²) to MLflow
   - Saves model artifact to MLflow Model Registry
   - Saves model locally to `models/`

6. **Data Versioning** (`version_data`)
   - Versions processed dataset with DVC
   - Pushes to MinIO remote storage
   - Commits `.dvc` metadata file to Git

## 🔀 CI/CD Pipeline

### Branch Strategy

```
feature-branch → dev → test → master
```

### Workflows

#### 1. Feature → Dev (`ci-dev.yml`)
- **Trigger**: PR to `dev` branch
- **Actions**:
  - Code formatting check (Black)
  - Linting (Flake8)
  - Unit tests (pytest)

#### 2. Dev → Test (`ci-test.yml`)
- **Trigger**: PR to `test` branch
- **Actions**:
  - Model retraining test
  - CML comparison report (compares new model vs production)
  - Posts metrics to PR comments
  - **Blocks merge if new model performs worse**

#### 3. Test → Master (`cd-master.yml`)
- **Trigger**: Push to `master` branch
- **Actions**:
  - Fetches best model from MLflow Model Registry
  - Builds Docker image
  - Pushes to Docker Hub
  - **Verifies deployment** (runs container, checks health endpoint)
  - Fails workflow if health check fails

### PR Approval Requirements

Configure branch protection rules in GitHub:
1. Go to **Settings → Branches**
2. Add rule for `test` branch:
   - ✅ Require pull request before merging
   - ✅ Require approvals: **1**
3. Add rule for `master` branch:
   - ✅ Require pull request before merging
   - ✅ Require approvals: **1**

## 📊 Monitoring & Observability

### Prometheus Metrics

The FastAPI service exposes metrics at `/metrics`:

- `api_requests_total`: Total API requests (by method, endpoint)
- `api_request_duration_seconds`: Request latency histogram
- `predictions_total`: Total predictions made
- `prediction_errors_total`: Total prediction errors
- `data_drift_ratio`: Ratio of requests with out-of-distribution features

### Grafana Dashboard

**URL**: http://localhost:3000

**Panels**:
- Request Rate (requests/second)
- Request Latency (95th percentile)
- Total Predictions
- Prediction Errors
- Data Drift Ratio
- Request Latency Distribution

**Refresh**: 10 seconds (real-time)

### Alerts

Configured alerts in Grafana:

1. **High Inference Latency** (Warning)
   - Threshold: >500ms for 2 minutes
   - Notification: File logger

2. **Data Drift Detected** (Critical)
   - Threshold: >50% drift ratio for 5 minutes
   - Action: Retrain model

3. **High Prediction Error Rate** (Warning)
   - Threshold: >10% error rate for 3 minutes

4. **Service Down** (Critical)
   - Threshold: Service unavailable for 1 minute
   - Action: Restart service

5. **No Predictions Received** (Info)
   - Threshold: No predictions for 10 minutes

Alert logs: `/var/log/grafana/alerts.log` (in Grafana container)

## 🔌 API Documentation

### FastAPI Service

**Base URL**: http://localhost:8000

### Endpoints

#### 1. Health Check
```http
GET /health
```

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2025-11-30T12:00:00"
}
```

#### 2. Make Prediction
```http
POST /predict
Content-Type: application/json

{
  "temp": 15.5,
  "humidity": 65.0,
  "pressure": 1013.25,
  "wind_speed": 5.2,
  "wind_deg": 180.0,
  "clouds": 20.0,
  "hour": 14,
  "day_of_week": 1,
  "month": 11
}
```

**Response**:
```json
{
  "predicted_temp": 16.2,
  "forecast_hours": 4,
  "timestamp": "2025-11-30T12:00:00",
  "data_drift_detected": false
}
```

#### 3. Prometheus Metrics
```http
GET /metrics
```

Returns Prometheus-formatted metrics.

#### 4. API Documentation
```http
GET /docs
```

Interactive Swagger UI documentation.

### Example Usage

```bash
# Health check
curl http://localhost:8000/health

# Make prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "temp": 15.5,
    "humidity": 65.0,
    "pressure": 1013.25,
    "wind_speed": 5.2
  }'
```

## 🐛 Troubleshooting

### Services Not Starting

```bash
# Check service status
docker-compose ps

# View logs
docker-compose logs -f [service-name]

# Restart specific service
docker-compose restart [service-name]
```

### Airflow DAG Not Appearing

1. Check DAG file syntax: `python dags/weather_forecast_pipeline.py`
2. Verify DAG folder is mounted: `docker-compose.yml` volumes
3. Check Airflow logs: `docker-compose logs airflow-scheduler`

### MLflow Not Connecting

1. Verify MLflow service is running: `docker-compose ps mlflow`
2. Check MLFLOW_TRACKING_URI in `.env`
3. For Dagshub: Verify credentials (USERNAME, REPO, TOKEN)

### DVC Push Fails

1. Ensure MinIO is running: `docker-compose ps minio`
2. Verify DVC remote: `dvc remote list`
3. Check credentials: `dvc remote modify storage access_key_id minioadmin`

### Port Conflicts

If ports are already in use:

```bash
# Stop all services
docker-compose down

# Free ports
lsof -ti:8080 | xargs kill -9  # Airflow
lsof -ti:8000 | xargs kill -9  # FastAPI
lsof -ti:9090 | xargs kill -9  # Prometheus
lsof -ti:3000 | xargs kill -9  # Grafana
```

## 🤝 Contributing

1. Create a feature branch from `dev`
2. Make your changes
3. Ensure code quality checks pass
4. Create PR to `dev` branch
5. After approval, merge to `dev`
6. Follow the branching model: `dev → test → master`

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- OpenWeatherMap for free weather API
- Dagshub for integrated ML platform
- Apache Airflow, MLflow, and all open-source contributors

---

**Repository**: [https://github.com/huda515k/MLOps-Pipeline](https://github.com/huda515k/MLOps-Pipeline)

**Last Updated**: November 2025
