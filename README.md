# Real-Time Predictive System (RPS) - MLOps Pipeline

A comprehensive MLOps pipeline for real-time weather forecasting with automated data ingestion, model training, deployment, and monitoring.

## Project Overview

This project implements a complete MLOps pipeline that:
- Ingests live weather data from OpenWeatherMap API
- Trains time-series forecasting models to predict temperature 4-6 hours ahead
- Deploys models via REST API with Docker
- Monitors model performance and data drift in real-time
- Automates CI/CD with GitHub Actions and CML

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Airflow    │────▶│  Data Store  │────▶│   MLflow    │
│   (ETL)     │     │  (DVC/S3)    │     │ (Tracking)  │
└─────────────┘     └──────────────┘     └─────────────┘
                                              │
                                              ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  FastAPI    │────▶│  Prometheus  │────▶│   Grafana   │
│  (Service)  │     │  (Metrics)   │     │ (Dashboard) │
└─────────────┘     └──────────────┘     └─────────────┘
```

## Setup Instructions

### 1. Prerequisites

- Python 3.9+
- Docker and Docker Compose
- OpenWeatherMap API key (free tier available)
- Dagshub account
- Git

### 2. Environment Configuration

```bash
cp .env.example .env
# Edit .env with your API keys and credentials
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Initialize DVC

```bash
dvc init
dvc remote add -d storage s3://your-bucket-name
```

### 5. Start Services with Docker Compose

```bash
docker-compose up -d
```

This will start:
- Apache Airflow (http://localhost:8080)
- FastAPI service (http://localhost:8000)
- Prometheus (http://localhost:9090)
- Grafana (http://localhost:3000)

### 6. Access Services

- **Airflow UI**: http://localhost:8080 (admin/admin)
- **FastAPI Docs**: http://localhost:8000/docs
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

## Project Structure

```
.
├── dags/                    # Airflow DAGs
├── scripts/                 # Data processing and training scripts
│   ├── extract.py          # API data extraction
│   ├── transform.py        # Feature engineering
│   ├── train.py            # Model training with MLflow
│   └── quality_check.py    # Data quality validation
├── src/                     # Application source code
│   ├── api/                # FastAPI application
│   ├── models/             # Model loading utilities
│   └── monitoring/         # Prometheus metrics
├── data/                    # Data storage (versioned with DVC)
│   ├── raw/                # Raw API data
│   └── processed/          # Processed datasets
├── docker/                  # Docker configurations
├── .github/workflows/       # GitHub Actions CI/CD
└── monitoring/              # Prometheus and Grafana configs
```

## Workflow

1. **Data Ingestion**: Airflow DAG runs daily, fetches weather data
2. **Quality Check**: Validates data quality before processing
3. **Feature Engineering**: Creates time-series features (lags, rolling means)
4. **Model Training**: Trains model and logs to MLflow
5. **Model Serving**: FastAPI serves predictions
6. **Monitoring**: Prometheus collects metrics, Grafana visualizes

## CI/CD Pipeline

- **Feature → dev**: Code quality checks and unit tests
- **dev → test**: Model retraining with CML comparison
- **test → master**: Production deployment with Docker

## Monitoring

- API latency and request counts
- Model prediction distributions
- Data drift detection
- Alerting on performance degradation



