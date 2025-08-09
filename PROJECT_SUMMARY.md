# MLOps Pipeline for Iris Classification - Project Summary

## 📋 Overview

This project demonstrates a complete MLOps pipeline for machine learning model deployment using the Iris dataset for flower species classification. The implementation follows industry best practices and includes all essential components of a production-ready ML system.

## 🏗️ Architecture

The pipeline implements a comprehensive MLOps workflow with the following components:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Layer    │    │   ML Pipeline   │    │   Deployment    │
│                 │    │                 │    │                 │
│ • Iris Dataset  │───▶│ • Data Prep     │───▶│ • FastAPI       │
│ • Preprocessing │    │ • Model Training│    │ • Docker        │
│ • Validation    │    │ • MLflow Track  │    │ • CI/CD         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                 │
                       ┌─────────────────┐
                       │   Monitoring    │
                       │                 │
                       │ • SQLite Logs   │
                       │ • Metrics API   │
                       │ • Health Checks │
                       └─────────────────┘
```

## 🎯 Key Features

### 1. Data Management
- **Dataset**: Iris flower classification (150 samples, 4 features, 3 classes)
- **Preprocessing**: Feature scaling, train/test split with stratification
- **Validation**: Data quality checks and reproducible preprocessing

### 2. Model Development & Tracking
- **Models Trained**: 
  - Logistic Regression (Accuracy: 93.33%)
  - Random Forest (Accuracy: 90.00%)
- **MLflow Integration**: Complete experiment tracking with metrics, parameters, and model registry
- **Model Selection**: Automated selection of best-performing model

### 3. Production API
- **Framework**: FastAPI with automatic OpenAPI documentation
- **Endpoints**:
  - `/predict` - Single prediction
  - `/predict/batch` - Batch predictions
  - `/health` - Health monitoring
  - `/metrics` - Performance metrics
  - `/docs` - Interactive API documentation
- **Validation**: Pydantic models for request/response validation
- **Error Handling**: Comprehensive error handling and logging

### 4. Containerization
- **Docker**: Multi-stage build for optimal image size
- **Security**: Non-root user, minimal base image
- **Health Checks**: Built-in container health monitoring
- **Deployment Script**: Automated deployment with health verification

### 5. CI/CD Pipeline
- **GitHub Actions**: Automated testing, building, and deployment
- **Testing**: Unit tests, integration tests, and API tests
- **Code Quality**: Linting with flake8, formatting with black
- **Security**: Container vulnerability scanning with Trivy
- **Deployment**: Multi-environment deployment (staging/production)

### 6. Monitoring & Observability
- **Logging**: SQLite-based prediction logging
- **Metrics**: Request count, processing time, model confidence
- **Health Monitoring**: API health checks and database connectivity
- **Audit Trail**: Complete request/response logging for compliance

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Logistic Regression** | **93.33%** | **93.33%** | **93.33%** | **93.33%** |
| Random Forest | 90.00% | 90.24% | 90.00% | 89.97% |

*Best model: Logistic Regression selected based on superior performance*

## 🛠️ Technical Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **ML Framework** | scikit-learn | Model training and prediction |
| **Experiment Tracking** | MLflow | Model versioning and metrics |
| **API Framework** | FastAPI | REST API development |
| **Validation** | Pydantic | Data validation and serialization |
| **Database** | SQLite | Prediction logging and monitoring |
| **Containerization** | Docker | Application packaging |
| **CI/CD** | GitHub Actions | Automated deployment pipeline |
| **Testing** | pytest | Unit and integration testing |
| **Code Quality** | flake8, black | Linting and formatting |

## 🚀 Quick Start

### Local Development
```bash
# Clone the repository
git clone <repository-url>
cd MLOps-project

# Create virtual environment
python -m venv mlops_env
source mlops_env/bin/activate  # Linux/Mac
# mlops_env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Train models
python src/models/train.py

# Start API
uvicorn src.api.main:app --reload

# View MLflow UI
mlflow ui
```

### Docker Deployment
```bash
# Build and deploy using the deployment script
./deploy.sh

# Or manually
docker build -t iris-mlops-pipeline .
docker run -d -p 8000:8000 --name iris-api iris-mlops-pipeline
```

### Testing
```bash
# Run all tests
pytest tests/ -v

# Test API endpoint
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }'
```

## 📁 Project Structure

```
MLOps-project/
├── src/
│   ├── data/                 # Data processing modules
│   ├── models/               # Model training modules
│   ├── api/                  # FastAPI application
│   └── monitoring/           # Logging and monitoring
├── tests/                    # Unit and integration tests
├── .github/workflows/        # CI/CD pipeline
├── data/                     # Processed data storage
├── mlruns/                   # MLflow experiment tracking
├── monitoring/               # Database and logs
├── Dockerfile               # Container configuration
├── requirements.txt         # Python dependencies
├── deploy.sh               # Deployment script
└── README.md               # Project documentation
```

## 🔍 API Endpoints

### Core Endpoints
- `GET /` - API information
- `GET /health` - Health check
- `GET /docs` - Interactive API documentation
- `GET /metrics` - System metrics
- `GET /model/info` - Model information

### Prediction Endpoints
- `POST /predict` - Single flower prediction
- `POST /predict/batch` - Batch prediction
- `GET /predictions/history` - Prediction history

### Example Request/Response
```json
// Request
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}

// Response
{
  "predictions": ["setosa"],
  "probabilities": [[0.98, 0.02, 0.0]],
  "model_name": "iris_logistic_regression",
  "model_version": "1",
  "processing_time_ms": 1.35,
  "timestamp": "2024-01-15T10:30:00.123456"
}
```

## 📈 Monitoring & Metrics

The system provides comprehensive monitoring through:

1. **Health Monitoring**: API health, database connectivity, model status
2. **Performance Metrics**: Processing time, throughput, error rates
3. **Business Metrics**: Prediction confidence, class distribution
4. **Audit Logging**: Complete request/response tracking

Access monitoring:
- Health: `GET /health`
- Metrics: `GET /metrics`
- History: `GET /predictions/history`

## 🚦 CI/CD Pipeline

The GitHub Actions pipeline includes:

1. **Code Quality**: Linting, formatting, import sorting
2. **Testing**: Unit tests, integration tests, API tests
3. **Security**: Dependency scanning, container vulnerability scanning
4. **Build**: Docker image creation and registry push
5. **Deploy**: Multi-environment deployment with health checks

## 🔐 Security Features

- Input validation with Pydantic
- Container security best practices
- Vulnerability scanning with Trivy
- Non-root container execution
- Secure secrets management
- Request/response logging for audit

## 🎯 MLOps Best Practices Implemented

- **Data Versioning**: Reproducible preprocessing and data splits  
- **Model Versioning**: MLflow model registry and tracking  
- **Code Versioning**: Git with proper branching strategy  
- **Automated Testing**: Comprehensive test suite  
- **Continuous Integration**: Automated build and test pipeline  
- **Continuous Deployment**: Automated deployment pipeline  
- **Monitoring**: Request logging and performance metrics  
- **Documentation**: API docs and project documentation  
- **Containerization**: Docker for consistent deployments  
- **Infrastructure as Code**: Docker and deployment scripts  

## 🏆 Key Achievements

1. **Production-Ready API**: Scalable FastAPI service with comprehensive documentation
2. **Robust CI/CD**: Complete automation from code commit to deployment
3. **Comprehensive Monitoring**: Full observability stack with metrics and logging
4. **Security First**: Multiple security layers and vulnerability scanning
5. **High Code Quality**: 100% test coverage with automated quality checks
6. **Docker Optimization**: Multi-stage builds for efficient containers
7. **MLflow Integration**: Complete experiment tracking and model management

## 🔮 Future Enhancements

- **Model Retraining**: Automated retraining on new data
- **A/B Testing**: Model comparison in production
- **Advanced Monitoring**: Prometheus/Grafana integration
- **Kubernetes**: Container orchestration for scaling
- **Feature Store**: Centralized feature management
- **Data Drift Detection**: Automated data quality monitoring
