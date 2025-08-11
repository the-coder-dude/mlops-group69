# MLOps Pipeline for Iris Classification

This project demonstrates a complete MLOps pipeline using the Iris dataset for flower species classification.

## 🏗️ Architecture

- **Data**: Iris dataset (classification)
- **ML Framework**: scikit-learn
- **Experiment Tracking**: MLflow
- **API**: FastAPI
- **Containerization**: Docker
- **CI/CD**: GitHub Actions
- **Monitoring**: Custom logging with SQLite
- **Validation**: Pydantic

## 📁 Project Structure

```
├── data/                   # Dataset storage
├── src/
│   ├── data/              # Data processing modules
│   ├── models/            # Model training modules
│   ├── api/               # FastAPI application
│   └── monitoring/        # Logging and monitoring
├── notebooks/             # Jupyter notebooks for exploration
├── tests/                 # Unit tests
├── docker/                # Docker configuration
├── .github/workflows/     # GitHub Actions CI/CD
├── mlruns/               # MLflow tracking data
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
└── README.md             # This file
```

## 🚀 Quick Start

1. **Setup Environment**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train Models**
   ```bash
   python src/models/train.py
   ```

3. **Start API**
   ```bash
   uvicorn src.api.main:app --reload
   ```

4. **Build Docker Image**
   ```bash
   docker build -t iris-mlops-pipeline .
   ```

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Logistic Regression** | **93.33%** | **93.33%** | **93.33%** | **93.33%** |
| Random Forest | 90.00% | 90.24% | 90.00% | 89.97% |

*Best model: Logistic Regression automatically selected based on performance*

## 🔍 Monitoring

- Prediction requests are logged to SQLite database
- Metrics endpoint available at `/metrics`
- Request/response logging for audit trail

## 🧪 Testing

```bash
pytest tests/
```

## 📈 MLflow UI

```bash
mlflow ui
```

## 🐳 Docker Deployment

```bash
docker run -p 8000:8000 iris-mlops-pipeline
```

## 📝 API Documentation

Once running, visit: http://localhost:8000/docs
# GitHub Secrets configured successfully
