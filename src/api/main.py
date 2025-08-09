"""
FastAPI application for Iris classification model serving.
"""

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
from datetime import datetime, timedelta
import sys
import os
from pathlib import Path
from typing import List
import asyncio

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from api.models import (
    IrisFeatures, 
    PredictionRequest, 
    PredictionResponse, 
    HealthResponse, 
    MetricsResponse,
    PredictionHistory
)
from api.model_loader import model_loader
from monitoring.database import prediction_logger

# Initialize FastAPI app
app = FastAPI(
    title="Iris Classification MLOps API",
    description="A production-ready API for Iris flower species classification with MLOps best practices",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for tracking
app_start_time = datetime.utcnow()


@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup."""
    print("🚀 Starting Iris Classification API...")
    
    # Load the model
    success = model_loader.load_best_model()
    if not success:
        print("❌ Failed to load model during startup")
    else:
        print("✅ Model loaded successfully")
    
    print("✅ API is ready to serve predictions")


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Iris Classification MLOps API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    # Check database connectivity
    try:
        stats = prediction_logger.get_statistics(days=1)
        db_connected = True
    except Exception:
        db_connected = False
    
    return HealthResponse(
        status="healthy" if model_loader.is_loaded() and db_connected else "unhealthy",
        timestamp=datetime.utcnow(),
        model_loaded=model_loader.is_loaded(),
        database_connected=db_connected
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(
    features: IrisFeatures,
    request: Request
):
    """
    Predict species for a single Iris flower.
    
    Args:
        features: Iris flower measurements
        request: FastAPI request object for metadata
        
    Returns:
        Prediction response with species and probabilities
    """
    if not model_loader.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Convert features to array format
        feature_array = [features.to_array()]
        
        # Make prediction
        predictions, probabilities = model_loader.predict(feature_array)
        
        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Get client information
        client_ip = request.client.host if request.client else None
        user_agent = request.headers.get("user-agent")
        
        # Log the prediction
        confidence = max(probabilities[0]) if probabilities else None
        prediction_logger.log_prediction(
            model_name=model_loader.model_name,
            model_version=model_loader.model_version,
            input_features=features.dict(),
            prediction=predictions[0],
            confidence=confidence,
            processing_time_ms=processing_time_ms,
            client_ip=client_ip,
            user_agent=user_agent
        )
        
        return PredictionResponse(
            predictions=predictions,
            probabilities=probabilities,
            model_name=model_loader.model_name,
            model_version=model_loader.model_version,
            processing_time_ms=processing_time_ms,
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=PredictionResponse)
async def predict_batch(
    request_data: PredictionRequest,
    request: Request
):
    """
    Predict species for multiple Iris flowers.
    
    Args:
        request_data: Batch prediction request with multiple features
        request: FastAPI request object for metadata
        
    Returns:
        Prediction response with species and probabilities for all inputs
    """
    if not model_loader.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Convert features to array format
        feature_arrays = [features.to_array() for features in request_data.features]
        
        # Make predictions
        predictions, probabilities = model_loader.predict(feature_arrays)
        
        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Get client information
        client_ip = request.client.host if request.client else None
        user_agent = request.headers.get("user-agent")
        
        # Log each prediction
        for i, (pred, prob, features) in enumerate(zip(predictions, probabilities, request_data.features)):
            confidence = max(prob) if prob else None
            prediction_logger.log_prediction(
                model_name=model_loader.model_name,
                model_version=model_loader.model_version,
                input_features=features.dict(),
                prediction=pred,
                confidence=confidence,
                processing_time_ms=processing_time_ms / len(predictions),
                client_ip=client_ip,
                user_agent=user_agent
            )
        
        return PredictionResponse(
            predictions=predictions,
            probabilities=probabilities,
            model_name=model_loader.model_name,
            model_version=model_loader.model_version,
            processing_time_ms=processing_time_ms,
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get API and model metrics."""
    try:
        # Get prediction statistics
        stats = prediction_logger.get_statistics(days=30)
        stats_24h = prediction_logger.get_statistics(days=1)
        
        # Calculate uptime
        uptime = (datetime.utcnow() - app_start_time).total_seconds() / 3600
        
        return MetricsResponse(
            total_predictions=stats['total_predictions'],
            avg_processing_time_ms=stats['avg_processing_time_ms'],
            avg_confidence=stats['avg_confidence'],
            predictions_last_24h=stats_24h['total_predictions'],
            uptime_hours=uptime
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


@app.get("/predictions/history", response_model=List[PredictionHistory])
async def get_prediction_history(
    limit: int = 50,
    offset: int = 0,
    model_name: str = None
):
    """
    Get prediction history.
    
    Args:
        limit: Maximum number of records to return (default: 50)
        offset: Number of records to skip (default: 0)
        model_name: Filter by model name (optional)
        
    Returns:
        List of historical predictions
    """
    try:
        records = prediction_logger.get_predictions(
            limit=limit,
            offset=offset,
            model_name=model_name
        )
        
        return [
            PredictionHistory(
                id=record['id'],
                timestamp=datetime.fromisoformat(record['timestamp']),
                model_name=record['model_name'],
                model_version=record['model_version'],
                input_features=record['input_features'],
                prediction=record['prediction'],
                confidence=record['confidence'],
                processing_time_ms=record['processing_time_ms'],
                client_ip=record['client_ip']
            )
            for record in records
        ]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get prediction history: {str(e)}")


@app.get("/model/info")
async def get_model_info():
    """Get information about the currently loaded model."""
    return model_loader.get_model_info()


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
