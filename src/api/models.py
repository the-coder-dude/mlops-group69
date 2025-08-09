"""
Pydantic models for API request/response validation.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
from datetime import datetime


class IrisFeatures(BaseModel):
    """Input features for Iris prediction."""
    
    sepal_length: float = Field(
        ...,
        description="Sepal length in cm",
        ge=0.0,
        le=10.0,
        example=5.1
    )
    sepal_width: float = Field(
        ...,
        description="Sepal width in cm", 
        ge=0.0,
        le=10.0,
        example=3.5
    )
    petal_length: float = Field(
        ...,
        description="Petal length in cm",
        ge=0.0,
        le=10.0,
        example=1.4
    )
    petal_width: float = Field(
        ...,
        description="Petal width in cm",
        ge=0.0,
        le=10.0,
        example=0.2
    )
    
    @validator('*', pre=True)
    def validate_numeric(cls, v):
        """Validate that all inputs are numeric."""
        if not isinstance(v, (int, float)):
            raise ValueError('All features must be numeric values')
        return float(v)
    
    def to_array(self) -> List[float]:
        """Convert to array for model prediction."""
        return [self.sepal_length, self.sepal_width, self.petal_length, self.petal_width]


class PredictionRequest(BaseModel):
    """Request model for batch prediction."""
    
    features: List[IrisFeatures] = Field(
        ...,
        description="List of iris flower measurements for prediction",
        min_items=1,
        max_items=100
    )


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    
    predictions: List[str] = Field(
        ...,
        description="Predicted species for each input"
    )
    probabilities: List[List[float]] = Field(
        ...,
        description="Prediction probabilities for each class"
    )
    model_name: str = Field(
        ...,
        description="Name of the model used for prediction"
    )
    model_version: str = Field(
        ...,
        description="Version of the model used"
    )
    processing_time_ms: float = Field(
        ...,
        description="Time taken for prediction in milliseconds"
    )
    timestamp: datetime = Field(
        ...,
        description="Timestamp when prediction was made"
    )


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(
        ...,
        description="Service status",
        example="healthy"
    )
    timestamp: datetime = Field(
        ...,
        description="Current timestamp"
    )
    model_loaded: bool = Field(
        ...,
        description="Whether the model is loaded and ready"
    )
    database_connected: bool = Field(
        ...,
        description="Whether the database is accessible"
    )


class MetricsResponse(BaseModel):
    """Metrics response for monitoring."""
    
    total_predictions: int = Field(
        ...,
        description="Total number of predictions made"
    )
    avg_processing_time_ms: float = Field(
        ...,
        description="Average processing time in milliseconds"
    )
    avg_confidence: float = Field(
        ...,
        description="Average prediction confidence"
    )
    predictions_last_24h: int = Field(
        ...,
        description="Number of predictions in the last 24 hours"
    )
    uptime_hours: float = Field(
        ...,
        description="Service uptime in hours"
    )


class PredictionHistory(BaseModel):
    """Historical prediction record."""
    
    id: int
    timestamp: datetime
    model_name: str
    model_version: str
    input_features: Dict[str, Any]
    prediction: Any
    confidence: Optional[float]
    processing_time_ms: Optional[float]
    client_ip: Optional[str]
