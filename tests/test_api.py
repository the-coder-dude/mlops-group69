"""
Tests for the API endpoints.
"""

import pytest
import sys
import os
from fastapi.testclient import TestClient

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from api.main import app

client = TestClient(app)


class TestAPIEndpoints:
    """Test class for API endpoints."""
    
    def test_root_endpoint(self):
        """Test the root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert data["message"] == "Iris Classification MLOps API"
    
    def test_health_endpoint(self):
        """Test the health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "model_loaded" in data
        assert "database_connected" in data
    
    def test_model_info_endpoint(self):
        """Test the model info endpoint."""
        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert "model_name" in data
        assert "model_version" in data
        assert "is_loaded" in data
        assert "class_names" in data
    
    def test_predict_endpoint_valid_input(self):
        """Test prediction with valid input."""
        test_data = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        
        response = client.post("/predict", json=test_data)
        
        # Check if model is loaded, if not, expect 503
        if response.status_code == 503:
            assert "Model not loaded" in response.json()["detail"]
        else:
            assert response.status_code == 200
            data = response.json()
            assert "predictions" in data
            assert "probabilities" in data
            assert "model_name" in data
            assert "model_version" in data
            assert "processing_time_ms" in data
            assert "timestamp" in data
            assert len(data["predictions"]) == 1
            assert data["predictions"][0] in ["setosa", "versicolor", "virginica"]
    
    def test_predict_endpoint_invalid_input(self):
        """Test prediction with invalid input."""
        test_data = {
            "sepal_length": "invalid",
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        
        response = client.post("/predict", json=test_data)
        assert response.status_code == 422  # Validation error
    
    def test_predict_endpoint_missing_fields(self):
        """Test prediction with missing required fields."""
        test_data = {
            "sepal_length": 5.1,
            "sepal_width": 3.5
            # Missing petal_length and petal_width
        }
        
        response = client.post("/predict", json=test_data)
        assert response.status_code == 422  # Validation error
    
    def test_predict_batch_endpoint(self):
        """Test batch prediction endpoint."""
        test_data = {
            "features": [
                {
                    "sepal_length": 5.1,
                    "sepal_width": 3.5,
                    "petal_length": 1.4,
                    "petal_width": 0.2
                },
                {
                    "sepal_length": 6.2,
                    "sepal_width": 2.9,
                    "petal_length": 4.3,
                    "petal_width": 1.3
                }
            ]
        }
        
        response = client.post("/predict/batch", json=test_data)
        
        # Check if model is loaded, if not, expect 503
        if response.status_code == 503:
            assert "Model not loaded" in response.json()["detail"]
        else:
            assert response.status_code == 200
            data = response.json()
            assert "predictions" in data
            assert "probabilities" in data
            assert len(data["predictions"]) == 2
            assert len(data["probabilities"]) == 2
    
    def test_metrics_endpoint(self):
        """Test the metrics endpoint."""
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "total_predictions" in data
        assert "avg_processing_time_ms" in data
        assert "avg_confidence" in data
        assert "predictions_last_24h" in data
        assert "uptime_hours" in data
    
    def test_prediction_history_endpoint(self):
        """Test the prediction history endpoint."""
        response = client.get("/predictions/history")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        
        # Test with query parameters
        response = client.get("/predictions/history?limit=10&offset=0")
        assert response.status_code == 200
