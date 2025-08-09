"""
Model loading utilities for the API.
"""

import mlflow
import mlflow.sklearn
import pickle
import numpy as np
from typing import Tuple, Any, List
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))


class ModelLoader:
    """Loads and manages ML models for prediction."""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.model_name = None
        self.model_version = None
        self.class_names = ['setosa', 'versicolor', 'virginica']
    
    def load_best_model(self) -> bool:
        """
        Load the best performing model from MLflow.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            # Set MLflow tracking URI  
            mlflow.set_tracking_uri("file:./mlruns")
            
            # Get the latest version of the best model (Logistic Regression based on our training)
            client = mlflow.tracking.MlflowClient()
            
            # Try to get the latest version of the logistic regression model
            try:
                latest_version = client.get_latest_versions(
                    "iris_logistic_regression", 
                    stages=["None"]
                )[0]
                
                model_uri = f"models:/iris_logistic_regression/{latest_version.version}"
                self.model = mlflow.sklearn.load_model(model_uri)
                self.model_name = "iris_logistic_regression"
                self.model_version = str(latest_version.version)
                
            except Exception as e:
                print(f"⚠️ Could not load logistic regression: {e}")
                
                # Fallback to random forest if logistic regression not available
                try:
                    latest_version = client.get_latest_versions(
                        "iris_random_forest",
                        stages=["None"]
                    )[0]
                    
                    model_uri = f"models:/iris_random_forest/{latest_version.version}"
                    self.model = mlflow.sklearn.load_model(model_uri)
                    self.model_name = "iris_random_forest"
                    self.model_version = str(latest_version.version)
                    
                except Exception as e2:
                    print(f"⚠️ Could not load random forest: {e2}")
                    
                    # Final fallback - try to load model directly from pickle files
                    import glob
                    import pickle as pkl
                    
                    model_paths = glob.glob("mlruns/*/models/*/artifacts/model.pkl")
                    if model_paths:
                        # Load the first available model
                        with open(model_paths[0], 'rb') as f:
                            self.model = pkl.load(f)
                        self.model_name = "direct_loaded_model"
                        self.model_version = "1"
                        print(f"✅ Loaded model directly from: {model_paths[0]}")
                    else:
                        raise Exception("No model pickle files found")
            
            # Load the scaler
            self._load_scaler()
            
            print(f"✅ Loaded model: {self.model_name} v{self.model_version}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    
    def _load_scaler(self) -> None:
        """Load the data scaler."""
        try:
            scaler_path = "data/scaler.pkl"
            if os.path.exists(scaler_path):
                with open(scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)
                print("✅ Scaler loaded successfully")
            else:
                print("⚠️ Scaler not found, predictions may be inaccurate")
        except Exception as e:
            print(f"⚠️ Failed to load scaler: {e}")
    
    def predict(self, features: List[List[float]]) -> Tuple[List[str], List[List[float]]]:
        """
        Make predictions on input features.
        
        Args:
            features: List of feature arrays for prediction
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Convert to numpy array
        X = np.array(features)
        
        # Apply scaling if scaler is available
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        # Convert predictions to class names if they're numeric indices
        if len(predictions) > 0 and isinstance(predictions[0], (int, np.integer)):
            prediction_names = [self.class_names[int(pred)] for pred in predictions]
        else:
            # Already string predictions
            prediction_names = predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions)
        
        return prediction_names, probabilities.tolist()
    
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready."""
        return self.model is not None
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            'model_name': self.model_name,
            'model_version': self.model_version,
            'is_loaded': self.is_loaded(),
            'class_names': self.class_names
        }


# Global model loader instance
model_loader = ModelLoader()
