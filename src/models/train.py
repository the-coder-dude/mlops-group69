"""
Model training module with MLflow experiment tracking.
"""

import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.load_data import load_iris_data, preprocess_data, save_preprocessed_data


def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary of metrics
    """
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted')
    }


def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series, 
                            X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[Any, Dict[str, float]]:
    """
    Train Logistic Regression model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Tuple of (trained model, metrics)
    """
    # Model parameters
    params = {
        'random_state': 42,
        'max_iter': 1000,
        'solver': 'lbfgs'
    }
    
    # Train model
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred)
    
    # Log with MLflow
    with mlflow.start_run(run_name="Logistic_Regression"):
        # Log parameters
        mlflow.log_params(params)
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Log model
        mlflow.sklearn.log_model(
            model, 
            "model",
            registered_model_name="iris_logistic_regression"
        )
        
        # Log classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        mlflow.log_text(str(report), "classification_report.txt")
        
        print(f"✅ Logistic Regression - Accuracy: {metrics['accuracy']:.4f}")
    
    return model, metrics


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series,
                       X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[Any, Dict[str, float]]:
    """
    Train Random Forest model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Tuple of (trained model, metrics)
    """
    # Model parameters
    params = {
        'n_estimators': 100,
        'random_state': 42,
        'max_depth': 10,
        'min_samples_split': 2,
        'min_samples_leaf': 1
    }
    
    # Train model
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred)
    
    # Log with MLflow
    with mlflow.start_run(run_name="Random_Forest"):
        # Log parameters
        mlflow.log_params(params)
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Log model
        mlflow.sklearn.log_model(
            model, 
            "model",
            registered_model_name="iris_random_forest"
        )
        
        # Log feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        mlflow.log_text(feature_importance.to_string(), "feature_importance.txt")
        
        # Log classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        mlflow.log_text(str(report), "classification_report.txt")
        
        print(f"✅ Random Forest - Accuracy: {metrics['accuracy']:.4f}")
    
    return model, metrics


def select_best_model(results: Dict[str, Dict[str, float]]) -> str:
    """
    Select the best model based on accuracy.
    
    Args:
        results: Dictionary of model results
        
    Returns:
        Name of the best model
    """
    best_model = max(results.keys(), key=lambda x: results[x]['accuracy'])
    print(f"🏆 Best model: {best_model} with accuracy: {results[best_model]['accuracy']:.4f}")
    return best_model


def main():
    """Main training pipeline."""
    # Set MLflow experiment
    mlflow.set_experiment("Iris Classification")
    
    print("🚀 Starting MLOps training pipeline...")
    
    # Load and preprocess data
    print("📊 Loading and preprocessing data...")
    features, target = load_iris_data()
    data_dict = preprocess_data(features, target)
    
    # Save preprocessed data
    save_preprocessed_data(data_dict)
    print("💾 Preprocessed data saved.")
    
    # Extract data
    X_train = data_dict['X_train']
    X_test = data_dict['X_test'] 
    y_train = data_dict['y_train']
    y_test = data_dict['y_test']
    
    print(f"📈 Training on {X_train.shape[0]} samples, testing on {X_test.shape[0]} samples")
    
    # Train models
    results = {}
    
    print("\n🤖 Training Logistic Regression...")
    lr_model, lr_metrics = train_logistic_regression(X_train, y_train, X_test, y_test)
    results['Logistic Regression'] = lr_metrics
    
    print("\n🌲 Training Random Forest...")
    rf_model, rf_metrics = train_random_forest(X_train, y_train, X_test, y_test)
    results['Random Forest'] = rf_metrics
    
    # Select best model
    print("\n🏆 Model Selection:")
    best_model_name = select_best_model(results)
    
    print("\n📊 Final Results:")
    for model_name, metrics in results.items():
        print(f"{model_name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        print()
    
    print("✅ Training completed! Check MLflow UI for detailed tracking.")
    print("Run 'mlflow ui' to view experiments.")


if __name__ == "__main__":
    main()
