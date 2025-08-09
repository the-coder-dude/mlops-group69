"""
Data loading and preprocessing module for Iris dataset.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Any
import os
import pickle


def load_iris_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the Iris dataset.
    
    Returns:
        Tuple of (features DataFrame, target Series)
    """
    iris = load_iris()
    
    # Create DataFrame with feature names
    features = pd.DataFrame(
        iris.data, 
        columns=iris.feature_names
    )
    
    # Create target series with class names
    target = pd.Series(iris.target, name='species')
    target = target.map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    
    return features, target


def preprocess_data(
    features: pd.DataFrame, 
    target: pd.Series, 
    test_size: float = 0.2, 
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Preprocess the data for model training.
    
    Args:
        features: Feature DataFrame
        target: Target Series
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary containing processed data and scaler
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=random_state, stratify=target
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrames for easier handling
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=features.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=features.columns, index=X_test.index)
    
    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'feature_names': list(features.columns)
    }


def save_preprocessed_data(data_dict: Dict[str, Any], output_dir: str = "data") -> None:
    """
    Save preprocessed data to files.
    
    Args:
        data_dict: Dictionary containing processed data
        output_dir: Directory to save the data
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save train/test data
    data_dict['X_train'].to_csv(f"{output_dir}/X_train.csv", index=False)
    data_dict['X_test'].to_csv(f"{output_dir}/X_test.csv", index=False)
    data_dict['y_train'].to_csv(f"{output_dir}/y_train.csv", index=False)
    data_dict['y_test'].to_csv(f"{output_dir}/y_test.csv", index=False)
    
    # Save scaler
    with open(f"{output_dir}/scaler.pkl", "wb") as f:
        pickle.dump(data_dict['scaler'], f)
    
    # Save feature names
    with open(f"{output_dir}/feature_names.pkl", "wb") as f:
        pickle.dump(data_dict['feature_names'], f)


def load_preprocessed_data(data_dir: str = "data") -> Dict[str, Any]:
    """
    Load preprocessed data from files.
    
    Args:
        data_dir: Directory containing the data files
        
    Returns:
        Dictionary containing loaded data
    """
    X_train = pd.read_csv(f"{data_dir}/X_train.csv")
    X_test = pd.read_csv(f"{data_dir}/X_test.csv")
    y_train = pd.read_csv(f"{data_dir}/y_train.csv").squeeze()
    y_test = pd.read_csv(f"{data_dir}/y_test.csv").squeeze()
    
    with open(f"{data_dir}/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    
    with open(f"{data_dir}/feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'feature_names': feature_names
    }


if __name__ == "__main__":
    # Load and preprocess data
    features, target = load_iris_data()
    data_dict = preprocess_data(features, target)
    
    # Save processed data
    save_preprocessed_data(data_dict)
    
    print("✅ Data loaded and preprocessed successfully!")
    print(f"Training set shape: {data_dict['X_train'].shape}")
    print(f"Test set shape: {data_dict['X_test'].shape}")
    print(f"Target distribution:\n{data_dict['y_train'].value_counts()}")
