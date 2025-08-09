"""
Tests for data processing functions.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.load_data import load_iris_data, preprocess_data


class TestDataProcessing:
    """Test class for data processing functions."""
    
    def test_load_iris_data(self):
        """Test loading iris data."""
        features, target = load_iris_data()
        
        # Check data types
        assert isinstance(features, pd.DataFrame)
        assert isinstance(target, pd.Series)
        
        # Check dimensions
        assert features.shape == (150, 4)
        assert target.shape == (150,)
        
        # Check column names
        expected_columns = [
            'sepal length (cm)',
            'sepal width (cm)', 
            'petal length (cm)',
            'petal width (cm)'
        ]
        assert list(features.columns) == expected_columns
        
        # Check target values
        unique_targets = set(target.unique())
        expected_targets = {'setosa', 'versicolor', 'virginica'}
        assert unique_targets == expected_targets
    
    def test_preprocess_data(self):
        """Test data preprocessing."""
        features, target = load_iris_data()
        data_dict = preprocess_data(features, target, test_size=0.2, random_state=42)
        
        # Check that all required keys are present
        required_keys = ['X_train', 'X_test', 'y_train', 'y_test', 'scaler', 'feature_names']
        assert all(key in data_dict for key in required_keys)
        
        # Check data shapes
        assert data_dict['X_train'].shape[0] == 120  # 80% of 150
        assert data_dict['X_test'].shape[0] == 30    # 20% of 150
        assert data_dict['X_train'].shape[1] == 4    # 4 features
        assert data_dict['X_test'].shape[1] == 4     # 4 features
        
        # Check target shapes
        assert len(data_dict['y_train']) == 120
        assert len(data_dict['y_test']) == 30
        
        # Check feature names
        assert len(data_dict['feature_names']) == 4
        
        # Check that data is scaled (mean should be close to 0, std close to 1)
        train_means = data_dict['X_train'].mean()
        train_stds = data_dict['X_train'].std()
        
        # Scaled data should have mean close to 0 and std close to 1
        assert all(abs(mean) < 0.1 for mean in train_means)
        assert all(abs(std - 1.0) < 0.1 for std in train_stds)
    
    def test_preprocess_data_stratification(self):
        """Test that preprocessing maintains class distribution."""
        features, target = load_iris_data()
        data_dict = preprocess_data(features, target, test_size=0.2, random_state=42)
        
        # Check that class distribution is maintained
        original_distribution = target.value_counts(normalize=True).sort_index()
        train_distribution = data_dict['y_train'].value_counts(normalize=True).sort_index()
        test_distribution = data_dict['y_test'].value_counts(normalize=True).sort_index()
        
        # The distributions should be similar (within 10% tolerance)
        for class_name in original_distribution.index:
            original_prop = original_distribution[class_name]
            train_prop = train_distribution[class_name]
            test_prop = test_distribution[class_name]
            
            assert abs(train_prop - original_prop) < 0.1
            assert abs(test_prop - original_prop) < 0.1
    
    def test_preprocess_data_reproducibility(self):
        """Test that preprocessing is reproducible with same random state."""
        features, target = load_iris_data()
        
        data_dict1 = preprocess_data(features, target, test_size=0.2, random_state=42)
        data_dict2 = preprocess_data(features, target, test_size=0.2, random_state=42)
        
        # Check that the splits are identical
        pd.testing.assert_frame_equal(data_dict1['X_train'], data_dict2['X_train'])
        pd.testing.assert_frame_equal(data_dict1['X_test'], data_dict2['X_test'])
        pd.testing.assert_series_equal(data_dict1['y_train'], data_dict2['y_train'])
        pd.testing.assert_series_equal(data_dict1['y_test'], data_dict2['y_test'])
