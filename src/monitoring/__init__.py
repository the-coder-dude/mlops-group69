"""Monitoring module for the MLOps pipeline."""

from .database import PredictionLogger, prediction_logger

__all__ = ['PredictionLogger', 'prediction_logger']
