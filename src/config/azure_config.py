"""
Azure-specific configuration for the MLOps pipeline.
"""

import os
from typing import Optional


class AzureConfig:
    """Configuration for Azure deployment."""
    
    # Database Configuration
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "iris-mlops-db.postgres.database.azure.com")
    POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
    POSTGRES_DB = os.getenv("POSTGRES_DB", "predictions")
    POSTGRES_USER = os.getenv("POSTGRES_USER", "mlopsadmin")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
    
    @property
    def database_url(self) -> str:
        """Get PostgreSQL connection string."""
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )
    
    # Storage Configuration
    AZURE_STORAGE_ACCOUNT = os.getenv("AZURE_STORAGE_ACCOUNT", "irismlopstorage")
    AZURE_STORAGE_KEY = os.getenv("AZURE_STORAGE_KEY")
    AZURE_STORAGE_CONTAINER = os.getenv("AZURE_STORAGE_CONTAINER", "mlflow-artifacts")
    
    # MLflow Configuration
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    MLFLOW_ARTIFACT_ROOT = os.getenv(
        "MLFLOW_ARTIFACT_ROOT", 
        f"azure://{AZURE_STORAGE_CONTAINER}"
    )
    
    # Application Insights
    APPINSIGHTS_INSTRUMENTATION_KEY = os.getenv("APPINSIGHTS_INSTRUMENTATION_KEY")
    
    # Application Configuration
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))


config = AzureConfig()
