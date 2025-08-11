"""
Secure Configuration Management for MLOps Pipeline
Implements multiple security layers for credential management
"""

import os
import json
import logging
from typing import Optional, Dict, Any
from functools import lru_cache
from dataclasses import dataclass
import warnings

# Try to import Azure Key Vault (optional)
try:
    from azure.keyvault.secrets import SecretClient
    from azure.identity import DefaultAzureCredential, ManagedIdentityCredential
    AZURE_KEYVAULT_AVAILABLE = True
except ImportError:
    AZURE_KEYVAULT_AVAILABLE = False
    warnings.warn("Azure Key Vault not available. Using environment variables only.")

# Setup secure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SecurityConfig:
    """Security-focused configuration management"""
    
    # Environment detection
    environment: str = os.getenv("ENVIRONMENT", "development")
    
    # Azure Key Vault configuration
    keyvault_name: Optional[str] = os.getenv("KEYVAULT_NAME")
    use_managed_identity: bool = os.getenv("USE_MANAGED_IDENTITY", "false").lower() == "true"
    
    # Fallback to environment variables
    postgres_host: Optional[str] = os.getenv("POSTGRES_HOST")
    postgres_user: Optional[str] = os.getenv("POSTGRES_USER", "mlopsadmin")
    postgres_db: Optional[str] = os.getenv("POSTGRES_DB", "predictions")
    postgres_port: str = os.getenv("POSTGRES_PORT", "5432")
    
    # Storage configuration
    azure_storage_account: Optional[str] = os.getenv("AZURE_STORAGE_ACCOUNT")
    
    # MLflow configuration
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    azure_storage_container: str = os.getenv("AZURE_STORAGE_CONTAINER", "mlflow-artifacts")
    
    # Application Insights
    appinsights_connection_string: Optional[str] = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")

class SecureSecretsManager:
    """Secure secrets management with multiple backends"""
    
    def __init__(self):
        self.config = SecurityConfig()
        self._keyvault_client = None
        self._secrets_cache: Dict[str, Any] = {}
        self._initialize_clients()
        
    def _initialize_clients(self):
        """Initialize Azure Key Vault client if available"""
        if not AZURE_KEYVAULT_AVAILABLE or not self.config.keyvault_name:
            logger.info("Using environment variables for secrets management")
            return
            
        try:
            if self.config.use_managed_identity:
                credential = ManagedIdentityCredential()
                logger.info("Using managed identity for Key Vault access")
            else:
                credential = DefaultAzureCredential()
                logger.info("Using default credential chain for Key Vault access")
                
            vault_url = f"https://{self.config.keyvault_name}.vault.azure.net/"
            self._keyvault_client = SecretClient(vault_url=vault_url, credential=credential)
            
            # Test connection
            try:
                list(self._keyvault_client.list_properties_of_secrets(max_page_size=1))
                logger.info("✅ Key Vault connection successful")
            except Exception as e:
                logger.warning(f"⚠️ Key Vault connection failed: {e}")
                self._keyvault_client = None
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize Key Vault client: {e}")
            self._keyvault_client = None
    
    def _get_from_keyvault(self, secret_name: str) -> Optional[str]:
        """Retrieve secret from Azure Key Vault"""
        if not self._keyvault_client:
            return None
            
        try:
            secret = self._keyvault_client.get_secret(secret_name)
            logger.info(f"✅ Retrieved secret '{secret_name}' from Key Vault")
            return secret.value
        except Exception as e:
            logger.warning(f"⚠️ Failed to retrieve '{secret_name}' from Key Vault: {e}")
            return None
    
    def _get_from_environment(self, env_var_name: str) -> Optional[str]:
        """Retrieve secret from environment variable"""
        value = os.getenv(env_var_name)
        if value:
            logger.info(f"✅ Retrieved '{env_var_name}' from environment variables")
        else:
            logger.warning(f"⚠️ Environment variable '{env_var_name}' not found")
        return value
    
    def get_secret(self, secret_name: str, env_var_name: Optional[str] = None) -> Optional[str]:
        """
        Get secret with fallback mechanism:
        1. Try Azure Key Vault (if configured)
        2. Fallback to environment variables
        
        Args:
            secret_name: Name of secret in Key Vault
            env_var_name: Environment variable name (defaults to secret_name.upper())
        """
        if env_var_name is None:
            env_var_name = secret_name.upper().replace("-", "_")
        
        # Check cache first
        cache_key = f"{secret_name}:{env_var_name}"
        if cache_key in self._secrets_cache:
            return self._secrets_cache[cache_key]
        
        # Try Key Vault first
        secret_value = self._get_from_keyvault(secret_name)
        
        # Fallback to environment variables
        if secret_value is None:
            secret_value = self._get_from_environment(env_var_name)
        
        # Cache the result (be careful with sensitive data in production)
        if secret_value and self.config.environment != "production":
            self._secrets_cache[cache_key] = secret_value
        
        return secret_value
    
    def validate_required_secrets(self, required_secrets: list) -> bool:
        """Validate that all required secrets are available"""
        missing_secrets = []
        
        for secret in required_secrets:
            if isinstance(secret, tuple):
                secret_name, env_var_name = secret
            else:
                secret_name, env_var_name = secret, None
                
            if not self.get_secret(secret_name, env_var_name):
                missing_secrets.append(secret_name)
        
        if missing_secrets:
            logger.error(f"❌ Missing required secrets: {missing_secrets}")
            return False
        
        logger.info("✅ All required secrets validated")
        return True
    
    # Specific secret getters with proper fallback
    @lru_cache(maxsize=1)
    def get_postgres_password(self) -> Optional[str]:
        """Get PostgreSQL password"""
        return self.get_secret("postgres-password", "POSTGRES_PASSWORD")
    
    @lru_cache(maxsize=1)
    def get_azure_storage_key(self) -> Optional[str]:
        """Get Azure Storage key"""
        return self.get_secret("azure-storage-key", "AZURE_STORAGE_KEY")
    
    @lru_cache(maxsize=1)
    def get_appinsights_key(self) -> Optional[str]:
        """Get Application Insights instrumentation key"""
        return self.get_secret("appinsights-key", "APPINSIGHTS_INSTRUMENTATION_KEY")
    
    def get_database_connection_string(self) -> Optional[str]:
        """Build secure database connection string"""
        password = self.get_postgres_password()
        
        if not password or not self.config.postgres_host:
            logger.warning("⚠️ Insufficient database configuration")
            return None
        
        # Build connection string without exposing password in logs
        conn_string = (
            f"postgresql://{self.config.postgres_user}:{password}@"
            f"{self.config.postgres_host}:{self.config.postgres_port}/"
            f"{self.config.postgres_db}"
        )
        
        logger.info("✅ Database connection string built successfully")
        return conn_string
    
    def mask_sensitive_config(self) -> Dict[str, Any]:
        """Return configuration with sensitive values masked for logging"""
        config_dict = {}
        
        # Safe to log
        config_dict["environment"] = self.config.environment
        config_dict["postgres_host"] = self.config.postgres_host
        config_dict["postgres_user"] = self.config.postgres_user
        config_dict["postgres_db"] = self.config.postgres_db
        config_dict["postgres_port"] = self.config.postgres_port
        config_dict["azure_storage_account"] = self.config.azure_storage_account
        config_dict["mlflow_tracking_uri"] = self.config.mlflow_tracking_uri
        config_dict["keyvault_configured"] = self._keyvault_client is not None
        
        # Masked values
        config_dict["postgres_password"] = "***MASKED***" if self.get_postgres_password() else "NOT_SET"
        config_dict["azure_storage_key"] = "***MASKED***" if self.get_azure_storage_key() else "NOT_SET"
        config_dict["appinsights_key"] = "***MASKED***" if self.get_appinsights_key() else "NOT_SET"
        
        return config_dict

# Global instance
secrets_manager = SecureSecretsManager()

def get_secure_config() -> SecureSecretsManager:
    """Get the global secrets manager instance"""
    return secrets_manager

def validate_security_config() -> bool:
    """Validate security configuration on startup"""
    required_secrets = [
        ("azure-storage-key", "AZURE_STORAGE_KEY"),
    ]
    
    # Add database secrets if PostgreSQL is configured
    if secrets_manager.config.postgres_host:
        required_secrets.append(("postgres-password", "POSTGRES_PASSWORD"))
    
    return secrets_manager.validate_required_secrets(required_secrets)

# Security audit logging
def log_security_event(event_type: str, details: Dict[str, Any]):
    """Log security events for audit trail"""
    audit_log = {
        "timestamp": "timestamp_here",  # Add proper timestamp
        "event_type": event_type,
        "environment": secrets_manager.config.environment,
        "details": details
    }
    
    # In production, send to SIEM or security logging service
    logger.info(f"SECURITY_AUDIT: {json.dumps(audit_log)}")

if __name__ == "__main__":
    # Test configuration
    config = get_secure_config()
    
    print("🔒 Security Configuration Test")
    print("=" * 40)
    
    # Display masked configuration
    masked_config = config.mask_sensitive_config()
    for key, value in masked_config.items():
        print(f"{key}: {value}")
    
    # Validate configuration
    print("\n🔍 Configuration Validation")
    print("=" * 40)
    
    if validate_security_config():
        print("✅ Configuration is valid and secure")
    else:
        print("❌ Configuration validation failed")
        exit(1)
