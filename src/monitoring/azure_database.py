"""
Azure PostgreSQL-based prediction logging for production deployment.
"""

import asyncio
import asyncpg
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config.azure_config import config


class AzurePredictionLogger:
    """PostgreSQL-based prediction logging for Azure deployment."""
    
    def __init__(self):
        self.pool = None
        self.database_url = config.database_url
    
    async def initialize(self):
        """Initialize database connection pool."""
        try:
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=1,
                max_size=10,
                command_timeout=60
            )
            await self._create_table()
            print("✅ Connected to Azure PostgreSQL")
        except Exception as e:
            print(f"❌ Failed to connect to Azure PostgreSQL: {e}")
            # Fallback to SQLite for development
            from .database import PredictionLogger
            self.fallback_logger = PredictionLogger()
    
    async def _create_table(self):
        """Create the predictions table if it doesn't exist."""
        if not self.pool:
            return
            
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    model_name VARCHAR(255) NOT NULL,
                    model_version VARCHAR(50) NOT NULL,
                    input_features JSONB NOT NULL,
                    prediction JSONB NOT NULL,
                    confidence REAL,
                    processing_time_ms REAL,
                    client_ip INET,
                    user_agent TEXT
                );
            """)
            
            # Create index on timestamp for faster queries
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_predictions_timestamp 
                ON predictions(timestamp);
            """)
    
    async def log_prediction(
        self,
        model_name: str,
        model_version: str,
        input_features: Dict[str, Any],
        prediction: Any,
        confidence: float = None,
        processing_time_ms: float = None,
        client_ip: str = None,
        user_agent: str = None
    ) -> int:
        """Log a prediction to PostgreSQL."""
        if not self.pool:
            # Use fallback logger
            if hasattr(self, 'fallback_logger'):
                return self.fallback_logger.log_prediction(
                    model_name, model_version, input_features, prediction,
                    confidence, processing_time_ms, client_ip, user_agent
                )
            return -1
        
        async with self.pool.acquire() as conn:
            result = await conn.fetchval("""
                INSERT INTO predictions (
                    model_name, model_version, input_features, prediction,
                    confidence, processing_time_ms, client_ip, user_agent
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                RETURNING id
            """, 
            model_name, model_version, 
            json.dumps(input_features), json.dumps(prediction),
            confidence, processing_time_ms, client_ip, user_agent
            )
            return result
    
    async def get_predictions(
        self,
        limit: int = 100,
        offset: int = 0,
        model_name: str = None,
        start_date: str = None,
        end_date: str = None
    ) -> List[Dict[str, Any]]:
        """Retrieve predictions from PostgreSQL."""
        if not self.pool:
            # Use fallback logger
            if hasattr(self, 'fallback_logger'):
                return self.fallback_logger.get_predictions(
                    limit, offset, model_name, start_date, end_date
                )
            return []
        
        query = "SELECT * FROM predictions WHERE 1=1"
        params = []
        param_count = 0
        
        if model_name:
            param_count += 1
            query += f" AND model_name = ${param_count}"
            params.append(model_name)
        
        if start_date:
            param_count += 1
            query += f" AND timestamp >= ${param_count}"
            params.append(start_date)
        
        if end_date:
            param_count += 1
            query += f" AND timestamp <= ${param_count}"
            params.append(end_date)
        
        query += f" ORDER BY timestamp DESC LIMIT ${param_count + 1} OFFSET ${param_count + 2}"
        params.extend([limit, offset])
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            
            results = []
            for row in rows:
                record = dict(row)
                # Parse JSON fields
                record['input_features'] = json.loads(record['input_features'])
                record['prediction'] = json.loads(record['prediction'])
                record['timestamp'] = record['timestamp'].isoformat()
                results.append(record)
            
            return results
    
    async def get_statistics(self, model_name: str = None, days: int = 30) -> Dict[str, Any]:
        """Get prediction statistics from PostgreSQL."""
        if not self.pool:
            # Use fallback logger
            if hasattr(self, 'fallback_logger'):
                return self.fallback_logger.get_statistics(model_name, days)
            return {}
        
        start_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        start_date = start_date.replace(day=start_date.day - days)
        
        query = """
            SELECT 
                COUNT(*) as total_predictions,
                AVG(processing_time_ms) as avg_processing_time,
                MIN(processing_time_ms) as min_processing_time,
                MAX(processing_time_ms) as max_processing_time,
                AVG(confidence) as avg_confidence
            FROM predictions 
            WHERE timestamp >= $1
        """
        params = [start_date]
        
        if model_name:
            query += " AND model_name = $2"
            params.append(model_name)
        
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, *params)
            
            return {
                'total_predictions': row['total_predictions'] or 0,
                'avg_processing_time_ms': float(row['avg_processing_time']) if row['avg_processing_time'] else 0,
                'min_processing_time_ms': float(row['min_processing_time']) if row['min_processing_time'] else 0,
                'max_processing_time_ms': float(row['max_processing_time']) if row['max_processing_time'] else 0,
                'avg_confidence': float(row['avg_confidence']) if row['avg_confidence'] else 0,
                'period_days': days
            }
    
    async def close(self):
        """Close database connection pool."""
        if self.pool:
            await self.pool.close()


# Global Azure logger instance
azure_prediction_logger = AzurePredictionLogger()
