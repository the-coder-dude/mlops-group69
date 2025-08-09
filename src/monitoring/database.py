"""
Database module for logging prediction requests and responses.
"""

import sqlite3
import json
from datetime import datetime
from typing import Dict, Any, List
import os
from pathlib import Path


class PredictionLogger:
    """SQLite-based prediction logging."""
    
    def __init__(self, db_path: str = "monitoring/predictions.db"):
        """
        Initialize the prediction logger.
        
        Args:
            db_path: Path to SQLite database file
        """
        # Ensure directory exists
        if os.path.dirname(db_path):
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.db_path = db_path
        self._create_table()
    
    def _create_table(self) -> None:
        """Create the predictions table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    input_features TEXT NOT NULL,
                    prediction TEXT NOT NULL,
                    confidence REAL,
                    processing_time_ms REAL,
                    client_ip TEXT,
                    user_agent TEXT
                )
            """)
            
            # Create index on timestamp for faster queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON predictions(timestamp)
            """)
    
    def log_prediction(
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
        """
        Log a prediction to the database.
        
        Args:
            model_name: Name of the model used
            model_version: Version of the model
            input_features: Input features used for prediction
            prediction: Model prediction result
            confidence: Prediction confidence score
            processing_time_ms: Time taken for prediction in milliseconds
            client_ip: Client IP address
            user_agent: Client user agent string
            
        Returns:
            ID of the logged prediction
        """
        timestamp = datetime.utcnow().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO predictions (
                    timestamp, model_name, model_version, input_features,
                    prediction, confidence, processing_time_ms, client_ip, user_agent
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp,
                model_name,
                model_version,
                json.dumps(input_features),
                json.dumps(prediction),
                confidence,
                processing_time_ms,
                client_ip,
                user_agent
            ))
            
            return cursor.lastrowid
    
    def get_predictions(
        self,
        limit: int = 100,
        offset: int = 0,
        model_name: str = None,
        start_date: str = None,
        end_date: str = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve predictions from the database.
        
        Args:
            limit: Maximum number of records to return
            offset: Number of records to skip
            model_name: Filter by model name
            start_date: Filter predictions after this date (ISO format)
            end_date: Filter predictions before this date (ISO format)
            
        Returns:
            List of prediction records
        """
        query = "SELECT * FROM predictions WHERE 1=1"
        params = []
        
        if model_name:
            query += " AND model_name = ?"
            params.append(model_name)
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)
        
        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            
            results = []
            for row in cursor.fetchall():
                record = dict(row)
                # Parse JSON fields
                record['input_features'] = json.loads(record['input_features'])
                record['prediction'] = json.loads(record['prediction'])
                results.append(record)
            
            return results
    
    def get_statistics(self, model_name: str = None, days: int = 30) -> Dict[str, Any]:
        """
        Get prediction statistics.
        
        Args:
            model_name: Filter by model name
            days: Number of days to look back
            
        Returns:
            Dictionary containing statistics
        """
        # Calculate start date
        from datetime import timedelta
        start_date = (datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0) - 
                     timedelta(days=days)).isoformat()
        
        query = """
            SELECT 
                COUNT(*) as total_predictions,
                AVG(processing_time_ms) as avg_processing_time,
                MIN(processing_time_ms) as min_processing_time,
                MAX(processing_time_ms) as max_processing_time,
                AVG(confidence) as avg_confidence
            FROM predictions 
            WHERE timestamp >= ?
        """
        params = [start_date]
        
        if model_name:
            query += " AND model_name = ?"
            params.append(model_name)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            row = cursor.fetchone()
            
            return {
                'total_predictions': row[0] or 0,
                'avg_processing_time_ms': row[1] or 0,
                'min_processing_time_ms': row[2] or 0,
                'max_processing_time_ms': row[3] or 0,
                'avg_confidence': row[4] or 0,
                'period_days': days
            }
    
    def get_daily_counts(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get daily prediction counts for the last N days.
        
        Args:
            days: Number of days to look back
            
        Returns:
            List of daily counts
        """
        query = """
            SELECT 
                DATE(timestamp) as date,
                COUNT(*) as count
            FROM predictions 
            WHERE timestamp >= date('now', '-{} days')
            GROUP BY DATE(timestamp)
            ORDER BY date DESC
        """.format(days)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query)
            
            return [dict(row) for row in cursor.fetchall()]


# Global logger instance
prediction_logger = PredictionLogger()
