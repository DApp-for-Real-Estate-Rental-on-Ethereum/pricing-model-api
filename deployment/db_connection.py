"""
Database connection utility for AI features.
Connects to PostgreSQL database 'lotfi' to extract features for ML models.
"""

import os
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

# Database configuration (can be overridden by environment variables)
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', '5432')),
    'database': os.getenv('DB_NAME', 'lotfi'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', '12345')
}


@contextmanager
def get_db_connection():
    """Context manager for database connections."""
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        yield conn
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        raise
    finally:
        if conn:
            conn.close()


def execute_query(query: str, params: tuple = None) -> list:
    """
    Execute a SELECT query and return results as list of dicts.
    
    Args:
        query: SQL query string
        params: Optional tuple of parameters for parameterized query
        
    Returns:
        List of dictionaries (one per row)
    """
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params)
            return [dict(row) for row in cur.fetchall()]


def execute_query_single(query: str, params: tuple = None) -> dict:
    """
    Execute a SELECT query and return single result as dict.
    
    Returns:
        Dictionary representing single row, or None if no results
    """
    results = execute_query(query, params)
    return results[0] if results else None

