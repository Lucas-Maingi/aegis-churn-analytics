import logging
import os
from typing import Optional

from supabase import Client, create_client

logger = logging.getLogger(__name__)

# Supabase configuration from environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

_supabase_client: Optional[Client] = None

def get_supabase_client() -> Optional[Client]:
    """
    Returns a configured Supabase client if URL and KEY are present.
    Returns None if the configuration is missing (allows the app to run locally without DB).
    """
    global _supabase_client
    
    if _supabase_client is not None:
        return _supabase_client
        
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
            logger.info("Supabase client initialized successfully.")
            return _supabase_client
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {e}")
            return None
    else:
        logger.warning("SUPABASE_URL or SUPABASE_KEY not set. Running without database integration.")
        return None

def log_prediction(customer_id: str, churn_probability: float, risk_tier: str, prediction_date: str = None) -> bool:
    """
    Logs a single prediction to the Supabase 'predictions' table.
    """
    client = get_supabase_client()
    if not client:
        return False
        
    try:
        data = {
            "customer_id": customer_id if customer_id else "UNKNOWN",
            "churn_probability": churn_probability,
            "risk_tier": risk_tier
        }
        # prediction_date will be auto-populated by Postgres now() if not provided,
        # or we could explicitly pass it if needed.
        
        client.table("predictions").insert(data).execute()
        return True
    except Exception as e:
        logger.error(f"Error logging prediction to Supabase: {e}")
        return False

def log_prediction_batch(predictions: list[dict]) -> bool:
    """
    Logs multiple predictions to the Supabase 'predictions' table in a single request.
    Expected format: [{"customer_id": "...", "churn_probability": 85.0, "risk_tier": "HIGH"}, ...]
    """
    client = get_supabase_client()
    if not client or not predictions:
        return False
        
    try:
        client.table("predictions").insert(predictions).execute()
        return True
    except Exception as e:
        logger.error(f"Error logging batch predictions to Supabase: {e}")
        return False
