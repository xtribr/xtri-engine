"""
Supabase client singleton.

Loads credentials from environment (set via .env file or Docker env vars):
    SUPABASE_URL, SUPABASE_KEY

The key should be the service_role key (not anon) for server-side operations.
"""

import os
from functools import lru_cache
from typing import Optional

from supabase import Client, create_client


@lru_cache(maxsize=1)
def get_supabase() -> Optional[Client]:
    """Return a cached Supabase client, or None if credentials are missing."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        return None
    return create_client(url, key.strip())


def is_supabase_configured() -> bool:
    """Check if Supabase credentials are present in the environment."""
    return bool(os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_KEY"))
