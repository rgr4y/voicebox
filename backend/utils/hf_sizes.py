"""
Query HuggingFace API for model repository sizes.
Caches results in memory for the server lifetime.
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# In-memory cache: repo_id -> size_mb
_size_cache: Dict[str, Optional[float]] = {}


async def get_repo_size_mb(repo_id: str) -> Optional[float]:
    """
    Get the total size of a HuggingFace model repo in MB.

    Queries the HF API on first call, then caches for the server lifetime.
    Returns None if the query fails (network error, repo not found, etc).
    """
    if repo_id in _size_cache:
        return _size_cache[repo_id]

    try:
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"https://huggingface.co/api/models/{repo_id}")
            if resp.status_code != 200:
                logger.warning(f"HF API returned {resp.status_code} for {repo_id}")
                _size_cache[repo_id] = None
                return None

            data = resp.json()

            # Method 1: Use safetensors metadata if available
            safetensors = data.get("safetensors")
            if safetensors and "total" in safetensors:
                size_mb = safetensors["total"] / (1024 * 1024)
                _size_cache[repo_id] = size_mb
                return size_mb

            # Method 2: Sum sibling file sizes
            siblings = data.get("siblings", [])
            total_bytes = sum(s.get("size", 0) for s in siblings if s.get("size"))
            if total_bytes > 0:
                size_mb = total_bytes / (1024 * 1024)
                _size_cache[repo_id] = size_mb
                return size_mb

            _size_cache[repo_id] = None
            return None

    except Exception as e:
        logger.warning(f"Failed to query HF for {repo_id}: {e}")
        _size_cache[repo_id] = None
        return None
