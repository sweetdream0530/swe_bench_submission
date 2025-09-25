import json
import logging
import random
import time
from typing import Dict, List, Any
from uuid import UUID

import httpx

from proxy.config import (
    CHUTES_API_KEYS,
    CHUTES_EMBEDDING_URL,
    CHUTES_INFERENCE_URL,
    EMBEDDING_PRICE_PER_SECOND,
    MODEL_PRICING,
    DEFAULT_MODEL,
    ENV,
)
from proxy.models import GPTMessage
from proxy.database import (
    create_embedding,
    update_embedding,
    create_inference,
    update_inference,
)

logger = logging.getLogger(__name__)


class ChutesClient:
    """Client for interacting with Chutes API services"""

    def __init__(self):
        self.api_keys = CHUTES_API_KEYS
        self.current_key_index = 0
        self.failed_keys = set()  # Track keys that have failed recently
        
        if not self.api_keys:
            logger.warning("No CHUTES_API_KEY found in environment variables")
        else:
            logger.info(f"Initialized with {len(self.api_keys)} Chutes API keys")
    
    def get_current_api_key(self) -> str:
        """Get the current API key, rotating if necessary"""
        if not self.api_keys:
            raise RuntimeError("No Chutes API keys available")
        
        # Find next available key
        attempts = 0
        while attempts < len(self.api_keys):
            current_key = self.api_keys[self.current_key_index]
            if current_key not in self.failed_keys:
                return current_key
            
            # Move to next key
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            attempts += 1
        
        # If all keys failed, reset failed keys and try again
        logger.warning("All API keys have failed recently, resetting failed keys list")
        self.failed_keys.clear()
        return self.api_keys[self.current_key_index]
    
    def mark_key_as_failed(self, api_key: str):
        """Mark an API key as failed (likely due to rate limits)"""
        self.failed_keys.add(api_key)
        logger.warning(f"Marked API key as failed: {api_key[:10]}...")
        
        # Move to next key
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)

    async def embed(self, run_id: UUID = None, input_text: str = None) -> Dict[str, Any]:
        """Get embedding for text input"""

        # Create embedding record in database (skip in dev mode)
        embedding_id = None
        if ENV != 'dev':
            embedding_id = await create_embedding(run_id, input_text)

        body = {"inputs": input_text, "seed": random.randint(0, 2**32 - 1)}
        start_time = time.time()

        # Try with current API key, rotate if it fails
        max_retries = len(self.api_keys)
        for attempt in range(max_retries):
            try:
                current_api_key = self.get_current_api_key()
                headers = {
                    "Authorization": f"Bearer {current_api_key}",
                    "Content-Type": "application/json",
                }

                async with httpx.AsyncClient(timeout=60) as client:
                    response = await client.post(CHUTES_EMBEDDING_URL, headers=headers, json=body)
                    response.raise_for_status()

                    total_time_seconds = time.time() - start_time
                    cost = total_time_seconds * EMBEDDING_PRICE_PER_SECOND

                    response_data = response.json()

                    # Update embedding record with cost and response (skip in dev mode)
                    if ENV != 'dev' and embedding_id:
                        await update_embedding(embedding_id, cost, response_data)

                    logger.debug(
                        f"Embedding request for run {run_id} completed in {total_time_seconds:.2f}s, cost: ${cost:.6f}"
                    )

                    return response_data

            except httpx.HTTPStatusError as e:
                error_msg = str(e).lower()
                # Check if this is a rate limit or quota exceeded error
                if any(keyword in error_msg for keyword in ['rate limit', 'quota', 'limit exceeded', '429', 'too many requests']):
                    logger.warning(f"Rate limit hit with API key, rotating to next key: {current_api_key[:10]}...")
                    self.mark_key_as_failed(current_api_key)
                    continue
                else:
                    # For non-rate-limit errors, log and return error
                    logger.error(
                        f"HTTP error in embedding request for run {run_id}: {e.response.status_code} - {e.response.text}"
                    )
                    # Update embedding record with error (skip in dev mode)
                    if ENV != 'dev' and embedding_id:
                        await update_embedding(
                            embedding_id,
                            0.0,
                            {"error": f"HTTP error: {e.response.status_code} - {e.response.text}"},
                        )
                    return {"error": f"HTTP error in embedding request: {e.response.status_code} - {e.response.text}"}
            except httpx.TimeoutException:
                logger.error(f"Timeout in embedding request for run {run_id}")
                # Update embedding record with error (skip in dev mode)
                if ENV != 'dev' and embedding_id:
                    await update_embedding(embedding_id, 0.0, {"error": "Embedding request timed out"})
                return {"error": "Embedding request timed out. Please try again."}
            except Exception as e:
                logger.error(f"Error in embedding request for run {run_id}: {e}")
                # Update embedding record with error (skip in dev mode)
                if ENV != 'dev' and embedding_id:
                    await update_embedding(embedding_id, 0.0, {"error": str(e)})
                return {"error": f"Error in embedding request: {str(e)}"}
        
        # If we've exhausted all keys, return an error
        error_msg = "All Chutes API keys have been exhausted due to rate limits"
        logger.error(error_msg)
        if ENV != 'dev' and embedding_id:
            await update_embedding(embedding_id, 0.0, {"error": error_msg})
        return {"error": error_msg}