"""
Chutes provider for inference requests.
"""

import json
import logging
import random
from typing import List
from uuid import UUID

import httpx

from .base import InferenceProvider
from proxy.models import GPTMessage
from proxy.config import (
    CHUTES_API_KEYS,
    CHUTES_INFERENCE_URL,
    MODEL_PRICING,
)

logger = logging.getLogger(__name__)


class ChutesProvider(InferenceProvider):
    """Provider for Chutes API inference"""
    
    def __init__(self):
        self.api_keys = CHUTES_API_KEYS
        self.current_key_index = 0
        self.failed_keys = set()  # Track keys that have failed recently
        
    @property
    def name(self) -> str:
        return "Chutes"
    
    def is_available(self) -> bool:
        """Check if Chutes provider is available"""
        return len(self.api_keys) > 0
    
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
    
    def supports_model(self, model: str) -> bool:
        """Check if model is supported by Chutes (supports all models in pricing)"""
        return model in MODEL_PRICING
    
    def get_pricing(self, model: str) -> float:
        """Get Chutes pricing for the model"""
        if not self.supports_model(model):
            raise KeyError(f"Model {model} not supported by Chutes provider")
        return MODEL_PRICING[model]
    
    async def inference(
        self,
        run_id: UUID = None,
        messages: List[GPTMessage] = None,
        temperature: float = None,
        model: str = None,
    ) -> tuple[str, int]:
        """Perform inference using Chutes API"""
        
        if not self.is_available():
            raise RuntimeError("No Chutes API keys available")
            
        if not self.supports_model(model):
            raise ValueError(f"Model {model} not supported by Chutes provider")
        
        # Convert messages to dict format
        messages_dict = []
        if messages:
            for message in messages:
                if message:
                    messages_dict.append({"role": message.role, "content": message.content})

        # Try with current API key, rotate if it fails
        max_retries = len(self.api_keys)
        for attempt in range(max_retries):
            try:
                current_api_key = self.get_current_api_key()
                headers = {
                    "Authorization": f"Bearer {current_api_key}",
                    "Content-Type": "application/json",
                }
                
                return await self._make_inference_request(headers, messages_dict, model, temperature, run_id)
                
            except Exception as e:
                error_msg = str(e).lower()
                # Check if this is a rate limit or quota exceeded error
                if any(keyword in error_msg for keyword in ['rate limit', 'quota', 'limit exceeded', '429', 'too many requests']):
                    logger.warning(f"Rate limit hit with API key, rotating to next key: {current_api_key[:10]}...")
                    self.mark_key_as_failed(current_api_key)
                    continue
                else:
                    # For non-rate-limit errors, re-raise immediately
                    raise e
        
        # If we've exhausted all keys, raise an error
        raise RuntimeError("All Chutes API keys have been exhausted due to rate limits")
    
    async def _make_inference_request(self, headers: dict, messages_dict: list, model: str, temperature: float, run_id: UUID) -> tuple[str, int]:
        """Make the actual inference request to Chutes API"""
        body = {
            "model": model,
            "messages": messages_dict,
            "stream": True,
            "max_tokens": 2048,
            "temperature": temperature,
            "seed": random.randint(0, 2**32 - 1),
        }

        # logger.debug(f"Chutes inference request for run {run_id} with model {model}")

        response_text = ""
        
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", CHUTES_INFERENCE_URL, headers=headers, json=body) as response:

                if response.status_code != 200:
                    error_text = await response.aread()
                    if isinstance(error_text, bytes):
                        error_message = error_text.decode()
                    else:
                        error_message = str(error_text)
                    logger.error(
                        f"Chutes API request failed for run {run_id} (model: {model}): {response.status_code} - {error_message}"
                    )
                    # Raise an exception that can be caught by the retry logic
                    raise RuntimeError(f"HTTP {response.status_code}: {error_message}")

                # Process streaming response
                async for chunk in response.aiter_lines():
                    if chunk:
                        chunk_str = chunk.strip()
                        if chunk_str.startswith("data: "):
                            chunk_data = chunk_str[6:]  # Remove "data: " prefix

                            if chunk_data == "[DONE]":
                                break

                            try:
                                chunk_json = json.loads(chunk_data)
                                if "choices" in chunk_json and len(chunk_json["choices"]) > 0:
                                    choice = chunk_json["choices"][0]
                                    if "delta" in choice and "content" in choice["delta"]:
                                        content = choice["delta"]["content"]
                                        if content:
                                            response_text += content

                            except json.JSONDecodeError:
                                # Skip malformed JSON chunks
                                continue

        # logger.debug(f"Chutes inference for run {run_id} completed")
        
        # Validate that we received actual content
        if not response_text.strip():
            # Don't care too much about empty responses for now
            error_msg = f"Chutes API returned empty response for model {model}. This may indicate API issues or malformed streaming response."
            # logger.error(f"Empty response for run {run_id}: {error_msg}")
            
            return error_msg, 200  # Status was 200 but response was empty
        
        return response_text, 200 