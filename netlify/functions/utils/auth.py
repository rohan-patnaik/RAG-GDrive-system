import os
import json
import hashlib
# import hmac # Not used if not generating signed tokens
import time
from typing import Dict, Any, Optional
from .config import config # Ensure this import works based on your structure

class AuthService:
    def __init__(self):
        self.expected_backend_access_key = config.BACKEND_ACCESS_KEY
        # For more complex multi-client scenarios:
        self.api_keys_config = self._load_api_keys_json()

    def _load_api_keys_json(self) -> Dict[str, Dict[str, Any]]:
        """Load multiple API keys from JSON string in env var API_KEYS_JSON"""
        api_keys_json_str = config.API_KEYS_JSON
        if api_keys_json_str:
            try:
                return json.loads(api_keys_json_str)
            except json.JSONDecodeError:
                print("Warning: API_KEYS_JSON is invalid. Falling back to single API_KEY.")
                return {}
        return {}

    def verify_api_key(self, submitted_key: str) -> Dict[str, Any]:
        if not submitted_key:
            return {'valid': False, 'message': 'API key required'}

        # 1. Check against the single, primary BACKEND_ACCESS_KEY
        if self.expected_backend_access_key and submitted_key == self.expected_backend_access_key:
            return {
                'valid': True,
                'client_id': 'primary_access_client', # Generic ID for this key
                'permissions': ['query', 'ingest', 'health'], # Default permissions
                'rate_limits': {} # Or define default rate limits
            }

        # 2. (Optional) Check against a list of client-specific keys if API_KEYS_JSON is used
        if submitted_key in self.api_keys_config:
            client_info = self.api_keys_config[submitted_key]
            return {
                'valid': True,
                'client_id': client_info.get('client_id', 'unknown_client'),
                'permissions': client_info.get('permissions', []),
                'rate_limits': client_info.get('rate_limits', {})
            }

        return {'valid': False, 'message': 'Invalid API key'}

    # ... (check_permission and generate_api_key can remain if you plan to use them)

def verify_api_key(event: Dict[str, Any]) -> Dict[str, Any]:
    headers = event.get('headers', {})
    auth_header = headers.get('authorization') or headers.get('Authorization')
    submitted_key: Optional[str] = None

    if auth_header:
        if auth_header.startswith('Bearer '):
            submitted_key = auth_header[7:]
        # You might have other schemes, e.g., 'ApiKey '
        elif auth_header.startswith('ApiKey '): # If you use this scheme
             submitted_key = auth_header[7:]


    if not submitted_key:
        submitted_key = headers.get('x-api-key') or headers.get('X-API-Key')

    if not submitted_key:
        query_params = event.get('queryStringParameters') or {}
        submitted_key = query_params.get('api_key')

    auth_service = AuthService()
    return auth_service.verify_api_key(submitted_key)