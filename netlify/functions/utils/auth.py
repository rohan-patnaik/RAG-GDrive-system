import os
import json
import hashlib
import hmac
import time
from typing import Dict, Any, Optional
from .config import config

class AuthService:
    """API key authentication service"""
    
    def __init__(self):
        # Load API keys from environment
        self.api_keys = self._load_api_keys()
        self.secret_key = config.SECRET_KEY or 'fallback-secret'
    
    def _load_api_keys(self) -> Dict[str, Dict[str, Any]]:
        """Load API keys from environment variables"""
        api_keys_json = os.getenv('API_KEYS', '{}')
        try:
            return json.loads(api_keys_json)
        except json.JSONDecodeError:
            # Fallback: single API key from environment
            single_key = os.getenv('API_KEY')
            if single_key:
                return {
                    single_key: {
                        'client_id': 'default',
                        'permissions': ['query', 'ingest'],
                        'rate_limits': {
                            'query': {'requests': 100, 'window': 3600},
                            'ingest': {'requests': 10, 'window': 3600}
                        }
                    }
                }
            return {}
    
    def verify_api_key(self, api_key: str) -> Dict[str, Any]:
        """Verify API key and return client info"""
        if not api_key:
            return {'valid': False, 'message': 'API key required'}
        
        if api_key not in self.api_keys:
            return {'valid': False, 'message': 'Invalid API key'}
        
        client_info = self.api_keys[api_key]
        return {
            'valid': True,
            'client_id': client_info['client_id'],
            'permissions': client_info.get('permissions', []),
            'rate_limits': client_info.get('rate_limits', {})
        }
    
    def generate_api_key(self, client_id: str) -> str:
        """Generate a new API key for a client"""
        timestamp = str(int(time.time()))
        raw_key = f"{client_id}:{timestamp}:{self.secret_key}"
        return hashlib.sha256(raw_key.encode()).hexdigest()
    
    def check_permission(self, client_info: Dict[str, Any], required_permission: str) -> bool:
        """Check if client has required permission"""
        permissions = client_info.get('permissions', [])
        return required_permission in permissions or 'admin' in permissions

def verify_api_key(event: Dict[str, Any]) -> Dict[str, Any]:
    """Extract and verify API key from request"""
    
    # Try Authorization header first
    headers = event.get('headers', {})
    auth_header = headers.get('authorization') or headers.get('Authorization')
    
    api_key = None
    if auth_header:
        if auth_header.startswith('Bearer '):
            api_key = auth_header[7:]
        elif auth_header.startswith('ApiKey '):
            api_key = auth_header[7:]
    
    # Try x-api-key header
    if not api_key:
        api_key = headers.get('x-api-key') or headers.get('X-API-Key')
    
    # Try query parameter
    if not api_key:
        query_params = event.get('queryStringParameters') or {}
        api_key = query_params.get('api_key')
    
    auth_service = AuthService()
    return auth_service.verify_api_key(api_key)