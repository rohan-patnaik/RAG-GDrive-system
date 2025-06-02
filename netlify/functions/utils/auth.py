import json
from typing import Dict, Any, Optional
from .config import config

def extract_api_key(event: Dict[str, Any]) -> Optional[str]:
    """Extract API key from request"""
    headers = event.get('headers', {})
    
    # Try Authorization header
    auth_header = headers.get('authorization') or headers.get('Authorization')
    if auth_header:
        if auth_header.startswith('Bearer '):
            return auth_header[7:]
        elif auth_header.startswith('ApiKey '):
            return auth_header[7:]
    
    # Try X-API-Key header
    api_key = headers.get('x-api-key') or headers.get('X-API-Key')
    if api_key:
        return api_key
    
    # Try query parameter
    query_params = event.get('queryStringParameters') or {}
    return query_params.get('api_key')

def verify_api_key(event: Dict[str, Any]) -> Dict[str, Any]:
    """Verify system API key"""
    api_key = extract_api_key(event)
    
    if not api_key:
        return {
            'valid': False,
            'message': 'API key required',
            'code': 'MISSING_API_KEY'
        }
    
    if config.verify_system_api_key(api_key):
        return {
            'valid': True,
            'client_id': 'frontend',
            'permissions': ['query', 'ingest', 'health']
        }
    
    return {
        'valid': False,
        'message': 'Invalid API key',
        'code': 'INVALID_API_KEY'
    }