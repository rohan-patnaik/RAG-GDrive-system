import time
import json
import hashlib
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

class RateLimiter:
    """In-memory rate limiter for serverless functions"""
    
    def __init__(self):
        # In production, this would use Redis or DynamoDB
        # For simplicity, using module-level storage (resets on cold start)
        self.storage = getattr(RateLimiter, '_storage', {})
        RateLimiter._storage = self.storage
        
        # Default rate limits
        self.default_limits = {
            'query': {'requests': 50, 'window': 3600},  # 50 requests per hour
            'ingest': {'requests': 5, 'window': 3600},   # 5 requests per hour
            'health': {'requests': 100, 'window': 3600}  # 100 requests per hour
        }
    
    def _get_client_key(self, client_id: str, operation: str) -> str:
        """Generate storage key for client and operation"""
        return f"rate_limit:{client_id}:{operation}"
    
    def _get_window_start(self, window_seconds: int) -> int:
        """Get the start of the current time window"""
        now = int(time.time())
        return (now // window_seconds) * window_seconds
    
    def check_limit(self, client_id: str, operation: str, custom_limits: Optional[Dict[str, Any]] = None) -> bool:
        """Check if client is within rate limits"""
        
        # Get rate limits (custom or default)
        limits = custom_limits or self.default_limits.get(operation, {'requests': 10, 'window': 3600})
        max_requests = limits['requests']
        window_seconds = limits['window']
        
        # Generate storage key
        key = self._get_client_key(client_id, operation)
        window_start = self._get_window_start(window_seconds)
        
        # Get current count for this window
        current_data = self.storage.get(key, {'count': 0, 'window_start': 0})
        
        # Reset count if we're in a new window
        if current_data['window_start'] != window_start:
            current_data = {'count': 0, 'window_start': window_start}
        
        # Check if limit exceeded
        if current_data['count'] >= max_requests:
            return False
        
        # Increment count
        current_data['count'] += 1
        self.storage[key] = current_data
        
        return True
    
    def get_limit_info(self, client_id: str, operation: str, custom_limits: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get current rate limit status for client"""
        
        limits = custom_limits or self.default_limits.get(operation, {'requests': 10, 'window': 3600})
        max_requests = limits['requests']
        window_seconds = limits['window']
        
        key = self._get_client_key(client_id, operation)
        window_start = self._get_window_start(window_seconds)
        
        current_data = self.storage.get(key, {'count': 0, 'window_start': 0})
        
        if current_data['window_start'] != window_start:
            current_data = {'count': 0, 'window_start': window_start}
        
        remaining = max(0, max_requests - current_data['count'])
        reset_time = window_start + window_seconds
        
        return {
            'limit': max_requests,
            'used': current_data['count'],
            'remaining': remaining,
            'reset_time': reset_time,
            'window_seconds': window_seconds
        }
    
    def reset_client_limits(self, client_id: str):
        """Reset all limits for a client (admin function)"""
        keys_to_remove = [key for key in self.storage.keys() if key.startswith(f"rate_limit:{client_id}:")]
        for key in keys_to_remove:
            del self.storage[key]

class DistributedRateLimiter:
    """Rate limiter using external storage (for production)"""
    
    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url
        # Implementation would use Redis for distributed rate limiting
        # For now, fallback to in-memory
        self.fallback = RateLimiter()
    
    def check_limit(self, client_id: str, operation: str, custom_limits: Optional[Dict[str, Any]] = None) -> bool:
        # In production, implement Redis-based rate limiting
        return self.fallback.check_limit(client_id, operation, custom_limits)
    
    def get_limit_info(self, client_id: str, operation: str, custom_limits: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.fallback.get_limit_info(client_id, operation, custom_limits)