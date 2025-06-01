import json
import hashlib
import time
from typing import Any, Optional, Dict
from datetime import datetime, timedelta

class QueryCache:
    """In-memory cache for query responses"""
    
    def __init__(self, default_ttl: int = 3600):
        # Module-level storage (persists across function invocations in same container)
        self.storage = getattr(QueryCache, '_storage', {})
        QueryCache._storage = self.storage
        self.default_ttl = default_ttl
    
    def _generate_cache_key(self, query_text: str, llm_provider: str = 'openai') -> str:
        """Generate cache key for query"""
        # Normalize query text
        normalized_query = query_text.lower().strip()
        key_string = f"{normalized_query}:{llm_provider}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _is_expired(self, cache_entry: Dict[str, Any]) -> bool:
        """Check if cache entry is expired"""
        return time.time() > cache_entry['expires_at']
    
    def get(self, query_text: str, llm_provider: str = 'openai') -> Optional[Dict[str, Any]]:
        """Get cached response for query"""
        cache_key = self._generate_cache_key(query_text, llm_provider)
        
        if cache_key not in self.storage:
            return None
        
        cache_entry = self.storage[cache_key]
        
        if self._is_expired(cache_entry):
            del self.storage[cache_key]
            return None
        
        # Update access time and hit count
        cache_entry['last_accessed'] = time.time()
        cache_entry['hit_count'] = cache_entry.get('hit_count', 0) + 1
        
        return cache_entry['data']
    
    def set(self, query_text: str, response_data: Dict[str, Any], llm_provider: str = 'openai', ttl: Optional[int] = None) -> None:
        """Cache response for query"""
        cache_key = self._generate_cache_key(query_text, llm_provider)
        expires_at = time.time() + (ttl or self.default_ttl)
        
        cache_entry = {
            'data': response_data,
            'created_at': time.time(),
            'last_accessed': time.time(),
            'expires_at': expires_at,
            'hit_count': 0,
            'query_text': query_text,
            'llm_provider': llm_provider
        }
        
        self.storage[cache_key] = cache_entry
        
        # Cleanup old entries if storage gets too large
        if len(self.storage) > 1000:
            self._cleanup_old_entries()
    
    def _cleanup_old_entries(self, max_entries: int = 500):
        """Remove oldest cache entries to prevent memory overflow"""
        # Sort by last_accessed time and keep only the most recent entries
        sorted_entries = sorted(
            self.storage.items(),
            key=lambda x: x[1]['last_accessed'],
            reverse=True
        )
        
        # Keep only the most recent entries
        new_storage = dict(sorted_entries[:max_entries])
        self.storage.clear()
        self.storage.update(new_storage)
    
    def invalidate(self, query_text: str, llm_provider: str = 'openai') -> bool:
        """Remove specific entry from cache"""
        cache_key = self._generate_cache_key(query_text, llm_provider)
        if cache_key in self.storage:
            del self.storage[cache_key]
            return True
        return False
    
    def clear_all(self):
        """Clear entire cache"""
        self.storage.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        now = time.time()
        total_entries = len(self.storage)
        expired_entries = sum(1 for entry in self.storage.values() if self._is_expired(entry))
        total_hits = sum(entry.get('hit_count', 0) for entry in self.storage.values())
        
        return {
            'total_entries': total_entries,
            'active_entries': total_entries - expired_entries,
            'expired_entries': expired_entries,
            'total_hits': total_hits,
            'cache_size_bytes': len(json.dumps(self.storage).encode())
        }

class SemanticCache(QueryCache):
    """Semantic cache that matches similar queries"""
    
    def __init__(self, default_ttl: int = 3600, similarity_threshold: float = 0.9):
        super().__init__(default_ttl)
        self.similarity_threshold = similarity_threshold
        self.embedding_service = None
    
    def _get_embedding_service(self):
        """Lazy load embedding service"""
        if self.embedding_service is None:
            from .embeddings import EmbeddingService
            self.embedding_service = EmbeddingService()
        return self.embedding_service
    
    def _cosine_similarity(self, vec1: list, vec2: list) -> float:
        """Calculate cosine similarity between two vectors"""
        import math
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def get_semantic(self, query_text: str, llm_provider: str = 'openai') -> Optional[Dict[str, Any]]:
        """Get cached response using semantic similarity"""
        
        # First try exact match
        exact_match = self.get(query_text, llm_provider)
        if exact_match:
            return exact_match
        
        # Generate embedding for query
        embedding_service = self._get_embedding_service()
        query_embedding = embedding_service.encode_text(query_text)
        
        # Search for semantically similar queries
        best_match = None
        best_similarity = 0
        
        for cache_key, cache_entry in self.storage.items():
            if self._is_expired(cache_entry):
                continue
            
            if cache_entry.get('llm_provider') != llm_provider:
                continue
            
            # Get or generate embedding for cached query
            if 'embedding' not in cache_entry:
                cached_query = cache_entry['query_text']
                cache_entry['embedding'] = embedding_service.encode_text(cached_query)
            
            similarity = self._cosine_similarity(query_embedding, cache_entry['embedding'])
            
            if similarity > best_similarity and similarity >= self.similarity_threshold:
                best_similarity = similarity
                best_match = cache_entry
        
        if best_match:
            # Update access stats
            best_match['last_accessed'] = time.time()
            best_match['hit_count'] = best_match.get('hit_count', 0) + 1
            best_match['semantic_match'] = True
            best_match['similarity_score'] = best_similarity
            return best_match['data']
        
        return None
    
    def set_semantic(self, query_text: str, response_data: Dict[str, Any], llm_provider: str = 'openai', ttl: Optional[int] = None) -> None:
        """Cache response with semantic indexing"""
        
        # Generate embedding for semantic matching
        embedding_service = self._get_embedding_service()
        query_embedding = embedding_service.encode_text(query_text)
        
        cache_key = self._generate_cache_key(query_text, llm_provider)
        expires_at = time.time() + (ttl or self.default_ttl)
        
        cache_entry = {
            'data': response_data,
            'created_at': time.time(),
            'last_accessed': time.time(),
            'expires_at': expires_at,
            'hit_count': 0,
            'query_text': query_text,
            'llm_provider': llm_provider,
            'embedding': query_embedding,
            'semantic_match': False
        }
        
        self.storage[cache_key] = cache_entry
        
        # Cleanup if needed
        if len(self.storage) > 1000:
            self._cleanup_old_entries()