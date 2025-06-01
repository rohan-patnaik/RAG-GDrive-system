import json
import asyncio
from typing import Dict, Any
from .utils.config import config
from .utils.embeddings import EmbeddingService
from .utils.vector_store import VectorStore
from .utils.llm_clients import LLMService
from .utils.auth import verify_api_key
from .utils.rate_limiter import RateLimiter
from .utils.cache import QueryCache, SemanticCache

async def process_query(query_text: str, llm_provider: str = None, use_cache: bool = True) -> Dict[str, Any]:
    """Enhanced query processing with caching"""
    
    provider = llm_provider or config.DEFAULT_LLM_PROVIDER
    
    # Check cache first
    if use_cache and config.ENABLE_CACHING:
        if config.ENABLE_SEMANTIC_CACHE:
            cache = SemanticCache(
                default_ttl=config.CACHE_TTL,
                similarity_threshold=config.SEMANTIC_CACHE_THRESHOLD
            )
            cached_response = cache.get_semantic(query_text, provider)
        else:
            cache = QueryCache(default_ttl=config.CACHE_TTL)
            cached_response = cache.get(query_text, provider)
        
        if cached_response:
            cached_response['from_cache'] = True
            return cached_response
    
    # Initialize services
    embedding_service = EmbeddingService()
    vector_store = VectorStore()
    llm_service = LLMService()
    
    # Generate query embedding
    query_embedding = embedding_service.encode_text(query_text)
    
    # Search for relevant chunks
    chunks = vector_store.search(query_embedding, top_k=config.TOP_K_RESULTS)
    
    # Filter by similarity threshold
    filtered_chunks = [
        chunk for chunk in chunks 
        if chunk['score'] >= config.SIMILARITY_THRESHOLD
    ]
    
    # Generate response using LLM
    context = '\n\n'.join([chunk['content'] for chunk in filtered_chunks])
    
    llm_response = await llm_service.generate_response(
        query=query_text,
        context=context,
        provider=provider
    )
    
    result = {
        'query_text': query_text,
        'llm_answer': llm_response['answer'],
        'llm_provider_used': provider,
        'llm_model_used': llm_response['model'],
        'retrieved_chunks': filtered_chunks,
        'from_cache': False
    }
    
    # Cache the response
    if use_cache and config.ENABLE_CACHING:
        if config.ENABLE_SEMANTIC_CACHE:
            cache.set_semantic(query_text, result, provider)
        else:
            cache.set(query_text, result, provider)
    
    return result

def handler(event, context):
    """Enhanced Netlify function handler with auth, rate limiting, and caching"""
    
    # Handle CORS preflight
    if event['httpMethod'] == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-API-Key',
                'Access-Control-Allow-Methods': 'POST, OPTIONS'
            },
            'body': ''
        }
    
    try:
        # Verify API key
        auth_result = verify_api_key(event)
        if not auth_result['valid']:
            return {
                'statusCode': 401,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({'error': auth_result['message']})
            }
        
        # Check rate limit
        if config.ENABLE_RATE_LIMITING:
            client_id = auth_result.get('client_id', 'anonymous')
            rate_limiter = RateLimiter()
            
            if not rate_limiter.check_limit(client_id, 'query'):
                limit_info = rate_limiter.get_limit_info(client_id, 'query')
                return {
                    'statusCode': 429,
                    'headers': {
                        'Content-Type': 'application/json',
                        'Access-Control-Allow-Origin': '*',
                        'X-RateLimit-Limit': str(limit_info['limit']),
                        'X-RateLimit-Remaining': str(limit_info['remaining']),
                        'X-RateLimit-Reset': str(limit_info['reset_time'])
                    },
                    'body': json.dumps({'error': 'Rate limit exceeded'})
                }
        
        # Parse request body
        body = json.loads(event['body'])
        query_text = body.get('query_text', '')
        llm_provider = body.get('llm_provider')
        use_cache = body.get('use_cache', True)
        
        if not query_text:
            return {
                'statusCode': 400,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({'error': 'query_text is required'})
            }
        
        # Process query
        result = asyncio.run(process_query(query_text, llm_provider, use_cache))
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(result)
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({'error': str(e)})
        }