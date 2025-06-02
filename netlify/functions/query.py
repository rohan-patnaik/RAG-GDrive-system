import json
import asyncio
from .utils.config import config
from .utils.auth import verify_api_key
from .utils.embeddings import EmbeddingService
from .utils.vector_store import VectorStore
from .utils.llm_service import LLMService

async def process_query(query_text: str, llm_provider: str = None) -> dict:
    """Process RAG query"""
    
    # Initialize services
    embedding_service = EmbeddingService()
    vector_store = VectorStore()
    llm_service = LLMService()
    
    # Generate query embedding
    query_embedding = embedding_service.encode_text(query_text)
    
    # Search for relevant chunks
    chunks = vector_store.search(query_embedding)
    
    # Generate response
    provider = llm_provider or config.DEFAULT_LLM_PROVIDER
    context = '\n\n'.join([chunk['content'] for chunk in chunks])
    
    if not context.strip():
        return {
            'query_text': query_text,
            'llm_answer': "I couldn't find relevant information in the knowledge base to answer your question.",
            'llm_provider_used': provider,
            'llm_model_used': 'N/A',
            'retrieved_chunks': []
        }
    
    llm_response = await llm_service.generate_response(query_text, context, provider)
    
    return {
        'query_text': query_text,
        'llm_answer': llm_response['answer'],
        'llm_provider_used': llm_response['provider'],
        'llm_model_used': llm_response['model'],
        'retrieved_chunks': chunks
    }

def handler(event, context):
    """Query endpoint handler"""
    
    if event.get('httpMethod') == 'OPTIONS':
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
        
        # Parse request
        body = json.loads(event.get('body', '{}'))
        query_text = body.get('query_text', '').strip()
        llm_provider = body.get('llm_provider')
        
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
        result = asyncio.run(process_query(query_text, llm_provider))
        
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