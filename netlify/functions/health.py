import json
import os
from .utils.config import config
from .utils.auth import verify_api_key

def handler(event, context):
    """Health check with authentication"""
    
    if event.get('httpMethod') == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-API-Key',
                'Access-Control-Allow-Methods': 'GET, OPTIONS'
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
        
        # Check system components
        components = []
        
        # Check LLM providers
        if config.OPENAI_API_KEY:
            components.append({'name': 'OpenAI', 'status': 'OK'})
        if config.GOOGLE_API_KEY:
            components.append({'name': 'Google Gemini', 'status': 'OK'})
        if config.ANTHROPIC_API_KEY:
            components.append({'name': 'Anthropic', 'status': 'OK'})
        
        # Check vector store
        if config.PINECONE_API_KEY:
            components.append({'name': 'Pinecone', 'status': 'OK'})
        
        status = 'OK' if components else 'DEGRADED'
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'status': status,
                'message': 'RAG system operational',
                'components': components,
                'default_provider': config.DEFAULT_LLM_PROVIDER
            })
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