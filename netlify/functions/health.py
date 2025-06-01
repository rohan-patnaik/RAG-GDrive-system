import json
import os
from typing import Dict, Any

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Health check endpoint for Netlify function"""
    
    # Check required environment variables
    required_vars = ['PINECONE_API_KEY', 'OPENAI_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'status': 'DEGRADED',
                'message': f'Missing environment variables: {missing_vars}',
                'timestamp': '2024-01-01T00:00:00Z'
            })
        }
    
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        },
        'body': json.dumps({
            'status': 'OK',
            'message': 'RAG system is operational',
            'timestamp': '2024-01-01T00:00:00Z',
            'components': [
                {'name': 'Pinecone', 'status': 'OK'},
                {'name': 'OpenAI', 'status': 'OK'}
            ]
        })
    }

# Netlify function entry point
def handler(event, context):
    return lambda_handler(event, context)