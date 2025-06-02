import json
from .utils.config import config

def handler(event, context):
    """Generate system API key for frontend"""
    
    # Only allow GET requests
    if event.get('httpMethod') != 'GET':
        return {
            'statusCode': 405,
            'headers': {'Access-Control-Allow-Origin': '*'},
            'body': json.dumps({'error': 'Method not allowed'})
        }
    
    try:
        # Generate system API key
        system_key = config.generate_system_api_key("frontend")
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'system_api_key': system_key,
                'message': 'Use this key for frontend authentication',
                'usage': 'Add as Authorization: Bearer <key> or X-API-Key: <key>'
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