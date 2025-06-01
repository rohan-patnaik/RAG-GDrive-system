import sys
import os
sys.path.append('netlify/functions')

from utils.config import config
from health import handler as health_handler
from query import handler as query_handler

# Test health endpoint
def test_health():
    event = {'httpMethod': 'GET'}
    result = health_handler(event, {})
    print("Health check:", result)

# Test query endpoint  
def test_query():
    event = {
        'httpMethod': 'POST',
        'body': '{"query_text": "What is this about?"}',
        'headers': {'authorization': 'Bearer your-api-key'}
    }
    result = query_handler(event, {})
    print("Query result:", result)

if __name__ == "__main__":
    # Set environment variables for testing
    os.environ['OPENAI_API_KEY'] = 'your-key-here'
    os.environ['PINECONE_API_KEY'] = 'your-key-here'
    os.environ['API_KEY'] = 'test-api-key'
    
    test_health()
    # test_query()  # Uncomment when ready