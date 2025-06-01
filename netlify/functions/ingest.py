import json
import uuid
import hashlib
import asyncio
from typing import Dict, Any, List
from io import BytesIO
import mimetypes
from .utils.config import config
from .utils.embeddings import EmbeddingService
from .utils.vector_store import VectorStore
from .utils.auth import verify_api_key
from .utils.rate_limiter import RateLimiter
from .utils.text_processor import TextProcessor

class DocumentProcessor:
    """Process and chunk documents for ingestion"""
    
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()
        self.text_processor = TextProcessor()
    
    def extract_text_from_file(self, file_content: bytes, filename: str) -> str:
        """Extract text from uploaded file"""
        mime_type, _ = mimetypes.guess_type(filename)
        
        if mime_type == 'text/plain':
            return file_content.decode('utf-8', errors='ignore')
        elif mime_type == 'application/pdf':
            return self.text_processor.extract_from_pdf(file_content)
        elif mime_type in ['application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
            return self.text_processor.extract_from_docx(file_content)
        else:
            raise ValueError(f"Unsupported file type: {mime_type}")
    
    def create_chunks(self, text: str, filename: str, source: str = None) -> List[Dict[str, Any]]:
        """Split text into chunks for embedding"""
        chunks = self.text_processor.split_text(text, chunk_size=1000, overlap=200)
        
        processed_chunks = []
        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{hashlib.md5((filename + str(i)).encode()).hexdigest()}"
            
            processed_chunks.append({
                'id': chunk_id,
                'content': chunk_text,
                'filename': filename,
                'source': source or filename,
                'chunk_index': i,
                'total_chunks': len(chunks)
            })
        
        return processed_chunks
    
    async def process_document(self, file_content: bytes, filename: str, source: str = None) -> Dict[str, Any]:
        """Complete document processing pipeline"""
        
        # Extract text
        text = self.extract_text_from_file(file_content, filename)
        
        # Create chunks
        chunks = self.create_chunks(text, filename, source)
        
        # Generate embeddings
        chunk_texts = [chunk['content'] for chunk in chunks]
        embeddings = self.embedding_service.encode_texts(chunk_texts)
        
        # Add embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk['embedding'] = embedding
        
        # Store in vector database
        self.vector_store.upsert_chunks(chunks)
        
        return {
            'filename': filename,
            'total_chunks': len(chunks),
            'total_characters': len(text),
            'status': 'success'
        }

def parse_multipart_form_data(body: bytes, content_type: str) -> Dict[str, Any]:
    """Parse multipart form data from request body"""
    import re
    
    # Extract boundary from content-type header
    boundary_match = re.search(r'boundary=([^;]+)', content_type)
    if not boundary_match:
        raise ValueError("No boundary found in content-type")
    
    boundary = boundary_match.group(1).strip('"')
    boundary_bytes = f'--{boundary}'.encode()
    
    # Split by boundary
    parts = body.split(boundary_bytes)
    
    files = {}
    fields = {}
    
    for part in parts[1:-1]:  # Skip first empty part and last boundary
        if not part.strip():
            continue
            
        # Find headers and content separator
        header_end = part.find(b'\r\n\r\n')
        if header_end == -1:
            continue
            
        headers = part[:header_end].decode('utf-8', errors='ignore')
        content = part[header_end + 4:]
        
        # Remove trailing CRLF
        if content.endswith(b'\r\n'):
            content = content[:-2]
        
        # Parse Content-Disposition header
        disposition_match = re.search(r'Content-Disposition: form-data; name="([^"]+)"(?:; filename="([^"]+)")?', headers)
        if not disposition_match:
            continue
            
        field_name = disposition_match.group(1)
        filename = disposition_match.group(2)
        
        if filename:
            # File field
            files[field_name] = {
                'filename': filename,
                'content': content
            }
        else:
            # Regular field
            fields[field_name] = content.decode('utf-8', errors='ignore')
    
    return {'files': files, 'fields': fields}

async def process_ingestion(event: Dict[str, Any]) -> Dict[str, Any]:
    """Process document ingestion request"""
    
    # Parse multipart form data
    content_type = event.get('headers', {}).get('content-type', '')
    body = event.get('body', '').encode() if event.get('isBase64Encoded') else event.get('body', '').encode()
    
    if 'multipart/form-data' not in content_type:
        raise ValueError("Content-Type must be multipart/form-data")
    
    form_data = parse_multipart_form_data(body, content_type)
    
    # Get uploaded files
    uploaded_files = form_data['files']
    source = form_data['fields'].get('source', '')
    
    if not uploaded_files:
        raise ValueError("No files uploaded")
    
    # Process each file
    processor = DocumentProcessor()
    results = []
    
    for field_name, file_info in uploaded_files.items():
        filename = file_info['filename']
        content = file_info['content']
        
        try:
            result = await processor.process_document(content, filename, source)
            results.append(result)
        except Exception as e:
            results.append({
                'filename': filename,
                'status': 'error',
                'error': str(e)
            })
    
    return {
        'message': f'Processed {len(results)} files',
        'results': results
    }

def handler(event, context):
    """Netlify function handler for document ingestion"""
    
    # Handle CORS preflight
    if event['httpMethod'] == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type, Authorization',
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
        client_id = auth_result.get('client_id', 'anonymous')
        rate_limiter = RateLimiter()
        
        if not rate_limiter.check_limit(client_id, 'ingest'):
            return {
                'statusCode': 429,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({'error': 'Rate limit exceeded'})
            }
        
        # Process ingestion
        result = asyncio.run(process_ingestion(event))
        
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