import openai
import requests
import json
from typing import Dict, Any, Optional
from .config import config

class LLMService:
    """Unified LLM service for multiple providers"""
    
    def __init__(self):
        self.openai_client = None
        if config.OPENAI_API_KEY:
            self.openai_client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
    
    async def generate_response(self, query: str, context: str, provider: str = 'openai') -> Dict[str, Any]:
        """Generate response using specified LLM provider"""
        
        if provider == 'openai':
            return await self._generate_openai_response(query, context)
        elif provider == 'anthropic':
            return await self._generate_anthropic_response(query, context)
        elif provider == 'google':
            return await self._generate_google_response(query, context)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    async def _generate_openai_response(self, query: str, context: str) -> Dict[str, Any]:
        """Generate response using OpenAI"""
        if not self.openai_client:
            raise ValueError("OpenAI API key not configured")
        
        prompt = f"""Based on the following context, answer the user's question.

Context:
{context}

Question: {query}

Answer:"""
        
        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        return {
            'answer': response.choices[0].message.content,
            'model': 'gpt-3.5-turbo',
            'provider': 'openai'
        }
    
    async def _generate_anthropic_response(self, query: str, context: str) -> Dict[str, Any]:
        """Generate response using Anthropic Claude"""
        if not config.ANTHROPIC_API_KEY:
            raise ValueError("Anthropic API key not configured")
        
        prompt = f"""Based on the following context, answer the user's question.

Context:
{context}

Question: {query}

Answer:"""
        
        headers = {
            'x-api-key': config.ANTHROPIC_API_KEY,
            'content-type': 'application/json',
            'anthropic-version': '2023-06-01'
        }
        
        data = {
            'model': 'claude-3-sonnet-20240229',
            'max_tokens': 500,
            'messages': [
                {'role': 'user', 'content': prompt}
            ]
        }
        
        response = requests.post(
            'https://api.anthropic.com/v1/messages',
            headers=headers,
            json=data
        )
        
        if response.status_code != 200:
            raise ValueError(f"Anthropic API error: {response.status_code}")
        
        result = response.json()
        
        return {
            'answer': result['content'][0]['text'],
            'model': 'claude-3-sonnet-20240229',
            'provider': 'anthropic'
        }
    
    async def _generate_google_response(self, query: str, context: str) -> Dict[str, Any]:
        """Generate response using Google Gemini"""
        if not config.GOOGLE_API_KEY:
            raise ValueError("Google API key not configured")
        
        prompt = f"""Based on the following context, answer the user's question.

Context:
{context}

Question: {query}

Answer:"""
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={config.GOOGLE_API_KEY}"
        
        headers = {'Content-Type': 'application/json'}
        data = {
            'contents': [{
                'parts': [{'text': prompt}]
            }],
            'generationConfig': {
                'temperature': 0.7,
                'maxOutputTokens': 500
            }
        }
        
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code != 200:
            raise ValueError(f"Google API error: {response.status_code}")
        
        result = response.json()
        
        return {
            'answer': result['candidates'][0]['content']['parts'][0]['text'],
            'model': 'gemini-pro',
            'provider': 'google'
        }