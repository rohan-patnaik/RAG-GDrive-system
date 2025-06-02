import openai
import google.generativeai as genai
from anthropic import Anthropic
from typing import Dict, Any
from .config import config

class LLMService:
    """Enhanced LLM service with all providers"""
    
    def __init__(self):
        # Initialize clients
        if config.OPENAI_API_KEY:
            self.openai_client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
        
        if config.GOOGLE_API_KEY:
            genai.configure(api_key=config.GOOGLE_API_KEY)
            self.gemini_model = genai.GenerativeModel('gemini-pro')
        
        if config.ANTHROPIC_API_KEY:
            self.anthropic_client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
    
    async def generate_response(self, query: str, context: str, provider: str = None) -> Dict[str, Any]:
        """Generate response using specified provider"""
        provider = provider or config.DEFAULT_LLM_PROVIDER
        
        if provider == 'openai':
            return await self._openai_response(query, context)
        elif provider == 'gemini':
            return await self._gemini_response(query, context)
        elif provider == 'anthropic':
            return await self._anthropic_response(query, context)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    async def _openai_response(self, query: str, context: str) -> Dict[str, Any]:
        """OpenAI response"""
        if not hasattr(self, 'openai_client'):
            raise ValueError("OpenAI not configured")
        
        prompt = f"""Based on the following context, answer the user's question accurately and concisely.

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
    
    async def _gemini_response(self, query: str, context: str) -> Dict[str, Any]:
        """Gemini response"""
        if not hasattr(self, 'gemini_model'):
            raise ValueError("Gemini not configured")
        
        prompt = f"""Based on the following context, answer the user's question accurately and concisely.

Context:
{context}

Question: {query}

Answer:"""
        
        response = self.gemini_model.generate_content(prompt)
        
        return {
            'answer': response.text,
            'model': 'gemini-pro',
            'provider': 'gemini'
        }
    
    async def _anthropic_response(self, query: str, context: str) -> Dict[str, Any]:
        """Anthropic response"""
        if not hasattr(self, 'anthropic_client'):
            raise ValueError("Anthropic not configured")
        
        prompt = f"""Based on the following context, answer the user's question accurately and concisely.

Context:
{context}

Question: {query}

Answer:"""
        
        response = self.anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=500,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return {
            'answer': response.content[0].text,
            'model': 'claude-3-haiku-20240307',
            'provider': 'anthropic'
        }