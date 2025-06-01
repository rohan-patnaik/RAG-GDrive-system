import re
from typing import List
from io import BytesIO
import PyPDF2

class TextProcessor:
    """Text processing utilities for document ingestion"""
    
    def __init__(self):
        self.sentence_endings = re.compile(r'[.!?]+\s+')
        self.paragraph_breaks = re.compile(r'\n\s*\n')
    
    def extract_from_pdf(self, pdf_content: bytes) -> str:
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except ImportError:
            raise ImportError("PyPDF2 not available. Install with: pip install PyPDF2")
    
    def extract_from_docx(self, docx_content: bytes) -> str:
        """Extract text from DOCX file"""
        try:
            import docx
            doc = docx.Document(BytesIO(docx_content))
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except ImportError:
            raise ImportError("python-docx not available. Install with: pip install python-docx")
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-"\']', '', text)
        
        # Fix common OCR errors
        text = text.replace('—', '-')
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        return text.strip()
    
    def split_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        
        # Clean text first
        text = self.clean_text(text)
        
        # Split by paragraphs first
        paragraphs = self.paragraph_breaks.split(text)
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # If adding this paragraph would exceed chunk size
            if len(current_chunk) + len(paragraph) > chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    
                    # Create overlap from end of current chunk
                    if overlap > 0:
                        overlap_text = self._get_overlap_text(current_chunk, overlap)
                        current_chunk = overlap_text + " " + paragraph
                    else:
                        current_chunk = paragraph
                else:
                    # Paragraph itself is too long, split by sentences
                    sentence_chunks = self._split_by_sentences(paragraph, chunk_size, overlap)
                    chunks.extend(sentence_chunks[:-1])  # Add all but last
                    current_chunk = sentence_chunks[-1] if sentence_chunks else ""
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """Get overlap text from end of chunk"""
        if len(text) <= overlap_size:
            return text
        
        # Try to find a good breaking point (sentence or word boundary)
        overlap_text = text[-overlap_size:]
        
        # Find the first sentence boundary
        sentence_match = self.sentence_endings.search(overlap_text)
        if sentence_match:
            return overlap_text[sentence_match.end():].strip()
        
        # Find the first word boundary
        space_index = overlap_text.find(' ')
        if space_index > 0:
            return overlap_text[space_index:].strip()
        
        return overlap_text
    
    def _split_by_sentences(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split long paragraph by sentences"""
        sentences = self.sentence_endings.split(text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            if len(current_chunk) + len(sentence) > chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    
                    if overlap > 0:
                        overlap_text = self._get_overlap_text(current_chunk, overlap)
                        current_chunk = overlap_text + " " + sentence
                    else:
                        current_chunk = sentence
                else:
                    # Single sentence is too long, split by words
                    word_chunks = self._split_by_words(sentence, chunk_size, overlap)
                    chunks.extend(word_chunks[:-1])
                    current_chunk = word_chunks[-1] if word_chunks else ""
            else:
                if current_chunk:
                    current_chunk += ". " + sentence
                else:
                    current_chunk = sentence
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _split_by_words(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split very long text by words (last resort)"""
        words = text.split()
        
        chunks = []
        current_chunk = ""
        
        for word in words:
            if len(current_chunk) + len(word) + 1 > chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    
                    if overlap > 0:
                        # Take last few words for overlap
                        chunk_words = current_chunk.split()
                        overlap_words = min(overlap // 10, len(chunk_words))  # Rough estimate
                        if overlap_words > 0:
                            current_chunk = " ".join(chunk_words[-overlap_words:]) + " " + word
                        else:
                            current_chunk = word
                    else:
                        current_chunk = word
                else:
                    # Single word is too long (shouldn't happen normally)
                    chunks.append(word)
                    current_chunk = ""
            else:
                if current_chunk:
                    current_chunk += " " + word
                else:
                    current_chunk = word
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks