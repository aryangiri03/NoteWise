import fitz
import os
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFProcessor:
    """Handles PDF processing and text extraction."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize PDF processor.
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract text from PDF file with page information.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        try:
            doc = fitz.open(pdf_path)
            text_content = []
            page_info = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                
                if text.strip():  # Only add non-empty pages
                    text_content.append(text)
                    page_info.append({
                        'page_number': page_num + 1,
                        'text_length': len(text),
                        'text': text
                    })
            
            doc.close()
            
            return {
                'full_text': '\n\n'.join(text_content),
                'pages': page_info,
                'total_pages': len(page_info),
                'total_text_length': len('\n\n'.join(text_content))
            }
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise Exception(f"Failed to extract text from PDF: {str(e)}")
    
    def create_chunks(self, text_content: str, metadata: Dict[str, Any] = None) -> List[Document]:
        """
        Split text into chunks using LangChain text splitter.
        
        Args:
            text_content: Full text content to split
            metadata: Additional metadata for the chunks
            
        Returns:
            List of Document objects
        """
        try:
            # Create documents with metadata
            documents = [Document(page_content=text_content, metadata=metadata or {})]
            
            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            logger.info(f"Created {len(chunks)} chunks from text content")
            return chunks
            
        except Exception as e:
            logger.error(f"Error creating chunks: {e}")
            raise Exception(f"Failed to create text chunks: {str(e)}")

def validate_pdf_file(uploaded_file) -> bool:
    
    if uploaded_file is None:
        return False
    
    
    if not uploaded_file.name.lower().endswith('.pdf'):
        return False
        
    if uploaded_file.size > 50 * 1024 * 1024:
        return False
    
    return True

def format_source_reference(page_number: int, chunk_text: str, max_length: int = 200) -> str:
    
    preview = chunk_text[:max_length] + "..." if len(chunk_text) > max_length else chunk_text
    return f"Page {page_number}: {preview}"

def clean_text(text: str) -> str:
    
    import re
        
    text = re.sub(r'\s+', ' ', text)        
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]]', '', text)
    
    return text.strip()

def get_file_size_mb(file_size_bytes: int) -> float:    
    return round(file_size_bytes / (1024 * 1024), 2)