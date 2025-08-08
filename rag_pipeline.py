"""
RAG Pipeline for NoteWise application.
Handles embeddings, vector storage, retrieval, and LLM interactions using Groq.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.schema import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipeline:
    """RAG Pipeline for document processing and question answering using Groq."""
    
    def __init__(self, model_name: str = "llama3-8b-8192", temperature: float = 0.7):
        """
        Initialize RAG Pipeline.
        
        Args:
            model_name: Groq model name to use (e.g., "llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768")
            temperature: Temperature for LLM responses
        """
        self.model_name = model_name
        self.temperature = temperature
        self.embeddings = None
        self.vector_store = None
        self.qa_chain = None
        self.memory = None
        
        # Initialize components
        self._initialize_embeddings()
        self._initialize_memory()
    
    def _initialize_embeddings(self):
        """Initialize sentence-transformers embeddings."""
        try:
            # Use sentence-transformers for embeddings (free and local)
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            logger.info("Sentence-transformers embeddings initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            raise Exception(f"Embeddings initialization failed: {str(e)}")
    
    def _initialize_memory(self):
        """Initialize conversation memory."""
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
    
    def create_vector_store(self, documents: List[Document]) -> FAISS:
        """
        Create FAISS vector store from documents.
        
        Args:
            documents: List of Document objects
            
        Returns:
            FAISS vector store
        """
        try:
            if not documents:
                raise ValueError("No documents provided for vector store creation")
            
            logger.info(f"Creating vector store with {len(documents)} documents")
            
            # Create FAISS vector store
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            
            logger.info("Vector store created successfully")
            return self.vector_store
            
        except Exception as e:
            logger.error(f"Failed to create vector store: {e}")
            raise Exception(f"Vector store creation failed: {str(e)}")
    
    def _create_qa_prompt(self) -> PromptTemplate:
        """Create custom prompt template for QA."""
        template = """You are a helpful AI assistant that answers questions based on the provided context from PDF documents.

Context: {context}

Chat History: {chat_history}

Question: {question}

Please provide a comprehensive answer based on the context provided. If the answer cannot be found in the context, say so. Always cite the source when possible.

Answer:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "chat_history", "question"]
        )
    
    def setup_qa_chain(self) -> ConversationalRetrievalChain:
        """
        Set up the conversational retrieval chain.
        
        Returns:
            ConversationalRetrievalChain
        """
        try:
            if not self.vector_store:
                raise ValueError("Vector store not initialized")
            
            # Check for Groq API key
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY not found in environment variables")
            
            # Initialize Groq LLM
            llm = ChatGroq(
                model_name=self.model_name,
                temperature=self.temperature,
                groq_api_key=api_key
            )
            
            # Create custom prompt
            qa_prompt = self._create_qa_prompt()
            
            # Create retrieval chain
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=self.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 3}
                ),
                memory=self.memory,
                return_source_documents=True,
                combine_docs_chain_kwargs={"prompt": qa_prompt}
            )
            
            logger.info("QA chain setup successfully with Groq")
            return self.qa_chain
            
        except Exception as e:
            logger.error(f"Failed to setup QA chain: {e}")
            raise Exception(f"QA chain setup failed: {str(e)}")
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """
        Ask a question and get answer with sources.
        
        Args:
            question: User's question
            
        Returns:
            Dictionary containing answer and source documents
        """
        try:
            if not self.qa_chain:
                self.setup_qa_chain()
            
            logger.info(f"Processing question: {question}")
            
            # Get response from QA chain
            response = self.qa_chain({"question": question})
            
            # Extract answer and sources
            answer = response.get("answer", "")
            source_documents = response.get("source_documents", [])
            
            # Process source documents
            sources = []
            for doc in source_documents:
                source_info = {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "page_number": doc.metadata.get("page_number", "Unknown")
                }
                sources.append(source_info)
            
            result = {
                "answer": answer,
                "sources": sources,
                "question": question
            }
            
            logger.info(f"Generated answer with {len(sources)} sources")
            return result
            
        except Exception as e:
            logger.error(f"Failed to process question: {e}")
            return {
                "answer": f"Sorry, I encountered an error while processing your question: {str(e)}",
                "sources": [],
                "question": question
            }
    
    def save_vector_store(self, file_path: str):
        """
        Save vector store to disk.
        
        Args:
            file_path: Path to save the vector store
        """
        try:
            if self.vector_store:
                self.vector_store.save_local(file_path)
                logger.info(f"Vector store saved to {file_path}")
            else:
                logger.warning("No vector store to save")
                
        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")
            raise Exception(f"Vector store save failed: {str(e)}")
    
    def load_vector_store(self, file_path: str):
        """
        Load vector store from disk.
        
        Args:
            file_path: Path to load the vector store from
        """
        try:
            if os.path.exists(file_path):
                self.vector_store = FAISS.load_local(file_path, self.embeddings)
                logger.info(f"Vector store loaded from {file_path}")
            else:
                logger.warning(f"Vector store file not found at {file_path}")
                
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            raise Exception(f"Vector store load failed: {str(e)}")
    
    def clear_memory(self):
        """Clear conversation memory."""
        if self.memory:
            self.memory.clear()
            logger.info("Conversation memory cleared")
    
    def get_vector_store_info(self) -> Dict[str, Any]:
        """
        Get information about the current vector store.
        
        Returns:
            Dictionary with vector store information
        """
        if not self.vector_store:
            return {"status": "not_initialized"}
        
        try:
            # Get basic info
            info = {
                "status": "initialized",
                "index_type": "FAISS",
                "embedding_dimension": self.embeddings.client.dimensions if hasattr(self.embeddings, 'client') else "unknown"
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get vector store info: {e}")
            return {"status": "error", "error": str(e)}
