"""
NoteWise - RAG-based AI Assistant
A voice and chat-enabled AI assistant for PDF document analysis.
"""

import streamlit as st
import os
import tempfile
import logging
from datetime import datetime
from typing import Dict, Any, List
import time

# Import custom modules
from utils import PDFProcessor, validate_pdf_file, format_source_reference, get_file_size_mb
from rag_pipeline import RAGPipeline
from voice_input import VoiceInput

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="NoteWise - AI Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .source-reference {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-radius: 5px;
        font-size: 0.9rem;
    }
    .sidebar-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-ready {
        background-color: #4caf50;
    }
    .status-processing {
        background-color: #ff9800;
    }
    .status-error {
        background-color: #f44336;
    }
</style>
""", unsafe_allow_html=True)

class NoteWiseApp:
    """Main NoteWise application class."""
    
    def __init__(self):
        """Initialize the application."""
        self.rag_pipeline = None
        self.voice_input = None
        self.pdf_processor = None
        
        # Initialize session state
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'current_pdf' not in st.session_state:
            st.session_state.current_pdf = None
        if 'vector_store_ready' not in st.session_state:
            st.session_state.vector_store_ready = False
        if 'processing_status' not in st.session_state:
            st.session_state.processing_status = "idle"
    
    def initialize_components(self):
        """Initialize RAG pipeline and voice input components."""
        try:
            if not self.rag_pipeline:
                self.rag_pipeline = RAGPipeline()
            if not self.voice_input:
                self.voice_input = VoiceInput()
            if not self.pdf_processor:
                self.pdf_processor = PDFProcessor()
                
        except Exception as e:
            st.error(f"Failed to initialize components: {str(e)}")
            logger.error(f"Component initialization failed: {e}")
    
    def render_header(self):
        """Render the main header."""
        st.markdown('<h1 class="main-header">📚 NoteWise</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Your AI Assistant for PDF Document Analysis</p>', unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render the sidebar with file upload and settings."""
        with st.sidebar:
            st.markdown("## 📁 Document Upload")
            
            # File uploader
            uploaded_file = st.file_uploader(
                "Choose a PDF file",
                type=['pdf'],
                help="Upload a PDF file to analyze (max 50MB)"
            )
            
            if uploaded_file is not None:
                if validate_pdf_file(uploaded_file):
                    self.handle_pdf_upload(uploaded_file)
                else:
                    st.error("Please upload a valid PDF file (max 50MB)")
            
            # Document info
            if st.session_state.current_pdf:
                st.markdown("### 📄 Current Document")
                st.info(f"**File:** {st.session_state.current_pdf['name']}")
                st.info(f"**Size:** {st.session_state.current_pdf['size']} MB")
                st.info(f"**Pages:** {st.session_state.current_pdf['pages']}")
                
                # Vector store status
                st.markdown("### 🔍 Vector Store Status")
                if st.session_state.vector_store_ready:
                    st.success("✅ Ready for questions")
                else:
                    st.warning("⏳ Processing document...")
            
            # Settings
            st.markdown("## ⚙️ Settings")
            
            # Voice recording duration
            recording_duration = st.slider(
                "Voice Recording Duration (seconds)",
                min_value=5,
                max_value=30,
                value=10,
                help="Duration for voice recording"
            )
            
            # Model selection
            model_name = st.selectbox(
                "AI Model",
                ["gpt-3.5-turbo", "gpt-4"],
                help="Select the AI model for responses"
            )
            
            # Clear chat button
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                self.clear_chat_history()
                st.success("Chat history cleared!")
            
            # About section
            st.markdown("## ℹ️ About")
            st.markdown("""
            **NoteWise** is an AI-powered assistant that helps you analyze PDF documents through natural language questions.
            
            **Features:**
            - 📄 PDF document analysis
            - 🎤 Voice input support (Hinglish-friendly)
            - 💬 Chat interface
            - 🔍 Source citations
            - 📊 Document insights
            
            **Tech Stack:**
            - LangChain + FAISS
            - Groq LLM models
            - Whisper STT
            - Streamlit UI
            """)
    
    def handle_pdf_upload(self, uploaded_file):
        """Handle PDF file upload and processing."""
        try:
            # Check if it's a new file
            if (st.session_state.current_pdf is None or 
                st.session_state.current_pdf['name'] != uploaded_file.name):
                
                st.session_state.processing_status = "processing"
                
                with st.spinner("Processing PDF document..."):
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    # Process PDF
                    pdf_info = self.pdf_processor.extract_text_from_pdf(tmp_file_path)
                    
                    # Create chunks
                    chunks = self.pdf_processor.create_chunks(
                        pdf_info['full_text'],
                        metadata={'source': uploaded_file.name}
                    )
                    
                    # Create vector store
                    self.rag_pipeline.create_vector_store(chunks)
                    
                    # Update session state
                    st.session_state.current_pdf = {
                        'name': uploaded_file.name,
                        'size': get_file_size_mb(uploaded_file.size),
                        'pages': pdf_info['total_pages'],
                        'chunks': len(chunks),
                        'text_length': pdf_info['total_text_length']
                    }
                    st.session_state.vector_store_ready = True
                    st.session_state.processing_status = "ready"
                    
                    # Clean up temporary file
                    os.unlink(tmp_file_path)
                    
                    st.success(f"✅ Document processed successfully! {len(chunks)} chunks created.")
                
        except Exception as e:
            st.error(f"Failed to process PDF: {str(e)}")
            st.session_state.processing_status = "error"
            logger.error(f"PDF processing failed: {e}")
    
    def render_chat_interface(self):
        """Render the main chat interface."""
        st.markdown("## 💬 Chat Interface")
        
        # Chat input section
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_input = st.text_input(
                "Ask a question about your document...",
                placeholder="e.g., What are the main topics discussed in this document?",
                key="user_input"
            )
        
        with col2:
            if st.button("🎤 Voice", use_container_width=True):
                self.handle_voice_input()
        
        # Process text input
        if user_input and st.button("Send", use_container_width=True):
            self.process_user_input(user_input)
        
        # Display chat history
        self.display_chat_history()
    
    def handle_voice_input(self):
        """Handle voice input and transcription."""
        try:
            if not st.session_state.vector_store_ready:
                st.error("Please upload and process a PDF document first.")
                return
            
            # Voice recording interface
            st.markdown("### 🎤 Voice Recording")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                if st.button("🎙️ Start Recording", use_container_width=True):
                    with st.spinner("Recording... Speak now!"):
                        # Record and transcribe
                        transcribed_text, audio_file_path = self.voice_input.record_and_transcribe(duration=10)
                        
                        # Clean up audio file
                        self.voice_input.cleanup_audio_file(audio_file_path)
                        
                        # Display transcribed text
                        st.success("🎯 Transcribed Text:")
                        st.write(transcribed_text)
                        
                        # Process the transcribed text
                        if transcribed_text.strip():
                            self.process_user_input(transcribed_text)
                        else:
                            st.warning("No speech detected. Please try again.")
            
        except Exception as e:
            st.error(f"Voice input failed: {str(e)}")
            logger.error(f"Voice input error: {e}")
    
    def process_user_input(self, user_input: str):
        """Process user input and generate response."""
        try:
            if not st.session_state.vector_store_ready:
                st.error("Please upload and process a PDF document first.")
                return
            
            # Add user message to chat history
            st.session_state.chat_history.append({
                'role': 'user',
                'content': user_input,
                'timestamp': datetime.now()
            })
            
            # Generate response
            with st.spinner("🤔 Thinking..."):
                response = self.rag_pipeline.ask_question(user_input)
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': response['answer'],
                'sources': response['sources'],
                'timestamp': datetime.now()
            })
            
            # Clear input
            st.session_state.user_input = ""
            
        except Exception as e:
            st.error(f"Failed to process question: {str(e)}")
            logger.error(f"Question processing failed: {e}")
    
    def display_chat_history(self):
        """Display the chat history."""
        if not st.session_state.chat_history:
            st.info("👋 Welcome! Upload a PDF and start asking questions.")
            return
        
        st.markdown("### 📝 Chat History")
        
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong> {message['content']}
                    <br><small>{message['timestamp'].strftime('%H:%M:%S')}</small>
                </div>
                """, unsafe_allow_html=True)
            
            elif message['role'] == 'assistant':
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>NoteWise:</strong> {message['content']}
                    <br><small>{message['timestamp'].strftime('%H:%M:%S')}</small>
                </div>
                """, unsafe_allow_html=True)
                
                # Display sources if available
                if 'sources' in message and message['sources']:
                    st.markdown("**📚 Sources:**")
                    for i, source in enumerate(message['sources'], 1):
                        page_num = source.get('page_number', 'Unknown')
                        content_preview = source.get('content', '')[:200] + "..." if len(source.get('content', '')) > 200 else source.get('content', '')
                        
                        st.markdown(f"""
                        <div class="source-reference">
                            <strong>Source {i} (Page {page_num}):</strong><br>
                            {content_preview}
                        </div>
                        """, unsafe_allow_html=True)
    
    def clear_chat_history(self):
        """Clear the chat history."""
        st.session_state.chat_history = []
        if self.rag_pipeline:
            self.rag_pipeline.clear_memory()
    
    def run(self):
        """Run the main application."""
        try:
            # Initialize components
            self.initialize_components()
            
            # Render header
            self.render_header()
            
            # Render sidebar
            self.render_sidebar()
            
            # Render main content
            self.render_chat_interface()
            
        except Exception as e:
            st.error(f"Application error: {str(e)}")
            logger.error(f"Application error: {e}")

def main():
    """Main function to run the application."""
    # Check for required environment variables
    if not os.getenv("GROQ_API_KEY"):
        st.error("""
        ⚠️ **Groq API Key Required**
        
        Please set your Groq API key in the environment variables:
        
        ```bash
        export GROQ_API_KEY="your-api-key-here"
        ```
        
        Or create a `.env` file in the project directory:
        ```
        GROQ_API_KEY=your-api-key-here
        ```
        
        You can get your Groq API key from: https://console.groq.com/
        """)
        return
    
    # Run the application
    app = NoteWiseApp()
    app.run()

if __name__ == "__main__":
    main()
