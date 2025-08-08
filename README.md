# 📚 NoteWise - AI Assistant for PDF Analysis

**NoteWise** is a production-grade RAG-based AI assistant that allows users to upload PDFs and ask questions via both text and voice (Hinglish-friendly). The assistant answers based on PDF context using Retrieval-Augmented Generation (RAG).

## 🎯 Features

- **📄 PDF Document Analysis**: Upload and process PDF documents with intelligent text extraction
- **🎤 Voice Input Support**: Ask questions using voice (supports Hinglish - Hindi-English mixed)
- **💬 Chat Interface**: Natural language conversation with your documents
- **🔍 Source Citations**: Get answers with references to specific pages and content
- **📊 Document Insights**: Comprehensive analysis of uploaded documents
- **🔄 Real-time Processing**: Instant responses with context-aware answers
- **🎨 Modern UI**: Clean, intuitive Streamlit interface

## 🛠️ Tech Stack

- **Python 3.8+**: Core programming language
- **Streamlit**: Web application framework
- **LangChain**: RAG pipeline and document processing
- **FAISS**: Vector storage and similarity search
- **Groq LLM**: Large language model for responses (Llama 3, Mixtral)
- **PyMuPDF**: PDF text extraction
- **Whisper**: Speech-to-text conversion
- **Sentence Transformers**: Text embeddings

## 📦 Installation

### Prerequisites

1. **Python 3.8 or higher**
2. **Groq API Key** (required for LLM)
3. **Git** (for cloning the repository)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Notewise
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```bash
   GROQ_API_KEY=your-groq-api-key-here
   ```
   
   Or set the environment variable directly:
   ```bash
   export GROQ_API_KEY="your-groq-api-key-here"
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

## 🚀 Usage

### Getting Started

1. **Launch the Application**
   - Run `streamlit run app.py`
   - Open your browser to `http://localhost:8501`

2. **Upload a PDF Document**
   - Use the sidebar to upload a PDF file (max 50MB)
   - Wait for processing to complete
   - View document information (pages, size, chunks)

3. **Ask Questions**
   - **Text Input**: Type your question in the chat interface
   - **Voice Input**: Click the microphone button and speak your question
   - **Hinglish Support**: Ask questions in Hindi-English mixed language

4. **View Responses**
   - Get AI-generated answers based on document content
   - See source citations with page references
   - Review chat history

### Example Questions

- "What are the main topics discussed in this document?"
- "Summarize the key findings from page 5"
- "What are the recommendations mentioned?"
- "Explain the methodology used in this research"
- "क्या इस document में कोई specific guidelines हैं?" (Hinglish)

## 🏗️ Architecture

### Code Structure

```
Notewise/
├── app.py                 # Main Streamlit application
├── rag_pipeline.py        # RAG logic (embeddings, retrieval, LLM)
├── voice_input.py         # Voice-to-text via Whisper
├── utils.py              # Helper functions (PDF loader, chunker)
├── requirements.txt      # Python dependencies
├── README.md            # Project documentation
└── .env                 # Environment variables (create this)
```

### Key Components

1. **PDFProcessor** (`utils.py`)
   - Handles PDF text extraction using PyMuPDF
   - Creates text chunks using LangChain
   - Manages document metadata

2. **RAGPipeline** (`rag_pipeline.py`)
   - Manages sentence-transformers embeddings
   - Creates and queries FAISS vector store
   - Handles Groq LLM interactions
   - Provides source citations

3. **VoiceInput** (`voice_input.py`)
   - Records audio using sounddevice
   - Transcribes speech using Whisper
   - Supports multiple languages including Hindi

4. **NoteWiseApp** (`app.py`)
   - Main Streamlit application
   - Handles UI interactions
   - Manages session state
   - Coordinates between components

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GROQ_API_KEY` | Groq API key for LLM | Yes |

### Model Settings

- **Default LLM**: llama3-8b-8192 (Groq)
- **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2
- **Whisper Model**: base (configurable)
- **Chunk Size**: 1000 characters (configurable)
- **Chunk Overlap**: 200 characters (configurable)

### Performance Tuning

- **Vector Store**: FAISS for fast similarity search
- **Retrieval**: Top-3 most similar chunks
- **Memory**: Conversation buffer for context
- **Caching**: Session state for document processing

## 🎨 UI Features

### Modern Interface
- **Clean Design**: Professional and intuitive layout
- **Responsive Layout**: Works on desktop and mobile
- **Dark/Light Mode**: Automatic theme detection
- **Custom CSS**: Enhanced styling and animations

### Interactive Elements
- **File Upload**: Drag-and-drop PDF upload
- **Voice Recording**: Real-time audio capture
- **Chat History**: Persistent conversation memory
- **Source Citations**: Clickable document references
- **Progress Indicators**: Visual feedback for processing

## 🔍 Advanced Features

### Document Processing
- **Multi-page Support**: Handles documents of any length
- **Text Extraction**: Intelligent content extraction
- **Chunking Strategy**: Optimal text segmentation
- **Metadata Preservation**: Page numbers and source tracking

### Voice Capabilities
- **Multi-language Support**: English, Hindi, and Hinglish
- **Noise Reduction**: Automatic audio processing
- **Duration Control**: Configurable recording length
- **Transcription Accuracy**: High-quality speech recognition

### RAG Pipeline
- **Context-Aware Answers**: Based on document content
- **Source Verification**: Accurate page references
- **Memory Management**: Conversation context preservation
- **Error Handling**: Robust error recovery

## 🚨 Troubleshooting

### Common Issues

1. **Groq API Key Error**
   ```
   Solution: Ensure GROQ_API_KEY is set in .env file or environment variables
   ```

2. **PDF Processing Failed**
   ```
   Solution: Check PDF file size (max 50MB) and ensure it's not corrupted
   ```

3. **Voice Recording Issues**
   ```
   Solution: Check microphone permissions and audio drivers
   ```

4. **Memory Issues**
   ```
   Solution: Reduce chunk size or use smaller documents
   ```

### Performance Optimization

- **Large Documents**: Increase chunk size for better processing
- **Memory Usage**: Clear chat history periodically
- **Response Time**: Use smaller Whisper model for faster transcription
- **Storage**: Clean up temporary files regularly

## 📊 Deployment

### Local Deployment
```bash
# Development
streamlit run app.py

# Production (with gunicorn)
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8501 app:main
```

### Cloud Deployment

#### Streamlit Cloud
1. Push code to GitHub
2. Connect repository to Streamlit Cloud
3. Set environment variables
4. Deploy automatically

#### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Groq** for LLM models and API
- **LangChain** for RAG pipeline framework
- **Streamlit** for web application framework
- **FAISS** for vector similarity search
- **Whisper** for speech-to-text conversion
- **Sentence Transformers** for text embeddings

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the documentation

---

**NoteWise** - Making PDF analysis intelligent and accessible! 📚✨
