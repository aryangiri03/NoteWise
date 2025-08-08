# 🚀 NoteWise Quick Start Guide

Get NoteWise up and running in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- Groq API key
- Git (optional)

## ⚡ Quick Setup

### 1. Clone or Download
```bash
# If using Git
git clone <repository-url>
cd Notewise

# Or download and extract the ZIP file
```

### 2. Install Dependencies
```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### 3. Set Up API Key
Create a `.env` file in the project directory:
```bash
# .env
GROQ_API_KEY=your-groq-api-key-here
```

**Get your API key from:** https://console.groq.com/

### 4. Run the Application
```bash
# Option 1: Use the startup script (recommended)
python run.py

# Option 2: Run directly with Streamlit
streamlit run app.py
```

### 5. Open in Browser
Navigate to: `http://localhost:8501`

## 🎯 First Steps

1. **Upload a PDF** - Use the sidebar to upload a PDF document
2. **Wait for Processing** - The app will extract text and create embeddings
3. **Ask Questions** - Type or speak your questions about the document
4. **View Sources** - See page references and source content

## 🎤 Voice Input

- Click the microphone button
- Speak your question (supports Hinglish)
- Wait for transcription
- Review and send

## 🔧 Troubleshooting

### Common Issues

**"Groq API Key not found"**
- Ensure `.env` file exists in project root
- Check API key is correct and active

**"Package not found"**
- Run: `pip install -r requirements.txt`
- Check Python version (3.8+)

**"PDF processing failed"**
- Check file size (max 50MB)
- Ensure PDF is not corrupted
- Try a different PDF file

### Test Installation
```bash
python test_installation.py
```

## 📚 Example Usage

### Sample Questions
- "What are the main topics in this document?"
- "Summarize the key findings"
- "What are the recommendations?"
- "क्या इस document में कोई specific guidelines हैं?"

### Supported Features
- ✅ PDF document analysis
- ✅ Voice input (Hinglish-friendly)
- ✅ Chat interface
- ✅ Source citations
- ✅ Document insights
- ✅ Real-time processing

## 🆘 Need Help?

- Check the full [README.md](README.md)
- Run the test script: `python test_installation.py`
- Review troubleshooting section
- Create an issue on GitHub

---

**Happy analyzing with NoteWise! 📚✨**
