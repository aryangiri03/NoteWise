#!/usr/bin/env python3
"""
Test script to verify NoteWise installation and dependencies.
Run this script to check if all required packages are installed correctly.
"""

import sys
import importlib
import os

def test_imports():
    """Test if all required packages can be imported."""
    required_packages = [
        'streamlit',
        'langchain',
        'langchain_groq',
        'langchain_community',
        'faiss',
        'groq',
        'fitz',  # PyMuPDF
        'dotenv',
        'whisper',
        'sounddevice',
        'numpy',
        'pandas',
        'tiktoken',
        'sentence_transformers'
    ]
    
    print("🔍 Testing package imports...")
    failed_imports = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
            failed_imports.append(package)
    
    return failed_imports

def test_environment():
    """Test environment variables."""
    print("\n🔍 Testing environment variables...")
    
    # Check Groq API key
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        print(f"✅ GROQ_API_KEY: {'*' * (len(api_key) - 4) + api_key[-4:]}")
    else:
        print("❌ GROQ_API_KEY: Not set")
        print("   Please set your Groq API key in the .env file or environment variables")
    
    return api_key is not None

def test_custom_modules():
    """Test if custom modules can be imported."""
    print("\n🔍 Testing custom modules...")
    
    custom_modules = [
        'utils',
        'rag_pipeline',
        'voice_input'
    ]
    
    failed_modules = []
    
    for module in custom_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed_modules.append(module)
    
    return failed_modules

def test_pdf_processing():
    """Test PDF processing capabilities."""
    print("\n🔍 Testing PDF processing...")
    
    try:
        from utils import PDFProcessor
        processor = PDFProcessor()
        print("✅ PDFProcessor initialized successfully")
        return True
    except Exception as e:
        print(f"❌ PDFProcessor initialization failed: {e}")
        return False

def test_voice_processing():
    """Test voice processing capabilities."""
    print("\n🔍 Testing voice processing...")
    
    try:
        from voice_input import VoiceInput
        voice_input = VoiceInput()
        print("✅ VoiceInput initialized successfully")
        return True
    except Exception as e:
        print(f"❌ VoiceInput initialization failed: {e}")
        return False

def test_rag_pipeline():
    """Test RAG pipeline initialization."""
    print("\n🔍 Testing RAG pipeline...")
    
    try:
        from rag_pipeline import RAGPipeline
        # Don't actually initialize if no API key
        if os.getenv("GROQ_API_KEY"):
            rag = RAGPipeline()
            print("✅ RAGPipeline initialized successfully")
        else:
            print("⚠️  RAGPipeline: Skipped (no API key)")
        return True
    except Exception as e:
        print(f"❌ RAGPipeline initialization failed: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 NoteWise Installation Test")
    print("=" * 50)
    
    # Test Python version
    print(f"🐍 Python version: {sys.version}")
    
    # Run tests
    failed_imports = test_imports()
    env_ok = test_environment()
    failed_modules = test_custom_modules()
    pdf_ok = test_pdf_processing()
    voice_ok = test_voice_processing()
    rag_ok = test_rag_pipeline()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary")
    print("=" * 50)
    
    if not failed_imports and not failed_modules and pdf_ok and voice_ok and rag_ok:
        print("🎉 All tests passed! NoteWise is ready to use.")
        if not env_ok:
            print("⚠️  Warning: GROQ_API_KEY not set. Please set it to use the application.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        if failed_imports:
            print(f"   - Failed package imports: {', '.join(failed_imports)}")
        if failed_modules:
            print(f"   - Failed module imports: {', '.join(failed_modules)}")
        if not pdf_ok:
            print("   - PDF processing failed")
        if not voice_ok:
            print("   - Voice processing failed")
        if not rag_ok:
            print("   - RAG pipeline failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
