#!/usr/bin/env python3
"""
NoteWise Startup Script
This script checks for required environment variables and starts the NoteWise application.
"""

import os
import sys
import subprocess
from dotenv import load_dotenv

def check_environment():
    """Check if required environment variables are set."""
    load_dotenv()
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ Error: GROQ_API_KEY not found!")
        print("\nPlease set your Groq API key:")
        print("1. Create a .env file in the project directory")
        print("2. Add: GROQ_API_KEY=your-api-key-here")
        print("3. Or set the environment variable: export GROQ_API_KEY=your-api-key-here")
        print("\nGet your API key from: https://console.groq.com/")
        return False
    
    print("✅ Environment variables configured")
    return True

def check_dependencies():
    """Check if required packages are installed."""
    required_packages = [
        'streamlit',
        'langchain',
        'langchain_groq',
        'langchain_community',
        'faiss',
        'groq',
        'fitz',
        'dotenv',
        'whisper',
        'sounddevice',
        'numpy',
        'pandas'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("\nPlease install missing packages:")
        print("pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed")
    return True

def main():
    """Main function to start the application."""
    print("🚀 Starting NoteWise...")
    print("=" * 50)
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    print("\n🎉 All checks passed! Starting NoteWise...")
    print("=" * 50)
    
    # Start the Streamlit application
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 NoteWise stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Failed to start NoteWise: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
