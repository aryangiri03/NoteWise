"""
Voice input module for NoteWise application.
Handles speech-to-text conversion using Whisper.
"""

import os
import tempfile
import logging
from typing import Optional, Tuple
import whisper
import sounddevice as sd
import numpy as np
import wave
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VoiceInput:
    """Handles voice input and speech-to-text conversion."""
    
    def __init__(self, model_name: str = "base"):
        """
        Initialize VoiceInput with Whisper model.
        
        Args:
            model_name: Whisper model name (tiny, base, small, medium, large)
        """
        self.model_name = model_name
        self.model = None
        self.sample_rate = 16000
        self.channels = 1
        
        # Initialize Whisper model
        self._load_whisper_model()
    
    def _load_whisper_model(self):
        """Load Whisper model."""
        try:
            logger.info(f"Loading Whisper model: {self.model_name}")
            self.model = whisper.load_model(self.model_name)
            logger.info("Whisper model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise Exception(f"Whisper model loading failed: {str(e)}")
    
    def record_audio(self, duration: int = 10, sample_rate: int = 16000) -> np.ndarray:
        """
        Record audio from microphone.
        
        Args:
            duration: Recording duration in seconds
            sample_rate: Audio sample rate
            
        Returns:
            Recorded audio as numpy array
        """
        try:
            logger.info(f"Recording audio for {duration} seconds...")
            
            # Record audio
            audio_data = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype=np.float32
            )
            
            # Wait for recording to complete
            sd.wait()
            
            logger.info("Audio recording completed")
            return audio_data
            
        except Exception as e:
            logger.error(f"Failed to record audio: {e}")
            raise Exception(f"Audio recording failed: {str(e)}")
    
    def save_audio_to_wav(self, audio_data: np.ndarray, file_path: str, sample_rate: int = 16000):
        """
        Save audio data to WAV file.
        
        Args:
            audio_data: Audio data as numpy array
            file_path: Path to save the WAV file
            sample_rate: Audio sample rate
        """
        try:
            # Convert to 16-bit PCM
            audio_data_16bit = (audio_data * 32767).astype(np.int16)
            
            # Save as WAV file
            with wave.open(file_path, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data_16bit.tobytes())
            
            logger.info(f"Audio saved to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save audio: {e}")
            raise Exception(f"Audio save failed: {str(e)}")
    
    def transcribe_audio(self, audio_file_path: str) -> str:
        """
        Transcribe audio file to text using Whisper.
        
        Args:
            audio_file_path: Path to the audio file
            
        Returns:
            Transcribed text
        """
        try:
            if not self.model:
                self._load_whisper_model()
            
            logger.info(f"Transcribing audio file: {audio_file_path}")
            
            # Transcribe audio
            result = self.model.transcribe(audio_file_path)
            transcribed_text = result["text"].strip()
            
            logger.info(f"Transcription completed: {len(transcribed_text)} characters")
            return transcribed_text
            
        except Exception as e:
            logger.error(f"Failed to transcribe audio: {e}")
            raise Exception(f"Audio transcription failed: {str(e)}")
    
    def record_and_transcribe(self, duration: int = 10) -> Tuple[str, str]:
        """
        Record audio and transcribe it to text.
        
        Args:
            duration: Recording duration in seconds
            
        Returns:
            Tuple of (transcribed_text, audio_file_path)
        """
        try:
            # Record audio
            audio_data = self.record_audio(duration=duration)
            
            # Create temporary file for audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                audio_file_path = temp_file.name
            
            # Save audio to file
            self.save_audio_to_wav(audio_data, audio_file_path)
            
            # Transcribe audio
            transcribed_text = self.transcribe_audio(audio_file_path)
            
            return transcribed_text, audio_file_path
            
        except Exception as e:
            logger.error(f"Failed to record and transcribe: {e}")
            raise Exception(f"Record and transcribe failed: {str(e)}")
    
    def cleanup_audio_file(self, audio_file_path: str):
        """
        Clean up temporary audio file.
        
        Args:
            audio_file_path: Path to the audio file to delete
        """
        try:
            if os.path.exists(audio_file_path):
                os.remove(audio_file_path)
                logger.info(f"Cleaned up audio file: {audio_file_path}")
                
        except Exception as e:
            logger.warning(f"Failed to cleanup audio file {audio_file_path}: {e}")
    
    def get_supported_languages(self) -> list:
        """
        Get list of supported languages by Whisper.
        
        Returns:
            List of supported language codes
        """
        # Whisper supports multiple languages including Hindi and English
        return [
            "en",  # English
            "hi",  # Hindi
            "auto"  # Auto-detect
        ]
    
    def transcribe_with_language(self, audio_file_path: str, language: str = "auto") -> str:
        """
        Transcribe audio with specified language.
        
        Args:
            audio_file_path: Path to the audio file
            language: Language code (en, hi, auto)
            
        Returns:
            Transcribed text
        """
        try:
            if not self.model:
                self._load_whisper_model()
            
            logger.info(f"Transcribing audio with language: {language}")
            
            # Transcribe with language specification
            if language == "auto":
                result = self.model.transcribe(audio_file_path)
            else:
                result = self.model.transcribe(audio_file_path, language=language)
            
            transcribed_text = result["text"].strip()
            
            logger.info(f"Transcription completed with language {language}: {len(transcribed_text)} characters")
            return transcribed_text
            
        except Exception as e:
            logger.error(f"Failed to transcribe audio with language {language}: {e}")
            raise Exception(f"Audio transcription failed: {str(e)}")
    
    def get_audio_duration(self, audio_file_path: str) -> float:
        """
        Get duration of audio file.
        
        Args:
            audio_file_path: Path to the audio file
            
        Returns:
            Duration in seconds
        """
        try:
            with wave.open(audio_file_path, 'rb') as wav_file:
                frames = wav_file.getnframes()
                sample_rate = wav_file.getframerate()
                duration = frames / sample_rate
                return duration
                
        except Exception as e:
            logger.error(f"Failed to get audio duration: {e}")
            return 0.0
