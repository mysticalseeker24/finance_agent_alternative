"""Voice Agent for handling speech-to-text and text-to-speech operations."""

from typing import Dict, List, Any, Optional, Union, BinaryIO
import os
import io
import asyncio
from datetime import datetime
import tempfile

import whisper
import requests
import sounddevice as sd
import numpy as np
from pydub import AudioSegment
from loguru import logger

from agents.base_agent import BaseAgent
from config import Config


class VoiceAgent(BaseAgent):
    """Agent for handling speech-to-text and text-to-speech operations."""
    
    def __init__(self):
        """Initialize the Voice agent."""
        super().__init__("Voice Agent")
        self.stt_model = None
        self.audio_cache_dir = "audio_cache"
        os.makedirs(self.audio_cache_dir, exist_ok=True)
        logger.info("Voice Agent initialized")
    
    async def initialize_stt_model(self) -> bool:
        """Initialize the speech-to-text model.
        
        Returns:
            True if initialization is successful, False otherwise.
        """
        try:
            # Initialize Whisper model
            # Use 'base' model for faster processing, 'medium' or 'large' for better accuracy
            self.stt_model = await asyncio.to_thread(whisper.load_model, "base")
            logger.info("Initialized Whisper STT model")
            return True
        except Exception as e:
            logger.error(f"Error initializing STT model: {str(e)}")
            return False
    
    async def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a voice-related request.
        
        Args:
            request: The request containing operation and parameters.
                    Expected format:
                    {
                        "operation": "speech_to_text"|"text_to_speech"|"record_audio",
                        "parameters": {...}  # Operation-specific parameters
                    }
            
        Returns:
            The processing result.
        """
        operation = request.get("operation")
        parameters = request.get("parameters", {})
        
        if not operation:
            return {"error": "No operation specified"}
        
        # Execute the requested operation
        if operation == "speech_to_text":
            audio_path = parameters.get("audio_path")
            audio_bytes = parameters.get("audio_bytes")
            language = parameters.get("language", "en")
            
            if not audio_path and not audio_bytes:
                return {"error": "Either audio_path or audio_bytes must be provided for speech-to-text"}
            
            result = await self._speech_to_text(audio_path, audio_bytes, language)
            return {"data": result}
        
        elif operation == "text_to_speech":
            text = parameters.get("text")
            voice = parameters.get("voice", Config.VOICE_MODEL)
            output_path = parameters.get("output_path")
            
            if not text:
                return {"error": "No text provided for text-to-speech"}
            
            result = await self._text_to_speech(text, voice, output_path)
            return {"data": result}
        
        elif operation == "record_audio":
            duration = parameters.get("duration", 10)  # Default 10 seconds
            sample_rate = parameters.get("sample_rate", 16000)
            output_path = parameters.get("output_path")
            
            result = await self._record_audio(duration, sample_rate, output_path)
            return {"data": result}
        
        else:
            return {"error": f"Unknown operation: {operation}"}
    
    async def _speech_to_text(self, audio_path: Optional[str] = None, audio_bytes: Optional[Union[bytes, BinaryIO]] = None, language: str = "en") -> Dict[str, Any]:
        """Convert speech to text using Whisper.
        
        Args:
            audio_path: Path to the audio file.
            audio_bytes: Audio data as bytes or file-like object.
            language: Language code (default: 'en' for English).
            
        Returns:
            Transcription result.
        """
        try:
            # Initialize STT model if not already done
            if not self.stt_model:
                success = await self.initialize_stt_model()
                if not success:
                    return {"error": "Failed to initialize STT model"}
            
            # Load audio data
            temp_file = None
            try:
                if audio_path:
                    # Use the provided audio file path
                    file_path = audio_path
                else:
                    # Create a temporary file for the audio bytes
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                    if isinstance(audio_bytes, bytes):
                        temp_file.write(audio_bytes)
                    else:  # file-like object
                        audio_bytes.seek(0)
                        temp_file.write(audio_bytes.read())
                    temp_file.close()
                    file_path = temp_file.name
                
                # Transcribe audio using Whisper
                result = await asyncio.to_thread(
                    self.stt_model.transcribe,
                    file_path,
                    language=language
                )
                
                # Extract transcription and other details
                transcription = {
                    "text": result["text"],
                    "language": result.get("language", language),
                    "segments": [
                        {
                            "id": i,
                            "start": segment["start"],
                            "end": segment["end"],
                            "text": segment["text"],
                        }
                        for i, segment in enumerate(result.get("segments", []))
                    ],
                    "confidence": result.get("confidence", 1.0),
                }
                
                logger.info(f"Transcribed audio to text (length: {len(transcription['text'])} chars)")
                return transcription
            
            finally:
                # Clean up temporary file if created
                if temp_file and os.path.exists(temp_file.name):
                    os.unlink(temp_file.name)
        
        except Exception as e:
            logger.error(f"Error in speech to text: {str(e)}")
            return {"error": str(e)}
    
    async def _text_to_speech(self, text: str, voice: str = None, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Convert text to speech using TTS service.
        
        Args:
            text: The text to convert to speech.
            voice: Voice model to use (default: from Config).
            output_path: Path to save the audio file (optional).
            
        Returns:
            Text-to-speech result with audio data or path.
        """
        try:
            # Use Sarvam AI TTS API (simulated)
            # In a real implementation, you would use the actual API
            voice = voice or Config.VOICE_MODEL
            
            # Generate a filename if output_path is not provided
            if not output_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(self.audio_cache_dir, f"tts_{timestamp}.mp3")
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Simulate TTS API call and save to file
            # In a real implementation, you would use the actual API
            logger.info(f"Simulating TTS for text (length: {len(text)} chars) using voice: {voice}")
            
            # Create a mock MP3 file (1 second of silence)
            # In a real implementation, this would be the actual audio data
            silence = AudioSegment.silent(duration=1000)  # 1 second of silence
            silence.export(output_path, format="mp3")
            
            # Prepare result with local file path and URL (if serving files)
            result = {
                "text": text,
                "voice": voice,
                "audio_path": output_path,
                "audio_url": f"file://{os.path.abspath(output_path)}",  # Local file URL
                "format": "mp3",
                "duration": 1.0,  # Mock duration in seconds
            }
            
            logger.info(f"Generated speech audio saved to: {output_path}")
            return result
        
        except Exception as e:
            logger.error(f"Error in text to speech: {str(e)}")
            return {"error": str(e)}
    
    async def _record_audio(self, duration: int = 10, sample_rate: int = 16000, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Record audio from the microphone.
        
        Args:
            duration: Recording duration in seconds.
            sample_rate: Audio sample rate.
            output_path: Path to save the audio file (optional).
            
        Returns:
            Recording result with audio data or path.
        """
        try:
            # Generate a filename if output_path is not provided
            if not output_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(self.audio_cache_dir, f"recording_{timestamp}.wav")
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            logger.info(f"Recording audio for {duration} seconds at {sample_rate} Hz")
            
            # Record audio using sounddevice
            # This would typically be run in the main thread, not in asyncio
            # For this example, we'll simulate it
            
            # In a real implementation, you would use something like:
            # recording = await asyncio.to_thread(
            #     sd.rec,
            #     int(duration * sample_rate),
            #     samplerate=sample_rate,
            #     channels=1,
            #     dtype='float32'
            # )
            # await asyncio.to_thread(sd.wait)  # Wait for recording to complete
            
            # Simulate recording by creating a silent audio file
            # In a real implementation, this would be the actual recorded audio
            silence = AudioSegment.silent(duration=duration * 1000)  # Convert seconds to milliseconds
            silence.export(output_path, format="wav")
            
            result = {
                "audio_path": output_path,
                "duration": duration,
                "sample_rate": sample_rate,
                "channels": 1,
                "format": "wav",
            }
            
            logger.info(f"Recorded audio saved to: {output_path}")
            return result
        
        except Exception as e:
            logger.error(f"Error recording audio: {str(e)}")
            return {"error": str(e)}
    
    async def speech_to_text(self, audio_path: Optional[str] = None, audio_bytes: Optional[Union[bytes, BinaryIO]] = None, language: str = "en") -> Dict[str, Any]:
        """Convert speech to text.
        
        Args:
            audio_path: Path to the audio file.
            audio_bytes: Audio data as bytes or file-like object.
            language: Language code (default: 'en' for English).
            
        Returns:
            Transcription result.
        """
        request = {
            "operation": "speech_to_text",
            "parameters": {
                "audio_path": audio_path,
                "audio_bytes": audio_bytes,
                "language": language
            }
        }
        return await self.run(request)
    
    async def text_to_speech(self, text: str, voice: str = None, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Convert text to speech.
        
        Args:
            text: The text to convert to speech.
            voice: Voice model to use.
            output_path: Path to save the audio file (optional).
            
        Returns:
            Text-to-speech result with audio data or path.
        """
        request = {
            "operation": "text_to_speech",
            "parameters": {
                "text": text,
                "voice": voice,
                "output_path": output_path
            }
        }
        return await self.run(request)
    
    async def record_audio(self, duration: int = 10, sample_rate: int = 16000, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Record audio from the microphone.
        
        Args:
            duration: Recording duration in seconds.
            sample_rate: Audio sample rate.
            output_path: Path to save the audio file (optional).
            
        Returns:
            Recording result with audio data or path.
        """
        request = {
            "operation": "record_audio",
            "parameters": {
                "duration": duration,
                "sample_rate": sample_rate,
                "output_path": output_path
            }
        }
        return await self.run(request)
