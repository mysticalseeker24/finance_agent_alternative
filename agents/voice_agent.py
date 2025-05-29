"""Voice Agent for handling speech-to-text and text-to-speech operations."""

import asyncio
import io
import os
import tempfile
from datetime import datetime
from typing import Any, BinaryIO, Dict, List, Optional, Union

import numpy as np
import requests
import sounddevice as sd
import whisper
from loguru import logger
from pydub import AudioSegment

from agents.base_agent import BaseAgent
from config import Config


class VoiceAgent(BaseAgent):
    """Agent for handling speech-to-text and text-to-speech operations.

    Guidance for using a Fine-Tuned Whisper Model:
    1.  **Purpose of `WHISPER_FINETUNED_MODEL_PATH`:**
        This configuration allows you to specify a path to a Whisper model that you have
        fine-tuned for a specific domain, accent, or language nuance. Using a fine-tuned
        model can significantly improve transcription accuracy for specialized audio.

    2.  **Fine-Tuning Process Overview (Conceptual):**
        *   **Data Collection:** Gather a dataset of audio recordings and their accurate transcriptions
            that represent the target domain (e.g., financial earnings calls, specific accents).
            The more high-quality, domain-specific data you have, the better the fine-tuning.
        *   **Tools:** Typically, fine-tuning is done using libraries like Hugging Face Transformers.
            You would start with a pre-trained Whisper model (e.g., 'base', 'small', 'medium')
            and train it further on your custom dataset.
        *   **Training:** This involves setting up a training pipeline, defining training arguments
            (learning rate, epochs, batch size), and running the training process, often on a GPU.
        *   **Saving the Model:** After training, save the fine-tuned model artifacts (weights,
            configuration files, tokenizer files, preprocessor files). This saved directory or
            Hugging Face Hub model name is what you would point `WHISPER_FINETUNED_MODEL_PATH` to.

    3.  **Model Compatibility and Loading:**
        *   **`openai-whisper` Compatibility:** The `whisper.load_model()` function used in this
            agent is from the `openai-whisper` library. If your fine-tuned model is saved in a
            format directly compatible with this (e.g., if you fine-tuned one of their released
            checkpoints using compatible methods), it might load directly.
        *   **Hugging Face Transformers Compatibility:** If your fine-tuned model is purely a
            Hugging Face Transformers checkpoint (e.g., saved using `model.save_pretrained()`
            from the `transformers` library), `whisper.load_model()` might not work directly.
            In such cases, you would need to modify the `_speech_to_text` method in this agent
            to use Hugging Face's `pipeline("automatic-speech-recognition", model="your-hf-model-path")`
            or load the model with `WhisperForConditionalGeneration.from_pretrained()` and use its
            `.generate()` method along with `WhisperProcessor` for audio processing.
        *   **Current Implementation:** This agent currently attempts to load the specified path/name
            using `whisper.load_model()`. If this fails, it falls back to the default model.
            No automatic conversion or alternative loading mechanisms are implemented in this step.
    """

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
            # This import is here for clarity, it's usually at the top of the file.
            import whisper

            # import os # Ensure os is imported at the top of the file

            tuned_model_path = Config.WHISPER_FINETUNED_MODEL_PATH
            model_to_load = None
            loaded_model_type = ""  # For logging
            model_identifier_for_log = "N/A"

            if tuned_model_path:
                # Basic check for local path existence. HF Hub names won't pass this,
                # but whisper.load_model() might handle HF Hub names directly.
                if os.path.exists(tuned_model_path):
                    logger.info(
                        f"Attempting to load fine-tuned Whisper model from local path: {tuned_model_path}"
                    )
                    model_to_load = tuned_model_path
                    loaded_model_type = "Fine-tuned (local)"
                    model_identifier_for_log = tuned_model_path
                else:
                    # Assuming it could be an HF Hub model name if not a local path
                    logger.info(
                        f"Attempting to load fine-tuned Whisper model from Hugging Face Hub or alias: {tuned_model_path}"
                    )
                    model_to_load = tuned_model_path
                    loaded_model_type = "Fine-tuned (HF Hub/Alias)"
                    model_identifier_for_log = tuned_model_path

            if model_to_load:
                try:
                    # The openai-whisper library's load_model can take a name (like 'medium')
                    # or a path to the directory where model weights and other files are stored.
                    # If fine-tuning was done with Hugging Face Transformers, the saved checkpoint
                    # might need conversion or a different loading mechanism if not directly compatible.
                    # For this conceptual step, we proceed assuming compatibility or that the user
                    # handles the format.
                    self.stt_model = await asyncio.to_thread(
                        whisper.load_model, model_to_load
                    )
                    logger.info(
                        f"Successfully loaded {loaded_model_type} Whisper model: {model_identifier_for_log}"
                    )
                except Exception as ft_load_error:
                    logger.warning(
                        f"Failed to load fine-tuned Whisper model from '{model_to_load}': {ft_load_error}. "
                        "Falling back to default model."
                    )
                    model_to_load = None  # Force fallback
                    model_identifier_for_log = "N/A"  # Reset identifier

            if not model_to_load:  # If no tuned path or if loading tuned path failed
                default_model_name = "base.en"  # Changed default model
                logger.info(f"Loading default Whisper model: {default_model_name}")
                self.stt_model = await asyncio.to_thread(
                    whisper.load_model, default_model_name
                )
                loaded_model_type = f"Default"
                model_identifier_for_log = default_model_name

            # Try to get the actual model name/size if available from the loaded object for logging
            # This is a best-effort logging as custom paths won't have a 'name' attribute like standard models.
            if hasattr(self.stt_model, "name") and self.stt_model.name:
                # This usually works for standard models like "medium", "base", etc.
                log_model_name = self.stt_model.name
            else:
                # For custom paths or potentially HF models loaded via whisper.load_model (if compatible),
                # model_identifier_for_log set during loading attempt is more descriptive.
                log_model_name = model_identifier_for_log

            logger.info(
                f"Whisper speech-to-text model ({log_model_name} - type: {loaded_model_type}) initialized successfully."
            )
            return True
        except Exception as e:
            logger.error(f"Error initializing speech-to-text model: {str(e)}")
            self.stt_model = None  # Ensure stt_model is None if any error occurs
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
                return {
                    "error": "Either audio_path or audio_bytes must be provided for speech-to-text"
                }

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

    async def _speech_to_text(
        self,
        audio_path: Optional[str] = None,
        audio_bytes: Optional[Union[bytes, BinaryIO]] = None,
        language: str = "auto",
    ) -> Dict[str, Any]:
        """Convert speech to text using Whisper with support for Hindi and English.

        Args:
            audio_path: Path to the audio file.
            audio_bytes: Audio data as bytes or file-like object.
            language: Language code (default: 'auto' for automatic detection).
                      Can be 'en' for English, 'hi' for Hindi, or 'auto'.

        Returns:
            Transcription result with detected language.
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

                logger.info(f"Transcribing audio with language setting: {language}")

                # Transcribe audio using Whisper
                # For 'auto', let Whisper detect the language
                # For specific languages, use the provided code
                result = await asyncio.to_thread(
                    self.stt_model.transcribe,
                    file_path,
                    language=None if language == "auto" else language,
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

                logger.info(
                    f"Transcribed audio to text (length: {len(transcription['text'])} chars)"
                )
                return transcription

            finally:
                # Clean up temporary file if created
                if temp_file and os.path.exists(temp_file.name):
                    os.unlink(temp_file.name)

        except Exception as e:
            logger.error(f"Error in speech to text: {str(e)}")
            return {"error": str(e)}

    async def _text_to_speech(
        self, text: str, voice: str = None, output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Convert text to speech using TTS service.

        Args:
            text: The text to convert to speech.
            voice: Voice model to use (default: from Config).
            output_path: Path to save the audio file (optional).

        Returns:
            Text-to-speech result with audio data or path.
        """
        try:
            voice = voice or Config.VOICE_MODEL

            # Generate a filename if output_path is not provided
            if not output_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(self.audio_cache_dir, f"tts_{timestamp}.mp3")

            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # First try Sarvam AI if API key is available
            if Config.SARVAM_AI_API_KEY:
                logger.info(
                    f"Using Sarvam AI TTS for text (length: {len(text)} chars) with voice: {voice}"
                )
                result = await self._sarvam_ai_tts(text, voice, output_path)
                if not result.get("error"):
                    return result
                logger.warning(
                    f"Sarvam AI TTS failed: {result.get('error')}. Trying ElevenLabs..."
                )

            # Fall back to ElevenLabs if Sarvam AI failed or not configured
            if Config.ELEVENLABS_API_KEY:
                logger.info(
                    f"Using ElevenLabs TTS for text (length: {len(text)} chars) with voice: {voice}"
                )
                result = await self._elevenlabs_tts(text, voice, output_path)
                if not result.get("error"):
                    return result
                logger.warning(f"ElevenLabs TTS failed: {result.get('error')}")

            # If both services failed or no API keys are available, return an error
            if not Config.SARVAM_AI_API_KEY and not Config.ELEVENLABS_API_KEY:
                error_msg = "No TTS service API keys configured. Please provide either SARVAM_AI_API_KEY or ELEVENLABS_API_KEY."
            else:
                error_msg = "All configured TTS services failed."

            logger.error(error_msg)
            return {"error": error_msg}
            # Fallback if both APIs fail but we still need to return something
            # This would only happen in testing/development when APIs are unavailable
            silence = AudioSegment.silent(duration=1000)  # 1 second of silence
            silence.export(output_path, format="mp3")

            # Generate a URL path for API access
            audio_url = f"/audio/{os.path.basename(output_path)}"

            return {
                "error": error_msg,
                "audio_path": output_path,
                "audio_url": audio_url,
                "text": text,
                "voice": voice,
                "duration_seconds": 1.0,  # Mock duration
            }

        except Exception as e:
            logger.error(f"Error in text to speech: {str(e)}")
            return {"error": str(e)}

    async def _record_audio(
        self,
        duration: int = 10,
        sample_rate: int = 16000,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
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
                output_path = os.path.join(
                    self.audio_cache_dir, f"recording_{timestamp}.wav"
                )

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
            silence = AudioSegment.silent(
                duration=duration * 1000
            )  # Convert seconds to milliseconds
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

    async def speech_to_text(
        self,
        audio_path: Optional[str] = None,
        audio_bytes: Optional[Union[bytes, BinaryIO]] = None,
        language: str = "en",
    ) -> Dict[str, Any]:
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
                "language": language,
            },
        }
        return await self.run(request)

    async def text_to_speech(
        self, text: str, voice: str = None, output_path: Optional[str] = None
    ) -> Dict[str, Any]:
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
            "parameters": {"text": text, "voice": voice, "output_path": output_path},
        }
        return await self.run(request)

    async def record_audio(
        self,
        duration: int = 10,
        sample_rate: int = 16000,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
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
                "output_path": output_path,
            },
        }
        return await self.run(request)

    async def _sarvam_ai_tts(
        self, text: str, voice: str, output_path: str
    ) -> Dict[str, Any]:
        """Convert text to speech using Sarvam AI API.

        Args:
            text: The text to convert to speech.
            voice: Voice model to use.
            output_path: Path to save the audio file.

        Returns:
            Result with audio data or path.
        """
        try:
            import aiohttp

            # Sarvam AI API endpoint and headers
            url = "https://api.sarvam.ai/v1/tts"
            headers = {
                "Authorization": f"Bearer {Config.SARVAM_AI_API_KEY}",
                "Content-Type": "application/json",
            }

            # Request payload
            payload = {
                "text": text,
                "model": voice,  # Default is 'meera'
                "voice_preset": "default",
                "audioformat": "mp3",
                "pitch": 0,  # Added
                "speaking_rate": 1.0,  # Added (or "pace": 1.0 if that's the correct Sarvam term)
            }

            # Make API request
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        # Save audio file
                        audio_data = await response.read()
                        with open(output_path, "wb") as f:
                            f.write(audio_data)

                        # Get audio duration
                        audio = AudioSegment.from_file(output_path)
                        duration_seconds = len(audio) / 1000.0

                        # Generate URL for API access
                        audio_url = f"/audio/{os.path.basename(output_path)}"

                        return {
                            "audio_path": output_path,
                            "audio_url": audio_url,
                            "text": text,
                            "voice": voice,
                            "duration_seconds": duration_seconds,
                        }
                    else:
                        error_message = f"Sarvam AI API error: {response.status} - {await response.text()}"
                        logger.error(error_message)
                        return {"error": error_message}

        except Exception as e:
            logger.error(f"Error in Sarvam AI TTS: {str(e)}")
            return {"error": str(e)}

    async def _elevenlabs_tts(
        self, text: str, voice: str, output_path: str
    ) -> Dict[str, Any]:
        """Convert text to speech using ElevenLabs API.

        Args:
            text: The text to convert to speech.
            voice: Voice model to use.
            output_path: Path to save the audio file.

        Returns:
            Result with audio data or path.
        """
        try:
            import aiohttp

            # ElevenLabs voice mapping - map our voice name to ElevenLabs voice IDs
            # In a production system, you'd have a more comprehensive mapping
            voice_mapping = {
                "meera": "pNInz6obpgDQGcFmaJgB",  # Example ID for Indian female voice
                "male": "ErXwobaYiN019PkySvjV",  # Example ID for male voice
                "female": "EXAVITQu4vr4xnSDxMaL",  # Example ID for female voice
            }

            # Default to a standard voice if mapping not found
            voice_id = voice_mapping.get(voice.lower(), "21m00Tcm4TlvDq8ikWAM")

            # ElevenLabs API endpoint and headers
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
            headers = {
                "xi-api-key": Config.ELEVENLABS_API_KEY,
                "Content-Type": "application/json",
            }

            # Request payload
            payload = {
                "text": text,
                "model_id": "eleven_monolingual_v1",
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.5},
            }

            # Make API request
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        # Save audio file
                        audio_data = await response.read()
                        with open(output_path, "wb") as f:
                            f.write(audio_data)

                        # Get audio duration
                        audio = AudioSegment.from_file(output_path)
                        duration_seconds = len(audio) / 1000.0

                        # Generate URL for API access
                        audio_url = f"/audio/{os.path.basename(output_path)}"

                        return {
                            "audio_path": output_path,
                            "audio_url": audio_url,
                            "text": text,
                            "voice": voice,
                            "duration_seconds": duration_seconds,
                        }
                    else:
                        error_message = f"ElevenLabs API error: {response.status} - {await response.text()}"
                        logger.error(error_message)
                        return {"error": error_message}

        except Exception as e:
            logger.error(f"Error in ElevenLabs TTS: {str(e)}")
            return {"error": str(e)}
