"""Main FastAPI application for orchestrating the Finance Agent."""

import time
import asyncio
from typing import Dict, List, Any, Optional

from fastapi import (
    FastAPI,
    HTTPException,
    Depends,
    BackgroundTasks,
    File,
    UploadFile,
    Form,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from loguru import logger

from config import Config
from orchestrator.orchestrator import Orchestrator

# Create FastAPI app
app = FastAPI(
    title="Finance Agent API",
    description="API for the Finance Agent application",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create orchestrator instance
orchestrator = Orchestrator()

# Mount static files directory for serving audio files
app.mount("/audio", StaticFiles(directory="audio_cache"), name="audio")


# Define API models
class QueryRequest(BaseModel):
    """Model for a financial query request."""

    query: str
    voice_input: bool = False
    voice_output: bool = True
    language: str = "auto"  # 'auto', 'en' for English, 'hi' for Hindi


class QueryResponse(BaseModel):
    """Model for a financial query response."""

    query: str
    response: str
    audio_url: Optional[str] = None
    sources: List[Dict[str, Any]] = []
    processing_time: float
    speech_recognition: Optional[Dict[str, Any]] = None


class MarketBriefResponse(BaseModel):
    """Model for a market brief response."""

    title: str
    date: str
    summary: str
    full_text: str
    audio_url: Optional[str] = None
    sections: Dict[str, str]
    processing_time: float


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the orchestrator and agents on startup."""
    logger.info("Starting Finance Agent API...")

    # Check for valid configuration
    if not Config.validate():
        logger.warning("Invalid configuration! Prompting for missing API keys...")
        Config.check_and_prompt_for_missing_keys()

        # Verify configuration again after prompting
        if not Config.validate():
            logger.error(
                "Still missing required API keys. Some functionality will be limited."
            )

    # Initialize the orchestrator
    await orchestrator.initialize()

    logger.info("Startup completed!")


# API endpoints
@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Welcome to the Finance Agent API", "version": "0.1.0"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "config_valid": Config.validate()}


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a financial query.

    Args:
        request: The query request.

    Returns:
        The query response.
    """
    logger.info(
        f"Received query request: {request.query} (language: {request.language})"
    )
    try:
        result = await orchestrator.process_query(
            query=request.query,
            voice_input=request.voice_input,
            voice_output=request.voice_output,
            language=request.language,
        )
        return result

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/voice-query")
async def process_voice_query(
    audio: UploadFile = File(...),
    voice_output: bool = Form(True),
    language: str = Form("auto"),  # 'auto', 'en' for English, 'hi' for Hindi
):
    """Process a voice query.

    Args:
        audio: The audio file containing the query.
        voice_output: Whether to generate voice output.
        language: Language code for STT (default: 'auto' for automatic detection).

    Returns:
        The query response.
    """
    logger.info(f"Received voice query: {audio.filename}")
    try:
        # Save the uploaded audio file
        audio_content = await audio.read()

        # Process the audio query with language parameter
        logger.info(f"Processing voice query with language setting: {language}")
        result = await orchestrator.process_query(
            query=audio_content,  # Pass the audio content directly
            voice_input=True,
            voice_output=voice_output,
            language=language,  # Pass the language parameter
        )
        return result

    except Exception as e:
        logger.error(f"Error processing voice query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/market-brief", response_model=MarketBriefResponse)
async def generate_market_brief():
    """Generate a market brief.

    Returns:
        The generated market brief.
    """
    logger.info("Received market brief generation request")
    try:
        result = await orchestrator.generate_market_brief()
        return result

    except Exception as e:
        logger.error(f"Error generating market brief: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/schedule-market-brief")
async def schedule_market_brief(background_tasks: BackgroundTasks):
    """Schedule a market brief to be generated.

    This endpoint triggers the generation of a market brief and returns immediately.
    The actual generation happens in the background.

    Args:
        background_tasks: FastAPI background tasks.

    Returns:
        A message indicating that the market brief generation has started.
    """
    logger.info("Received scheduled market brief generation request")

    async def generate_brief_task():
        try:
            await orchestrator.generate_market_brief()
            logger.info("Scheduled market brief generation completed")
        except Exception as e:
            logger.error(f"Error in scheduled market brief generation: {str(e)}")

    background_tasks.add_task(generate_brief_task)
    return {"message": "Market brief generation scheduled"}


@app.get("/audio/{filename}")
async def get_audio_file(filename: str):
    """Get an audio file by filename.

    Args:
        filename: The name of the audio file.

    Returns:
        The audio file.
    """
    file_path = f"audio_cache/{filename}"
    return FileResponse(file_path, media_type="audio/mpeg")


# Run the application
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=Config.FASTAPI_HOST, port=Config.FASTAPI_PORT)
