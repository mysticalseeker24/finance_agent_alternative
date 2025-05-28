# Finance Assistant

## Overview
A multi-agent finance assistant delivering spoken market briefs via a Streamlit app. It integrates YFinance, Scrapy, Firecrawl, Pinecone, Langgraph, Whisper, and Sarvam AI/ElevenLabs for data ingestion, RAG, and voice interactions.

## Features
- Daily market brief delivery at 8 AM covering market trends and portfolio details
- Voice response to complex financial queries (e.g., risk exposure in Asia tech stocks)
- Multi-source data integration (APIs, web scraping, documents) with RAG
- Specialized agents orchestrated via FastAPI microservices
- Voice interactions support

## System Architecture

### Data Ingestion
- YFinance for market data
- Scrapy/Firecrawl for web scraping
- PDFPlumber/PyPDF2 for document processing

### Vector Store for RAG
- Pinecone with SentenceTransformers

### Specialized Agents
- API Agent: YFinance
- Scraping Agent: Scrapy, Firecrawl
- Retriever Agent: Pinecone, LangChain
- Analysis Agent: Pandas, NumPy
- Language Agent: Langgraph, LangChain
- Voice Agent: Whisper (STT), Sarvam AI/ElevenLabs (TTS)

### Orchestration
- FastAPI microservices

### Deployment
- Streamlit, Docker, GitHub Actions

## Setup Instructions

### Prerequisites
- Python 3.9+
- Docker and Docker Compose (optional for containerized deployment)
- API keys for various services (Pinecone, Sarvam AI/ElevenLabs)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/mysticalseeker24/finance_agent_windsurf.git
cd finance_agent_windsurf
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

4. Run the FastAPI server:
```bash
uvicorn orchestrator.main:app --reload
```

5. Launch the Streamlit app:
```bash
streamlit run streamlit_app/app.py
```

## Docker Deployment

```bash
docker-compose up -d
```

## Project Structure

```
/data_ingestion: Contains scripts for data ingestion pipelines.
/agents: Implements specialized agents.
/orchestrator: FastAPI application for agent coordination.
/streamlit_app: Streamlit application for user interface.
/tests: Unit and integration tests using pytest.
```

## License
MIT

## Contributors
- [Your Name]
