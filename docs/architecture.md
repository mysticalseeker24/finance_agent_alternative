# Finance Agent Architecture

## Overview

The Finance Agent is a multi-agent system that provides financial insights and analysis through natural language interactions. The system integrates multiple data sources, employs Retrieval-Augmented Generation (RAG), and supports voice interactions through a Streamlit interface.

## System Architecture Diagram

```ascii
┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│                   │     │                   │     │                   │
│  Streamlit UI     │◄────┤  FastAPI Backend  │◄────┤  Agent System     │
│                   │     │                   │     │                   │
└───────────────────┘     └───────────────────┘     └───────────────────┘
         ▲                          ▲                         ▲
         │                          │                         │
         │                          │                         │
         ▼                          ▼                         ▼
┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│                   │     │                   │     │                   │
│  Voice Processing │     │  Data Ingestion   │     │  Vector Store     │
│  (STT/TTS)        │     │  Pipelines        │     │  (Pinecone)       │
│                   │     │                   │     │                   │
└───────────────────┘     └───────────────────┘     └───────────────────┘
```

## Component Architecture

### 1. Data Ingestion Layer

The data ingestion layer is responsible for gathering data from various sources and processing it for use in the system.

#### YFinance Client

- Fetches real-time and historical market data from Yahoo Finance
- Provides functionality for stock information, historical prices, and market summaries
- Implements caching to reduce API calls and improve performance

#### Scrapy Spiders

- Crawls financial news websites and SEC filings
- Extracts structured data from HTML pages
- Processes and normalizes the extracted data

#### Document Loader

- Processes PDF documents using PDFPlumber and PyPDF2
- Extracts text and tables from financial reports
- Structures data for indexing in the vector store

#### Firecrawl Integration

- Handles dynamic content that requires JavaScript rendering
- Provides structured data from complex web pages
- Complements Scrapy for comprehensive web data extraction

### 2. Agent System

The agent system consists of specialized agents that work together to process queries and generate responses.

#### API Agent

- Interfaces with YFinance and other APIs
- Retrieves market data, stock information, and financial metrics
- Transforms raw data into structured formats for analysis

#### Scraping Agent

- Coordinates between Scrapy and Firecrawl
- Extracts financial news, earnings reports, and SEC filings
- Processes and normalizes web data

#### Retriever Agent

- Manages the vector store (Pinecone)
- Indexes and retrieves relevant context for queries
- Implements RAG for accurate information retrieval

#### Analysis Agent

- Performs quantitative financial analysis
- Calculates risk metrics, portfolio performance, and other financial indicators
- Generates visualizations and data insights

#### Language Agent

- Uses Langgraph and LangChain for narrative generation
- Synthesizes information from multiple sources
- Creates coherent and contextually relevant responses

#### Voice Agent

- Handles speech-to-text conversion using Whisper
- Manages text-to-speech using Sarvam AI/ElevenLabs
- Provides voice interaction capabilities

### 3. Orchestration Layer

The orchestration layer coordinates the interactions between agents and manages the overall workflow.

#### FastAPI Backend

- Exposes RESTful API endpoints
- Manages asynchronous requests and responses
- Handles authentication and security

#### Orchestrator

- Coordinates between specialized agents
- Implements workflows for query processing and market brief generation
- Manages error handling and recovery

### 4. Presentation Layer

#### Streamlit Application

- Provides a web-based user interface
- Displays market briefs, portfolio analysis, and query responses
- Supports voice interactions and visualizations

## Data Flow

### Query Processing Flow

1. User submits a query via text or voice through the Streamlit UI
2. If voice input, the Voice Agent converts speech to text
3. The Orchestrator routes the query to the Retriever Agent to gather relevant context
4. The Orchestrator determines which additional data is needed and calls appropriate agents:
   - API Agent for market data
   - Scraping Agent for news and filings
   - Analysis Agent for financial calculations
5. The Language Agent generates a response using the retrieved context and agent outputs
6. If voice output is enabled, the Voice Agent converts the text response to speech
7. The Streamlit UI displays the response and optional audio

### Market Brief Generation Flow

1. Triggered by scheduler or user request
2. The Orchestrator calls the API Agent to fetch market data
3. The Scraping Agent retrieves recent financial news
4. The Analysis Agent processes portfolio data
5. The Language Agent generates the market brief narrative
6. The Voice Agent converts the brief to speech
7. The result is stored and made available through the Streamlit UI

## Technical Implementation

### Vector Store and RAG

- Uses Pinecone for scalable vector storage and retrieval
- Implements SentenceTransformers for text embeddings
- Employs fallback mechanisms for handling ambiguous queries

### Asynchronous Processing

- Uses Python's asyncio for concurrent operations
- Implements asynchronous endpoints in FastAPI
- Manages background tasks for long-running processes

### Containerization and Deployment

- Uses Docker for containerization
- Implements docker-compose for service orchestration
- Configures GitHub Actions for CI/CD

## Security Considerations

- API keys are stored in environment variables
- CORS protection for API endpoints
- Input validation and sanitization

## Future Enhancements

- Additional data sources for more comprehensive insights
- Fine-tuning of embedding models for improved retrieval
- Enhanced visualization capabilities
- User authentication and personalization
- Alerting system for market events
