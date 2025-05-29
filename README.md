# Finance Assistant: Your Intelligent Multi-Agent Financial Analyst

## 1. Overview
The Finance Assistant is an advanced multi-agent system designed to provide users with timely financial insights, market analysis, and answers to complex financial queries. It leverages a suite of specialized AI agents to ingest data from diverse sources, process it, and deliver information through a user-friendly Streamlit interface, including voice interactions. This system aims to simplify financial data consumption and empower users to make more informed decisions. It's built for financial enthusiasts, analysts, and anyone looking to get quick, contextual answers to their finance-related questions.

## 2. Features
- **Automated Daily Market Briefs:** Delivers comprehensive market summaries at a scheduled time (e.g., 8 AM), covering market trends, key news, and configurable portfolio performance.
- **Voice-Driven Financial Queries:** Ask complex financial questions in natural language (e.g., "What's my portfolio's exposure to Asian tech stocks?") and receive spoken responses.
- **Retrieval-Augmented Generation (RAG):** Utilizes a vector store (Pinecone) with advanced sentence embeddings to provide contextually relevant answers from ingested documents and web content.
- **Multi-Source Data Integration:** Gathers information from:
    - Financial APIs (YFinance for market data).
    - Web Scraping (Firecrawl SDK for dynamic sites, Scrapy for static sites).
    - Document Processing (PDFs for financial reports).
- **Specialized AI Agents:** Employs a modular system of agents, each an expert in its domain (API interaction, data scraping, retrieval, analysis, language generation, voice processing), orchestrated via FastAPI.
- **Configurable Portfolio Analysis:** Allows users to define a default portfolio for analysis and brief generation via environment settings.
- **Extensible Architecture:** Designed with modularity for easier addition of new data sources, agents, or functionalities.
- **CI/CD Ready:** Includes GitHub Actions workflow for continuous integration and testing.
- **Dockerized Deployment:** Supports containerized deployment using Docker and Docker Compose for ease of setup and scalability.

## 3. System Architecture

The Finance Assistant employs a sophisticated multi-agent architecture designed for modularity and scalability. It integrates various specialized services to deliver comprehensive financial insights through a Streamlit-based user interface.

**Key Architectural Layers:**

*   **Presentation Layer (Streamlit UI):** Provides the user interface for text and voice interactions, displays information, and manages user preferences.
*   **Orchestration Layer (FastAPI Backend & Orchestrator):** Exposes API endpoints, manages user requests, and coordinates the workflow between various specialized agents.
*   **Agent System (Specialized Agents):** A suite of agents each responsible for a specific task:
    *   `APIAgent`: Fetches data from financial APIs (e.g., YFinance).
    *   `ScrapingAgent`: Performs web scraping (static/dynamic sites) using Scrapy and Firecrawl-py SDK.
    *   `RetrieverAgent`: Manages interactions with the Pinecone vector store for Retrieval-Augmented Generation (RAG).
    *   `AnalysisAgent`: Conducts quantitative financial analysis and calculations.
    *   `LanguageAgent`: Handles natural language understanding, generation, and query reformulation using OpenAI models via LangChain and LangGraph.
    *   `VoiceAgent`: Manages Speech-to-Text (Whisper) and Text-to-Speech (Sarvam AI, ElevenLabs) operations.
*   **Data Ingestion Layer:** Collects and processes data from diverse sources including APIs, web pages (via scraping/crawling), and PDF documents.
*   **Data Storage & Caching:**
    *   `Pinecone`: Vector database for storing embeddings for RAG.
    *   `Redis`: Used for caching embeddings and LLM responses to improve performance and reduce redundant computations.

**High-Level Diagram:**

```ascii
┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│                   │     │                   │     │                   │
│  Streamlit UI     │◄────┤  FastAPI Backend  │◄────┤  Agent System     │
│  (Presentation)   │     │  (Orchestration)  │     │  (Processing)     │
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

For a more detailed component breakdown and data flow, please refer to the [System Architecture Document](./docs/architecture.md).

## 4. Project Flow / Workflows

The Finance Assistant operates through several key workflows to deliver its functionalities:

### Query Processing Flow

1.  **Input:** User submits a query via text or voice through the Streamlit UI.
2.  **Speech-to-Text (STT):** If voice input is used, the `VoiceAgent` converts speech to text using the configured Whisper model.
3.  **Context Retrieval (RAG):** The `Orchestrator` routes the query (now text) to the `RetrieverAgent`. The `RetrieverAgent` generates an embedding for the query and searches the Pinecone vector store for relevant documents/context.
4.  **Query Reformulation (if needed):** If initial search results are insufficient, the `LanguageAgent` may be called to reformulate the query for better retrieval.
5.  **Supplemental Data Gathering:** The `Orchestrator` determines if additional real-time data is needed based on the query and retrieved context. It may invoke:
    *   `APIAgent` for current market data or stock information.
    *   `ScrapingAgent` for recent news, articles, or specific web content via scrape or crawl operations.
    *   `AnalysisAgent` for financial calculations if portfolio data or specific metrics are involved.
6.  **Response Generation:** The `LanguageAgent` (using LangGraph workflows and an OpenAI model like GPT-4o) synthesizes all gathered information (original query, retrieved context, supplemental data) to generate a coherent, contextual answer.
7.  **Text-to-Speech (TTS):** If voice output is enabled, the `VoiceAgent` converts the textual response to speech using Sarvam AI or ElevenLabs.
8.  **Output:** The Streamlit UI displays the textual response and provides an interface to play the audio if generated.

### Market Brief Generation Flow

1.  **Trigger:** This flow is typically triggered by a scheduler (e.g., daily at 8 AM as per configuration) but can also be initiated on demand.
2.  **Data Collection:**
    *   The `Orchestrator` calls the `APIAgent` to fetch current market summary data (e.g., major indices).
    *   The `ScrapingAgent` is invoked to retrieve recent financial news from configured sources.
    *   The default portfolio (from `Config.DEFAULT_PORTFOLIO`) is retrieved.
3.  **Data Analysis:** The `AnalysisAgent` processes the default portfolio data to calculate performance and relevant metrics.
4.  **Narrative Generation:** The `LanguageAgent` uses its market brief generation LangGraph workflow. It takes the market data, news articles, and portfolio analysis as input to generate a structured and narrative market brief.
5.  **Speech Synthesis:** The generated brief's text is converted to speech by the `VoiceAgent`.
6.  **Delivery/Storage:** The final brief (text and audio URL) is made available, typically for display in the Streamlit UI. (Further storage or notification mechanisms could be added).

### Data Ingestion Flow (Conceptual)

1.  **Source Identification:** Data sources include financial APIs (YFinance), websites (news, reports), and PDF documents.
2.  **Extraction:**
    *   `APIAgent` fetches structured data from APIs.
    *   `ScrapingAgent` (using Firecrawl SDK and Scrapy) extracts content from web pages. This can be targeted scraping or broader website crawling.
    *   `DocumentLoader` extracts text and tables from PDFs.
3.  **Processing & Embedding:** Extracted textual content is chunked, cleaned, and then converted into vector embeddings by the `RetrieverAgent` using a SentenceTransformer model.
4.  **Indexing:** The `RetrieverAgent` upserts these embeddings along with their associated metadata (source, text content, etc.) into the Pinecone vector store for later retrieval.
5.  **Caching:** Embeddings are cached in Redis to avoid re-computation.

## 5. Technology Stack

| Category              | Technologies                                                                 | Purpose                                                                 |
|-----------------------|------------------------------------------------------------------------------|-------------------------------------------------------------------------|
| **Backend Framework**   | FastAPI, Uvicorn                                                             | Building efficient, asynchronous APIs; ASGI server                       |
| **Frontend UI**       | Streamlit                                                                    | Creating interactive web applications for data and chat                 |
| **Data Ingestion**    | YFinance, Scrapy, Firecrawl-py SDK, PDFPlumber, PyPDF2                     | Market data, web scraping (static/dynamic), PDF text/table extraction |
| **Database/Storage**  | Pinecone, Redis                                                              | Vector embeddings storage (RAG), caching (embeddings, LLM responses)    |
| **AI - Core**         | LangChain, LangGraph                                                         | LLM application development, multi-agent orchestration (graphs)         |
| **AI - Language Models**| OpenAI (configurable, e.g., GPT-4o), SentenceTransformers                  | Text generation, understanding, query reformulation; text embeddings      |
| **AI - Voice**        | OpenAI Whisper, Sarvam AI, ElevenLabs                                        | Speech-to-Text (STT), Text-to-Speech (TTS)                              |
| **Orchestration**     | Custom Python (Orchestrator class)                                           | Coordinating agent interactions and workflows                           |
| **Containerization**  | Docker, Docker Compose                                                       | Packaging, deployment, and service management                           |
| **CI/CD**             | GitHub Actions                                                               | Continuous integration, automated testing and linting                   |
| **Testing**           | Pytest, Pytest-Cov                                                           | Unit and integration testing, code coverage                             |
| **Code Quality**      | Flake8, Black, Isort                                                         | Linting, code formatting, import sorting                                |
| **Utilities**         | Python-dotenv, Loguru, Pydub, Sounddevice, Asyncio, Pandas, NumPy            | Environment vars, logging, audio manipulation, async ops, data handling |

## 6. Setup and Installation

Follow these steps to get the Finance Assistant up and running on your local machine.

### Prerequisites
*   **Python:** Version 3.9 or higher.
*   **Pip:** Python package installer.
*   **Git:** For cloning the repository.
*   **Docker and Docker Compose (Recommended for easier setup):** For containerized deployment of the application and services like Redis.
*   **API Keys:** You will need API keys for various external services used by the agent. Refer to the "API Key Configuration" section below and the detailed [Setup Guide](./docs/setup_guide.md).

### Installation Steps

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/mysticalseeker24/finance_agent_windsurf.git
    cd finance_agent_windsurf
    ```

2.  **Set Up Environment Variables:**
    *   Copy the example environment file:
        ```bash
        cp .env.example .env
        ```
    *   Edit the `.env` file with your actual API keys and any custom configurations. Minimally, you'll need to add your `OPENAI_API_KEY` and `PINECONE_API_KEY` / `PINECONE_ENVIRONMENT`. For voice features, `SARVAM_AI_API_KEY` or `ELEVENLABS_API_KEY` are needed. `FIRECRAWL_API_KEY` is required for web scraping features.

3.  **Install Dependencies:**
    *   It's highly recommended to use a Python virtual environment:
        ```bash
        python -m venv venv
        source venv/bin/activate  # On Windows: venv\Scripts\activate
        ```
    *   Install the required Python packages:
        ```bash
        pip install -r requirements.txt
        ```

### Running the Application

There are two main ways to run the application:

**Option 1: Using Docker Compose (Recommended)**
This method starts all services (FastAPI backend, Streamlit frontend, Redis) in containers.
1.  Ensure Docker Desktop is running.
2.  From the project root directory, run:
    ```bash
    docker-compose up -d --build
    ```
3.  **Accessing the services:**
    *   Streamlit UI: `http://localhost:8501`
    *   FastAPI Backend (API Docs): `http://localhost:8000/docs`

**Option 2: Running Services Manually**
1.  **Start Redis:**
    *   If you have Redis installed locally, ensure it's running.
    *   Alternatively, start Redis using Docker: `docker-compose up -d redis` (ensure other services in `docker-compose.yml` are commented out or managed if you only want Redis via Docker).
2.  **Run the FastAPI Backend Server:**
    ```bash
    uvicorn orchestrator.main:app --reload --host 0.0.0.0 --port 8000
    ```
3.  **Launch the Streamlit Application:**
    ```bash
    streamlit run streamlit_app/app.py --server.port 8501
    ```

### API Key Configuration

The application requires API keys for several external services:
*   **OpenAI (`OPENAI_API_KEY`):** For language model capabilities.
*   **Pinecone (`PINECONE_API_KEY`, `PINECONE_ENVIRONMENT`):** For the vector database.
*   **Firecrawl (`FIRECRAWL_API_KEY`):** For web scraping.
*   **Sarvam AI (`SARVAM_AI_API_KEY`) / ElevenLabs (`ELEVENLABS_API_KEY`):** For Text-to-Speech (at least one is recommended for full voice functionality).

For detailed instructions on obtaining these keys and configuring all necessary environment variables, please refer to the comprehensive [**Setup Guide](./docs/setup_guide.md)**. That guide also covers optional settings like `OPENAI_CHAT_MODEL_NAME`, `EMBEDDING_MODEL`, and `DEFAULT_PORTFOLIO_JSON`.

## 7. Usage

Once the application is running (see "Running the Application" under Setup and Installation), you can interact with the Finance Assistant primarily through the Streamlit web interface.

1.  **Access the Streamlit UI:**
    *   Open your web browser and navigate to `http://localhost:8501` (or the configured port if you changed it).

2.  **Interacting with the Assistant:**
    *   **Text Input:** Use the chat input box to type your financial queries. Press Enter to send.
    *   **Voice Input:** Click the "Record Voice" button (or similar, if available) to ask your query using your microphone. The system will transcribe your speech and process the query.
    *   **Market Brief:** The application may display a daily market brief automatically, or there might be an option to request it. This brief provides a summary of market conditions, news, and portfolio performance.
    *   **Output Preferences:** Look for options to customize how you receive responses (e.g., text-only, voice-only, or both).

3.  **Example Queries:**
    *   "What's the latest news on Apple?"
    *   "Analyze the risk profile of my current portfolio." (Assumes default portfolio is configured or user-specific portfolio features are implemented)
    *   "What are the key drivers for the S&P 500 this week?"
    *   "Tell me about the recent performance of Microsoft stock."
    *   If configured, you can ask queries in supported languages (e.g., Hindi).

4.  **API Endpoints (for developers):**
    *   The FastAPI backend exposes several API endpoints for programmatic interaction with the agents and orchestrator.
    *   API documentation is typically available at `http://localhost:8000/docs` when the backend server is running. This allows developers to test individual agent functionalities or integrate the assistant's capabilities into other systems.

Please refer to specific UI elements and instructions within the Streamlit application for the most up-to-date interaction details.

## 8. Project Structure
```
/data_ingestion: Contains scripts for data ingestion pipelines.
/agents: Implements specialized agents.
/orchestrator: FastAPI application for agent coordination.
/streamlit_app: Streamlit application for user interface.
/tests: Unit and integration tests using pytest.
```

## 9. Configuration

The Finance Assistant's behavior and service integrations are primarily configured through environment variables.

1.  **Environment File (`.env`):**
    *   Create a `.env` file in the project root by copying the template: `cp .env.example .env`.
    *   Edit this file to include your API keys and custom settings.

2.  **Key Configurable Variables (see `.env.example` for a full list):**
    *   **API Keys:**
        *   `OPENAI_API_KEY`: For OpenAI language models.
        *   `PINECONE_API_KEY`, `PINECONE_ENVIRONMENT`: For Pinecone vector database.
        *   `FIRECRAWL_API_KEY`: For Firecrawl web scraping/crawling.
        *   `SARVAM_AI_API_KEY`, `ELEVENLABS_API_KEY`: For Text-to-Speech services.
    *   **Model Choices:**
        *   `OPENAI_CHAT_MODEL_NAME`: Specifies the OpenAI model (e.g., "gpt-4o", "gpt-4o-mini"). Defaults to "gpt-4o".
        *   `EMBEDDING_MODEL`: Primary SentenceTransformer model for embeddings.
        *   `SECONDARY_EMBEDDING_MODEL`: Optional secondary SentenceTransformer model for experimentation.
        *   `VOICE_MODEL`: Default voice for TTS (e.g., "meera" for Sarvam AI).
        *   `WHISPER_FINETUNED_MODEL_PATH`: Path or Hub name for a custom fine-tuned Whisper STT model.
    *   **Application Behavior:**
        *   `DEFAULT_PORTFOLIO_JSON`: Define a default portfolio as a JSON string.
        *   `MARKET_BRIEF_HOUR`, `MARKET_BRIEF_MINUTE`, `TIMEZONE`: Schedule for daily market briefs.
        *   `LOG_LEVEL`, `DEBUG`: Application logging and debug settings.
    *   **Service Endpoints (rarely changed unless self-hosting):**
        *   `REDIS_HOST`, `REDIS_PORT`: Connection details for Redis cache.

3.  **Detailed Setup:**
    *   For comprehensive information on all environment variables, their default values, and how to obtain necessary API keys, please refer to the [**Full Setup Guide](./docs/setup_guide.md)**.

Proper configuration is crucial for all features of the Finance Assistant to function correctly.

## 10. Framework Comparisons (Placeholder)
[This section is intended for a comparative analysis of frameworks or libraries considered or used, along with the rationale for choices. Contributions welcome.]

## 11. Performance Benchmarks (Placeholder)
[This section is intended for performance metrics, such as query response times, data ingestion speed, etc., under various conditions. Contributions welcome.]

## 12. Contributing
Contributions to the Finance Assistant are welcome! Please refer to the project's issue tracker and consider submitting pull requests for new features or bug fixes.

## 13. License
MIT

## 14. Acknowledgements / Contributors
(User to fill this section.)
