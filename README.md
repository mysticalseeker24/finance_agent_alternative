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
*Placeholder for System Architecture Overview. Refer to `docs/architecture.md` for full details.*

## 4. Project Flow / Workflows
*Placeholder for Project Flow / Workflows. Details on Query Processing, Market Brief Generation, and Data Ingestion will be added here.*

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
*Placeholder for Setup and Installation. Refer to `docs/setup_guide.md` for detailed instructions.*

## 7. Usage
*Placeholder for Usage instructions.*

## 8. Project Structure
```
/data_ingestion: Contains scripts for data ingestion pipelines.
/agents: Implements specialized agents.
/orchestrator: FastAPI application for agent coordination.
/streamlit_app: Streamlit application for user interface.
/tests: Unit and integration tests using pytest.
```

## 9. Configuration
*Placeholder for Configuration highlights.*

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
