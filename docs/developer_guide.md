# Developer Guide

## Getting Started

This guide is intended to help developers understand and contribute to the Finance Agent project.

### Prerequisites

- Python 3.9 or higher
- Docker and docker-compose (for containerized development)
- Git

### Development Environment Setup

1. **Clone the repository**

```bash
git clone https://github.com/mysticalseeker24/finance_agent_windsurf.git
cd finance_agent_windsurf
```

2. **Create a virtual environment**

```bash
python -m venv venv
```

3. **Activate the virtual environment**

On Windows:
```bash
venv\Scripts\activate
```

On macOS/Linux:
```bash
source venv/bin/activate
```

4. **Install dependencies**

```bash
pip install -r requirements.txt
```

5. **Set up environment variables**

Copy the example environment file:
```bash
cp .env.example .env
```

Edit the `.env` file with your API keys and configuration values.

### Running the Application

#### Using Docker

The recommended way to run the application is using Docker:

```bash
docker-compose up --build
```

This will start all required services:
- FastAPI backend on port 8000
- Streamlit frontend on port 8501
- Redis for caching on port 6379

#### Manual Development Mode

For development, you can run the services separately:

1. **Start Redis**

```bash
docker-compose up -d redis
```

2. **Run the FastAPI backend**

```bash
uvicorn orchestrator.main:app --reload --host 0.0.0.0 --port 8000
```

3. **Run the Streamlit frontend**

```bash
streamlit run streamlit_app/app.py
```

### Project Structure

```
finance_agent/
├── agents/                  # Specialized agent implementations
│   ├── __init__.py
│   ├── base_agent.py        # Base agent class with common functionality
│   ├── api_agent.py         # Agent for market data APIs
│   ├── scraping_agent.py    # Agent for web scraping
│   ├── retriever_agent.py   # Agent for vector store operations
│   ├── analysis_agent.py    # Agent for financial analysis
│   ├── language_agent.py    # Agent for text generation
│   └── voice_agent.py       # Agent for voice processing
├── data_ingestion/          # Data collection modules
│   ├── __init__.py
│   ├── yfinance_client.py   # YFinance API client
│   ├── scrapy_spiders.py    # Scrapy spiders for web scraping
│   ├── document_loader.py   # PDF document processing
│   └── firecrawl_scraper.py # Dynamic content scraping
├── orchestrator/            # Coordination and API endpoints
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   └── orchestrator.py      # Agent coordination logic
├── streamlit_app/           # Streamlit UI
│   ├── __init__.py
│   └── app.py               # Main Streamlit application
├── tests/                   # Test modules
│   ├── __init__.py
│   ├── test_api_agent.py    # Tests for API agent
│   └── test_retriever_agent.py # Tests for retriever agent
├── docs/                    # Documentation
│   ├── architecture.md      # System architecture
│   ├── api_reference.md     # API documentation
│   └── developer_guide.md   # This guide
├── .github/                 # GitHub configuration
│   └── workflows/           # GitHub Actions workflows
│       └── ci.yml           # CI pipeline configuration
├── .env.example             # Environment variable template
├── config.py                # Configuration management
├── docker-compose.yml       # Docker services configuration
├── Dockerfile               # Docker image definition
├── pytest.ini               # Test configuration
├── README.md                # Project overview
└── requirements.txt         # Python dependencies
```

## Development Workflow

### Adding New Features

1. **Create a new branch**

```bash
git checkout -b feature/your-feature-name
```

2. **Implement your changes**

Follow the existing code style and architecture. Add appropriate tests for your changes.

3. **Run tests locally**

```bash
pytest
```

4. **Create a pull request**

Push your branch and create a pull request on GitHub.

### Coding Standards

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Include docstrings for all functions, classes, and modules
- Keep functions small and focused on a single responsibility
- Use async/await for I/O-bound operations

### Testing Guidelines

- Write unit tests for all new functionality
- Mock external dependencies to avoid actual API calls during tests
- Add integration tests for complex interactions between components
- Aim for high test coverage, especially for critical components

## Working with Agents

Each agent in the system is designed to handle specific responsibilities:

### Adding a New Agent

1. Create a new file in the `agents/` directory
2. Extend the `BaseAgent` class to inherit common functionality
3. Implement the required methods:
   - `__init__`: Initialize the agent and its resources
   - `initialize`: Set up connections and resources
   - `process`: Handle incoming requests and return responses
   - `run`: Execute the agent's main functionality

Example:

```python
from agents.base_agent import BaseAgent

class NewAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(name="New Agent")
        # Initialize agent-specific properties
    
    async def initialize(self):
        # Set up connections and resources
        return True
    
    async def process(self, request):
        # Process the incoming request
        operation = request.get("operation")
        parameters = request.get("parameters", {})
        
        if operation == "some_operation":
            result = await self._handle_operation(parameters)
            return self.create_response(result)
        
        return self.create_error_response("Unknown operation")
    
    async def run(self, request):
        # Main entry point for the agent
        return await self.process(request)
    
    async def _handle_operation(self, parameters):
        # Internal method to handle a specific operation
        # Implement your logic here
        return {"result": "success"}
```

4. Register the agent with the orchestrator in `orchestrator/orchestrator.py`

### Modifying Existing Agents

When modifying an existing agent:

1. Ensure backward compatibility or update all dependent components
2. Update tests to reflect changes
3. Document any API changes in the API reference

## Working with the Vector Store

The Finance Agent uses Pinecone for vector storage and retrieval. Here's how to work with it:

### Indexing New Content

```python
from agents.retriever_agent import RetrieverAgent

async def index_financial_news(news_articles):
    agent = RetrieverAgent(index_name="finance-news")
    await agent.initialize()
    
    texts = [article["content"] for article in news_articles]
    metadata = [{"title": article["title"], "source": article["source"], 
                "date": article["date"]} for article in news_articles]
    
    result = await agent.index_content(texts, metadata, namespace="news")
    return result
```

### Searching for Content

```python
from agents.retriever_agent import RetrieverAgent

async def search_financial_content(query):
    agent = RetrieverAgent(index_name="finance-data")
    await agent.initialize()
    
    result = await agent.search_content(query, namespace="all", top_k=5)
    return result
```

## API Development

The Finance Agent uses FastAPI for its backend API:

### Adding a New Endpoint

1. Open `orchestrator/main.py`
2. Define a new route using FastAPI decorators
3. Implement the endpoint handler function
4. Document the endpoint in `docs/api_reference.md`

Example:

```python
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/custom", tags=["custom"])

class CustomRequest(BaseModel):
    parameter1: str
    parameter2: int = 10

class CustomResponse(BaseModel):
    result: str
    status: str

@router.post("/process", response_model=CustomResponse)
async def process_custom_request(request: CustomRequest):
    try:
        # Process the request
        result = await custom_processing_logic(request.parameter1, request.parameter2)
        return CustomResponse(result=result, status="success")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def custom_processing_logic(param1, param2):
    # Implement your custom logic
    return f"Processed {param1} with parameter {param2}"
```

## Streamlit UI Development

The Finance Agent's frontend is built with Streamlit:

### Adding a New UI Component

1. Open `streamlit_app/app.py`
2. Create a new function for your component
3. Add your component to the appropriate section of the UI

Example:

```python
import streamlit as st
import plotly.express as px
import pandas as pd

def show_custom_visualization(data):
    st.subheader("Custom Visualization")
    
    # Create a DataFrame from the data
    df = pd.DataFrame(data)
    
    # Create a visualization using Plotly
    fig = px.line(df, x="date", y="value", color="category",
                  title="Custom Data Visualization")
    
    # Display the visualization
    st.plotly_chart(fig, use_container_width=True)
    
    # Add additional context or controls
    st.markdown("**Analysis:**")
    st.write("This visualization shows the trend of various categories over time.")
    
    # Add interactive elements if needed
    if st.button("Refresh Data"):
        st.session_state.data = fetch_updated_data()
        st.experimental_rerun()
```

## Troubleshooting

### Common Issues

#### API Authentication Errors

If you encounter authentication errors with external APIs:

1. Check that your API keys are correctly set in the `.env` file
2. Verify that the environment variables are being loaded in `config.py`
3. Ensure that your API subscription is active

#### Docker Networking Issues

If services can't communicate within Docker:

1. Check that all services are on the same network in `docker-compose.yml`
2. Verify that you're using service names as hostnames within the containers
3. Ensure ports are correctly mapped

#### Vector Store Connection Issues

If Pinecone connection fails:

1. Verify your Pinecone API key and environment
2. Check that your index exists in the Pinecone console
3. Ensure you're using the correct dimensionality for your embeddings

## Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [YFinance Documentation](https://pypi.org/project/yfinance/)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [Langchain Documentation](https://python.langchain.com/docs/)
