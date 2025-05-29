# Finance Agent Setup Guide

## External Configuration Requirements

This guide will help you set up all external configurations and API keys needed for the Finance Agent system to function properly.

## Quick Start

The easiest way to configure the system is to run it once, which will prompt you for any missing API keys and configurations. These will be saved to your `.env` file automatically.

```bash
python -m streamlit_app.app
```

## Manual Configuration

If you prefer to configure everything manually, follow these steps:

### 1. Environment File Setup

Copy the example environment file:

```bash
cp .env.example .env
```

Then edit the `.env` file with your actual API keys and configuration values.
This includes setting your desired `EMBEDDING_MODEL` (primary model for generating embeddings) and optionally the `SECONDARY_EMBEDDING_MODEL` (for experimentation, e.g., re-ranking). Refer to the comments in `.env.example` for model choices and their implications (like embedding dimensions).

### 2. Required API Keys

#### Vector Database (Pinecone)

1. **Create a Pinecone Account**:
   - Go to [Pinecone](https://www.pinecone.io/) and sign up for an account
   - Choose the free tier to start

2. **Get API Key**:
   - Navigate to API Keys in your Pinecone dashboard
   - Create a new API key or copy your existing one
   - Note your environment (e.g., `us-west1-gcp`)

3. **Add to Environment**:

```bash
PINCONE_API_KEY=your_pinecone_api_key
PINCONE_ENVIRONMENT=your_pinecone_environment
```

#### Voice Processing

You need at least ONE of these voice API providers:

##### Option 1: Sarvam AI (Recommended for Indian voices)

1. **Create a Sarvam AI Account**:
   - Go to [Sarvam AI](https://www.sarvam.ai/) and sign up

2. **Get API Key**:
   - Navigate to your account settings
   - Generate and copy your API key

3. **Add to Environment**:
   ```
   SARVAM_AI_API_KEY=your_sarvam_ai_api_key
   ```

##### Option 2: ElevenLabs

1. **Create an ElevenLabs Account**:
   - Go to [ElevenLabs](https://elevenlabs.io/) and sign up
   - The free tier provides limited usage

2. **Get API Key**:
   - Go to your profile settings
   - Copy your API key

3. **Add to Environment**:
   ```
   ELEVENLABS_API_KEY=your_elevenlabs_api_key
   ```

#### Language Model (LLM)

Configure at least ONE of these LLM providers:

##### Option 1: OpenAI (Recommended)

1. **Create an OpenAI Account**:
   - Go to [OpenAI](https://platform.openai.com/) and sign up

2. **Get API Key**:
   - Navigate to API keys in your account
   - Create a new secret key

3. **Add to Environment**:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ```

##### Option 2: Anthropic

1. **Create an Anthropic Account**:
   - Go to [Anthropic](https://www.anthropic.com/) and sign up

2. **Get API Key**:
   - Generate an API key from your account

3. **Add to Environment**:
   ```
   ANTHROPIC_API_KEY=your_anthropic_api_key
   ```

##### Option 3: Local Models

For local model deployment (advanced):

1. **Configure Local Model**:
   ```
   LLM_PROVIDER=local
   LOCAL_MODEL_PATH=/path/to/your/model
   ```

#### Web Scraping (Firecrawl)

1. **Create a Firecrawl Account**:
   - Go to [Firecrawl](https://firecrawl.dev/) and sign up

2. **Get API Key**:
   - Copy your API key from the dashboard

3. **Add to Environment**:
   ```
   FIRECRAWL_API_KEY=your_firecrawl_api_key
   ```

### 3. Database Configuration

#### Redis (for Caching)

1. **Local Redis**:
   - Default configuration works with local Redis or Docker setup:
   ```
   REDIS_HOST=localhost
   REDIS_PORT=6379
   ```

2. **Remote Redis**:
   - For hosted Redis (e.g., Redis Labs):
   ```
   REDIS_HOST=your-redis-host.redislabs.com
   REDIS_PORT=10000
   REDIS_PASSWORD=your_redis_password
   ```

### 4. Application Settings

#### General Settings

```
DEBUG=True  # Set to False in production
LOG_LEVEL=INFO  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
```

#### Market Brief Scheduling

```
TIMEZONE=Asia/Kolkata  # Set to your local timezone
MARKET_BRIEF_HOUR=8  # Hour to deliver daily market brief (24-hour format)
MARKET_BRIEF_MINUTE=0  # Minute to deliver daily market brief
```

#### FastAPI Settings

```
FASTAPI_HOST=0.0.0.0  # Listen on all interfaces
FASTAPI_PORT=8000  # Port for the API server
```

#### Streamlit Settings

```
STREAMLIT_PORT=8501  # Port for the Streamlit app
```

## Vector Database Setup

### Pinecone Index

The system will automatically create a Pinecone index if it doesn't exist, but you can also create one manually:

1. Go to your Pinecone dashboard
2. Create a new index with:
   - Name: `finance-data` (default, can be changed in config)
   - Dimension: 1024 (using the primary embedding model, e.g., roberta-large-nli-stsb-mean-tokens)
   - Metric: cosine

## Testing Your Configuration

After setting up all required configurations:

1. **Test API Connections**:
   ```bash
   python -m tests.test_api_connections
   ```

2. **Verify Vector Store**:
   ```bash
   python -m tests.test_vector_store
   ```

## Troubleshooting

### Common Issues

#### API Key Issues

- **Pinecone Connection Error**: Verify your API key and environment are correct
- **Voice API Error**: Ensure you've set up at least one voice provider correctly
- **LLM API Error**: Check your LLM provider key and quota limits

#### Redis Connection

- **Connection Refused**: Ensure Redis is running on the specified host and port
- **Authentication Error**: Verify Redis password if using a password-protected instance

#### Missing Configurations

If you see prompts asking for API keys when starting the application, it means some required configurations are missing from your `.env` file. Follow the prompts to input them or add them manually to the file.

## Security Considerations

- Never commit your `.env` file to version control
- Restrict access to your API keys and credentials
- Consider using environment-specific configuration files for development and production

## Next Steps

Once your configuration is complete, you can:

1. **Start the Application**:
   ```bash
   docker-compose up
   ```

2. **Access the UI**:
   - Open your browser to `http://localhost:8501`

3. **Test the API**:
   - Access the API documentation at `http://localhost:8000/docs`
