# AI Tool Usage and Configurations

## 1. Overview

This document logs the usage of AI tools, models, and configurations within the Finance Assistant project. It aims to provide transparency on how AI assistance was leveraged for code generation and refactoring, and to detail the parameters of the AI models used in the application, as per project requirements.

## 2. Prompts and Code Generation (Illustrative Examples)

The codebase for this Finance Assistant project was significantly developed and refactored with the assistance of "Jules," a software engineering agent developed by Google. This AI agent helped in generating complex code structures, implementing features, and refactoring existing code for better performance, clarity, and adherence to best practices.

Below are illustrative examples of prompts that could have been used to generate or refactor key components of the system. These demonstrate the nature and complexity of the tasks handled with AI assistance.

*   **For `AnalysisAgent._calculate_risk` method:**
    ```
    "Refactor the `AnalysisAgent._calculate_risk` method in Python to correctly calculate annualized volatility, Sharpe ratio (given a risk-free rate), max drawdown, historical VaR (95th percentile), and Beta against a benchmark. Input should be pandas DataFrames for asset and benchmark historical prices ('Date', 'Close'). Ensure proper date alignment and handling of missing data."
    ```

*   **For a LangGraph node in `LanguageAgent`:**
    ```
    "Generate the Python code for a LangGraph node in `LanguageAgent` that takes the current state (including a user query and summarized context) and uses an initialized `ChatOpenAI` model (`self.llm`) to generate a final response. Include error handling for the LLM call."
    ```

*   **For the System Architecture Diagram:**
    ```
    "Draft a Mermaid syntax diagram for a multi-agent financial system with components: User, Streamlit UI, FastAPI Orchestrator, specialized agents (API, Scraping, Analysis, Language, Voice, Retriever), Data Ingestion modules, Pinecone, Redis, and external data sources."
    ```

*   **For `RetrieverAgent` - Embedding Caching Logic:**
    ```
    "Design a Python caching mechanism for sentence embeddings in the RetrieverAgent. It should use Redis. The cache key should incorporate the embedding model name and a hash of the text. Embeddings should be stored as bytes and retrieved as numpy arrays. Implement methods for getting from cache and putting into cache with an expiry of 7 days."
    ```

*   **For `Orchestrator` - `_gather_data_for_query` method:**
    ```
    "Develop a Python async method for the Orchestrator called '_gather_data_for_query'. It should take a user query string and retrieved context (list of dicts) as input. Based on keywords in the query (e.g., 'portfolio', 'market', 'news', stock tickers), it should call appropriate methods on other agents (api_agent, analysis_agent, scraping_agent) to fetch relevant data. Consolidate all fetched data into a dictionary. Include basic ticker extraction logic for common stock symbols."
    ```

*   **For `FirecrawlScraper` - `crawl_website` method (using `firecrawl-py` SDK):**
    ```
    "Implement an async method 'crawl_website' in the FirecrawlScraper class. This method should use the 'crawl_url' function from the 'firecrawl-py' SDK (e.g., 'self.app.crawl_url') executed via 'asyncio.to_thread' due to its synchronous nature. The method should accept a URL to crawl ('url_to_crawl'), an optional dictionary 'crawler_options' (to be passed as 'params' to the SDK's 'crawl_url'), and an optional 'timeout_seconds' (defaulting to 180). Process the list of results from the SDK; each item in the list should be transformed into a dictionary containing 'success' (bool), 'url' (str, from 'sourceURL' or 'url' in SDK item), 'content' (str, preferring 'markdown' over 'content'), 'metadata' (dict), and 'raw_data' (the original SDK item for that page). Implement comprehensive error handling for SDK calls (general exceptions) and asyncio timeouts, returning a list containing a single error dictionary in case of such failures."
    ```

These examples represent a fraction of the interactions but showcase how AI was used to accelerate development, implement complex algorithms, and structure the application.

## 3. AI Model Parameters and Configurations

### Primary Language Model (LLM)

*   **Model Used:** OpenAI model specified by `Config.OPENAI_CHAT_MODEL_NAME`, defaulting to `gpt-4o`. Previously `gpt-4-1106-preview`.
    *   This model (e.g., `gpt-4o`) is selected for its strong reasoning capabilities, large context window, speed, and improved cost-effectiveness, suitable for tasks within the `LanguageAgent`.
*   **Configuration in `LanguageAgent`:**
    *   `temperature`: `0.7` (This setting balances creativity and determinism, suitable for generating diverse yet coherent financial narratives and responses).
*   **Usage:**
    *   Dynamic response generation in LangGraph workflows within the `LanguageAgent` (e.g., for market briefs and query responses).
    *   Query reformulation in the `Orchestrator` (via `LanguageAgent.reformulate_query`) to improve RAG retrieval quality.

### System Prompts for Language Agent

To enhance contextual relevance and guide the Language Agent's behavior, specific system prompts have been implemented:

*   **`SYSTEM_GUIDANCE_FINANCIAL_ASSISTANT`**:
    *   **Purpose**: Provides overall guidance to the LLM when acting as a financial assistant (e.g., for market brief generation, query responses).
    *   **Content Summary**: Instructs the LLM to be a specialized financial assistant, focus on data-driven insights related to finance, maintain a professional and analytical tone, and avoid speculation, financial advice, or non-financial topics. It also emphasizes basing responses on provided information.
    *   **Implementation**: Prepended to various prompt templates within the `LanguageAgent`'s graph workflows and helper methods.

*   **`SYSTEM_GUIDANCE_QUERY_REFORMULATION`**:
    *   **Purpose**: Guides the LLM specifically when reformulating user queries for better search results.
    *   **Content Summary**: Instructs the LLM to act as an AI assistant refining financial queries, aiming for clarity, specificity, or appropriate breadth while strictly maintaining financial context, and to return only the reformulated query.
    *   **Implementation**: Prepended to the prompt template in the `LanguageAgent.reformulate_query` method.

### Embedding Models (SentenceTransformers)

*   **Primary Model:** `roberta-large-nli-stsb-mean-tokens`
    *   **Dimensions:** 1024 (Selected to match the user's existing Pinecone index configuration).
    *   **Configuration:** Loaded via `SentenceTransformer(Config.EMBEDDING_MODEL)` in `RetrieverAgent` using default parameters from the library.
    *   **Usage:** Generating embeddings for text data (documents, web content) for storage in Pinecone and for generating query embeddings at search time. This is central to the RAG pipeline managed by `RetrieverAgent`.
    *   **Caching:** Embeddings generated by this model are cached in Redis to reduce redundant computations, as implemented in `RetrieverAgent`.

*   **Secondary Model (Conceptual):**
    *   **Configuration:** Loaded if a model name/path is specified in `Config.SECONDARY_EMBEDDING_MODEL` (e.g., `stsb-xlm-r-multilingual`). This model also has 1024 dimensions, allowing for direct comparison or use with the primary Pinecone index.
    *   **Usage:** The `RetrieverAgent` is equipped to load this model. Potential experimental uses include comparing retrieval results against the primary RoBERTa model, exploring multilingual capabilities, or using it in ensemble methods. No active use-case has been implemented in the current version.

### Speech-to-Text (STT)

*   **Model:** OpenAI Whisper.
*   **Default Loaded:** The `base.en` model is loaded by default in `VoiceAgent.initialize_stt_model()` if no fine-tuned model is specified.
    *   **Rationale for "base.en":** Chosen as a good balance of speed and performance for English language audio, and allows for more focused fine-tuning if desired. It can be replaced by a user-specified fine-tuned model via configuration.
*   **Custom Model Support:** The system supports loading a pre-fine-tuned Whisper model if its path or Hugging Face Hub name is provided via the `Config.WHISPER_FINETUNED_MODEL_PATH` environment variable.
*   **Note on Model Choice:** The default model has been set to `base.en` to provide a lightweight, English-focused starting point. Users can specify a different or fine-tuned model via `Config.WHISPER_FINETUNED_MODEL_PATH`.

### Text-to-Speech (TTS)

*   **Sarvam AI:**
    *   **Model:** Uses the voice model specified in `Config.VOICE_MODEL` (e.g., "meera" for an Indian female voice).
    *   **API Parameters (as per `VoiceAgent._sarvam_ai_tts`):**
        *   `model`: Value from `Config.VOICE_MODEL`.
        *   `voice_preset`: `"default"`.
        *   `audioformat`: `"mp3"`.
        *   `pitch`: `0` (Integer value).
        *   `speaking_rate`: `1.0` (Float value, controls pace).
    *   **Note on Pitch/Pace:** The `VoiceAgent._sarvam_ai_tts` method now includes `"pitch": 0` and `"speaking_rate": 1.0` in the API payload to align with typical TTS customization options.

*   **ElevenLabs (Fallback):**
    *   **Model ID:** `eleven_monolingual_v1` (as specified in `VoiceAgent._elevenlabs_tts`).
    *   **API Parameters:**
        *   `voice_settings`: `{"stability": 0.5, "similarity_boost": 0.5}`.
        *   Voice ID: Selected via a basic internal mapping in `VoiceAgent` (e.g., "meera" maps to a specific ElevenLabs voice ID) or falls back to a default ID if the configured voice is not in the map.

## 4. Fine-Tuning Steps

### Whisper STT Model

*   **Current Status:** No actual fine-tuning of the Whisper model has been performed by this AI assistant ("Jules") as part of the project development. The system is designed to *support loading* a pre-fine-tuned model if one is provided by the user via the `WHISPER_FINETUNED_MODEL_PATH` configuration.
*   **Conceptual Guidance for User-Performed Fine-Tuning:**
    Should the user wish to fine-tune a Whisper model for improved accuracy on specific financial jargon, accents, or noisy environments, the general process would involve:
    1.  **Data Collection:** Gather a high-quality dataset of audio recordings relevant to the target domain (e.g., financial news broadcasts, earnings calls, customer service interactions with financial queries).
    2.  **Accurate Transcription:** Transcribe these audio recordings meticulously. The quality of transcriptions is crucial for effective fine-tuning.
    3.  **Training Environment Setup:** Utilize a machine learning framework, typically Hugging Face Transformers, along with libraries like `datasets`, `torchaudio`, and an appropriate training accelerator (GPU).
    4.  **Model Preparation:** Start with a pre-trained Whisper model checkpoint (e.g., `openai/whisper-base` or `openai/whisper-medium`). Load the model, tokenizer, feature extractor, and processor.
    5.  **Dataset Preparation:** Convert the collected audio and transcriptions into a format suitable for the training library (e.g., a Hugging Face `Dataset` object). This includes audio resampling, feature extraction, and tokenization of transcripts.
    6.  **Training:** Define training arguments (e.g., learning rate, number of epochs, batch size, evaluation strategy) and use the Hugging Face `Trainer` or a custom training loop to fine-tune the model on the prepared dataset.
    7.  **Evaluation:** Evaluate the fine-tuned model on a separate test set using metrics like Word Error Rate (WER) to assess performance improvement.
    8.  **Saving the Model:** Save the fine-tuned model artifacts (including model weights, configuration files, and any processor/tokenizer files) to a local directory or upload it to the Hugging Face Hub.
    9.  **Configuration:** Set the `WHISPER_FINETUNED_MODEL_PATH` environment variable in the `.env` file to point to the local directory path or the Hugging Face Hub model name. The `VoiceAgent` will then attempt to load this model.
    *   **Note on Compatibility:** As mentioned in the `VoiceAgent`'s docstring, ensure the saved fine-tuned model format is compatible with `openai-whisper`'s `load_model()` function, or adapt the loading mechanism in `VoiceAgent` if using a pure Hugging Face Transformers checkpoint.
