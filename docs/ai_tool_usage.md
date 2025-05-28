# AI Tool Usage Log

## Overview
This document logs the usage of AI tools, configurations, and methodologies employed during the development and refactoring of the Finance Agent project, as assisted by Jules, a software engineering agent from Google.

## Prompts and Code Generation (Illustrative Examples)
AI assistance was leveraged for code generation, refactoring, and analysis. Examples of prompts that could guide such tasks include:

*   **For `AnalysisAgent` Refactoring:** "Refactor the `AnalysisAgent._calculate_risk` method in Python to correctly calculate annualized volatility, Sharpe ratio (given a risk-free rate), max drawdown, historical VaR (95th percentile), and Beta against a benchmark. Input should be pandas DataFrames for asset and benchmark historical prices ('Date', 'Close'). Ensure proper date alignment and handling of missing data."
*   **For `LanguageAgent` LangGraph Node:** "Generate the Python code for a LangGraph node in `LanguageAgent` that takes the current state (including a user query and summarized context) and uses an initialized `ChatOpenAI` model (`self.llm`) to generate a final response. Include error handling for the LLM call."
*   **For Documentation (Mermaid Diagram):** "Draft a Mermaid syntax diagram for a multi-agent financial system with components: User, Streamlit UI, FastAPI Orchestrator, specialized agents (API, Scraping, Analysis, Language, Voice, Retriever), Data Ingestion modules, Pinecone, Redis, and external data sources."

## AI Model Parameters and Configurations

### Primary Language Model (LLM)
*   **Model Used:** OpenAI `gpt-4-1106-preview` (GPT-4 Turbo)
*   **Configuration in `LanguageAgent`:** Temperature `0.7`.
*   **Usage:** Dynamic response generation in LangGraph workflows, query reformulation.

### Embedding Models (SentenceTransformers)
*   **Primary Model:** `roberta-large-nli-stsb-mean-tokens`
    *   **Dimensions:** 1024 (to match your Pinecone index).
    *   **Configuration:** Loaded via `SentenceTransformer(Config.EMBEDDING_MODEL)` with default parameters.
    *   **Usage:** Generating embeddings for text data for RAG in `RetrieverAgent`.
*   **Secondary Model (Conceptual Support):**
    *   **Configuration:** Via `Config.SECONDARY_EMBEDDING_MODEL` (e.g., "all-distilroberta-v1").
    *   **Usage:** Loaded if specified; no active use-case (e.g., re-ranking) is currently implemented. System warns if dimensions do not match primary model's index.

### Speech-to-Text (STT)
*   **Model:** OpenAI Whisper.
*   **Default Loaded:** `medium` model (as per `VoiceAgent` fallback for general capability).
*   **Custom Model Support:** System supports loading a pre-fine-tuned model via `Config.WHISPER_FINETUNED_MODEL_PATH`.
*   **Note on Project Specification:** The original project document mentioned potentially using `base.en`. The "medium" model was chosen as the default for broader initial language support before any fine-tuning.

### Text-to-Speech (TTS)
*   **Sarvam AI:**
    *   **Model:** Uses `Config.VOICE_MODEL` (e.g., "meera").
    *   **API Parameters in `VoiceAgent`:** `voice_preset: "default"`, `audioformat: "mp3"`.
    *   **Note on Project Specification:** The project document mentioned desired parameters "pitch=0, pace=1.0". The `VoiceAgent` currently does not explicitly pass these specific pitch/pace parameters; it relies on the Sarvam API's default behavior for the selected model/preset. These could be added to the `VoiceAgent`'s Sarvam API call if required.
*   **ElevenLabs:**
    *   **Model ID in `VoiceAgent`:** `eleven_monolingual_v1`.
    *   **API Parameters in `VoiceAgent`:** `voice_settings: {"stability": 0.5, "similarity_boost": 0.5}`. Voice ID is selected via a basic internal mapping or a default.

## Fine-Tuning Steps

### Whisper STT Model
*   **Current Status:** No actual fine-tuning has been performed on the Whisper model by me during the refactoring.
*   **Conceptual Guidance for Your Performed Fine-tuning:**
    1.  **Data Collection:** Gather a corpus of domain-specific audio (e.g., financial news, earnings calls, typical user queries related to finance).
    2.  **Transcription:** Ensure accurate, verbatim transcriptions for all audio data.
    3.  **Tools:** Utilize libraries such as Hugging Face `transformers` and `datasets` for the fine-tuning process.
    4.  **Training:** Follow standard procedures for training sequence-to-sequence models, including data preprocessing with `WhisperProcessor`, setting up `Seq2SeqTrainingArguments`, and using the `Seq2SeqTrainer`.
    5.  **Model Saving:** Save the fine-tuned model checkpoint and its associated processor configuration.
    6.  **Integration:** Configure the `WHISPER_FINETUNED_MODEL_PATH` in the `.env` file to point to this fine-tuned model directory or Hugging Face Hub location for the `VoiceAgent` to load.
