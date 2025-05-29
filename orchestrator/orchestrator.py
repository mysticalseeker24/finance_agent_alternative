"""Orchestrator for coordinating between specialized agents."""

from typing import Dict, List, Any, Optional, Union
import asyncio
from datetime import datetime
import json
import hashlib  # Added
import redis  # Added

from fastapi import HTTPException
from loguru import logger

from agents.api_agent import APIAgent
from agents.scraping_agent import ScrapingAgent
from agents.retriever_agent import RetrieverAgent
from agents.analysis_agent import AnalysisAgent
from agents.language_agent import LanguageAgent
from agents.voice_agent import VoiceAgent
from config import Config


class Orchestrator:
    """Orchestrator for coordinating between specialized agents."""

    def __init__(self):
        """Initialize the orchestrator and all agents."""
        # Initialize agents
        self.api_agent = APIAgent()
        self.scraping_agent = ScrapingAgent()
        self.retriever_agent = RetrieverAgent()
        self.analysis_agent = AnalysisAgent()
        self.language_agent = LanguageAgent()
        self.voice_agent = VoiceAgent()

        # Initialize agent status tracking
        self.agent_status = {
            "api_agent": "initialized",
            "scraping_agent": "initialized",
            "retriever_agent": "initialized",
            "analysis_agent": "initialized",
            "language_agent": "initialized",
            "voice_agent": "initialized",
        }

        # Initialize Redis cache for LLM responses
        try:
            self.cache = redis.Redis(
                host=Config.REDIS_HOST,
                port=Config.REDIS_PORT,
                decode_responses=True,  # Store strings for JSON
            )
            self.cache.ping()  # Check connection
            logger.info(
                f"Orchestrator connected to Redis cache at {Config.REDIS_HOST}:{Config.REDIS_PORT}"
            )
        except redis.exceptions.ConnectionError as e:
            logger.warning(
                f"Orchestrator failed to connect to Redis cache: {e}. LLM response caching will be disabled."
            )
            self.cache = None

        logger.info("Orchestrator initialized with all agents")

    async def initialize_agents(self) -> Dict[str, str]:
        """Initialize all agents that require explicit initialization.

        Returns:
            Status of each agent initialization.
        """
        # Some agents require explicit initialization
        initialization_tasks = [
            self._initialize_agent(
                "retriever_agent", self.retriever_agent.initialize()
            ),
            self._initialize_agent(
                "language_agent", self.language_agent.initialize_llm()
            ),
            self._initialize_agent(
                "voice_agent", self.voice_agent.initialize_stt_model()
            ),
        ]

        # Run initializations concurrently
        await asyncio.gather(*initialization_tasks)

        return self.agent_status

    async def _initialize_agent(self, agent_name: str, init_coroutine) -> None:
        """Initialize a specific agent and update its status.

        Args:
            agent_name: Name of the agent to initialize.
            init_coroutine: Coroutine that initializes the agent.
        """
        try:
            success = await init_coroutine
            self.agent_status[agent_name] = "ready" if success else "failed"
            logger.info(
                f"Agent {agent_name} initialization: {self.agent_status[agent_name]}"
            )
        except Exception as e:
            self.agent_status[agent_name] = "error"
            logger.error(f"Error initializing {agent_name}: {str(e)}")

    async def process_query(
        self, query: str, voice_input: bool = False, voice_output: bool = True
    ) -> Dict[str, Any]:
        """Process a financial query from start to finish.

        This method orchestrates the entire query processing pipeline:
        1. Convert speech to text (if voice_input is True)
        2. Retrieve relevant context
        3. Analyze the query and context
        4. Generate a response
        5. Convert text to speech (if voice_output is True)

        Args:
            query: The user's query (text or audio file path).
            voice_input: Whether the query is provided as voice.
            voice_output: Whether to generate voice output.

        Returns:
            The processed result with response text and optional audio.
        """
        start_time = datetime.now()
        query_text = query
        stt_result = None
        cache_key = None  # Initialize cache_key

        try:
            # Step 1: Handle speech input if provided
            if voice_input:
                # Assume query contains path to audio file
                logger.info(f"Processing voice input: {query}")
                stt_response = await self.voice_agent.speech_to_text(audio_path=query)

                if "error" in stt_response:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Speech-to-text error: {stt_response['error']}",
                    )

                stt_result = stt_response.get("data", {})
                query_text = stt_result.get("text", "")
                logger.info(f"Transcribed query: {query_text}")

            # Check cache for existing LLM response
            if self.cache:
                query_hash = hashlib.sha256(query_text.encode()).hexdigest()
                # Consider adding other relevant params like language to the hash if they significantly alter the response
                # For now, just using query_text
                cache_key = f"llm_response:{query_hash}"
                cached_response_str = await asyncio.to_thread(self.cache.get, cache_key)
                if cached_response_str:
                    logger.info(
                        f"Returning cached LLM response for query: {query_text[:50]}..."
                    )
                    cached_result = json.loads(cached_response_str)
                    # Update dynamic fields like processing_time and timestamp
                    cached_result["processing_time"] = (
                        datetime.now() - start_time
                    ).total_seconds()
                    cached_result["timestamp"] = (
                        datetime.now().isoformat()
                    )  # Or use cached timestamp if preferred
                    return cached_result

            # Step 2: Retrieve relevant context using RAG
            logger.info(f"Retrieving context for query: {query_text}")
            context = await self._retrieve_context(query_text)

            # Step 3: Gather necessary data based on the query type
            logger.info("Gathering data for query analysis")
            data = await self._gather_data_for_query(query_text, context)

            # Step 4: Generate a response using the Language Agent
            logger.info("Generating response")
            response_result = await self.language_agent.generate_query_response(
                query=query_text,
                context=context,
                portfolio_data=data.get("portfolio_data", {}),
            )

            if "error" in response_result:
                raise HTTPException(
                    status_code=500,
                    detail=f"Response generation error: {response_result['error']}",
                )

            response_data = response_result.get("data", {})
            response_text = response_data.get("response", "")

            # Step 5: Convert text to speech if requested
            audio_url = None
            if voice_output and response_text:
                logger.info("Converting response to speech")
                tts_response = await self.voice_agent.text_to_speech(text=response_text)

                if "error" not in tts_response:
                    tts_data = tts_response.get("data", {})
                    audio_url = tts_data.get("audio_url")

            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()

            # Prepare the final result
            result = {
                "query": query_text,
                "response": response_text,
                "audio_url": audio_url,
                "sources": response_data.get("sources", []),
                "processing_time": processing_time,
            }

            # Add speech recognition info if applicable
            if stt_result:
                result["speech_recognition"] = {
                    "confidence": stt_result.get("confidence"),
                    "language": stt_result.get("language"),
                }

            # Cache the successful result
            if (
                self.cache and cache_key
            ):  # cache_key would have been set if cache is available
                try:
                    # Ensure result is JSON serializable (it should be as it's FastAPI response)
                    response_to_cache_str = json.dumps(result)
                    await asyncio.to_thread(
                        self.cache.set,
                        cache_key,
                        response_to_cache_str,
                        ex=60 * 60 * 12,
                    )  # Cache for 12 hours
                    logger.info(f"Cached LLM response for query: {query_text[:50]}...")
                except Exception as cache_e:
                    logger.warning(
                        f"Failed to cache LLM response for query '{query_text[:50]}...': {cache_e}"
                    )

            logger.info(f"Query processed in {processing_time:.2f} seconds")
            return result

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def generate_market_brief(self) -> Dict[str, Any]:
        """Generate a comprehensive market brief.

        This method orchestrates the entire market brief generation pipeline:
        1. Fetch market data
        2. Scrape news
        3. Analyze portfolio data
        4. Generate the brief
        5. Convert to speech

        Returns:
            The generated market brief with text and audio.
        """
        start_time = datetime.now()

        try:
            # Step 1: Fetch market data
            logger.info("Fetching market data")
            market_data_response = await self.api_agent.get_market_summary()
            market_data = market_data_response.get("data", {})

            # Step 2: Scrape financial news
            logger.info("Scraping financial news")
            news_response = await self.scraping_agent.scrape_market_news()
            news_data = news_response.get("data", [])

            # Step 3: Get and analyze portfolio data
            # This would typically come from a database or user configuration
            # Using the default portfolio from Config
            portfolio = Config.DEFAULT_PORTFOLIO
            if not portfolio:
                logger.warning(
                    "Config.DEFAULT_PORTFOLIO is empty. Portfolio analysis will be based on an empty portfolio."
                )

            logger.info("Analyzing portfolio data")
            portfolio_response = await self.analysis_agent.analyze_portfolio(portfolio)
            portfolio_data = portfolio_response.get("data", {})

            # Step 4: Generate the market brief
            logger.info("Generating market brief")
            brief_response = await self.language_agent.generate_market_brief(
                market_data=market_data,
                portfolio_data=portfolio_data,
                news_data=news_data,
            )

            if "error" in brief_response:
                raise HTTPException(
                    status_code=500,
                    detail=f"Brief generation error: {brief_response['error']}",
                )

            brief_data = brief_response.get("data", {})
            brief_text = brief_data.get("full_text", "")

            # Step 5: Convert to speech
            logger.info("Converting brief to speech")
            tts_response = await self.voice_agent.text_to_speech(text=brief_text)

            audio_url = None
            if "error" not in tts_response:
                tts_data = tts_response.get("data", {})
                audio_url = tts_data.get("audio_url")

            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()

            # Prepare the final result
            result = {
                "title": brief_data.get("title", "Daily Market Brief"),
                "date": brief_data.get("date", datetime.now().strftime("%Y-%m-%d")),
                "summary": brief_data.get("summary", ""),
                "full_text": brief_text,
                "audio_url": audio_url,
                "sections": {
                    "market_overview": brief_data.get("market_overview", ""),
                    "portfolio_performance": brief_data.get(
                        "portfolio_performance", ""
                    ),
                    "news_highlights": brief_data.get("news_highlights", ""),
                    "outlook": brief_data.get("outlook", ""),
                },
                "processing_time": processing_time,
            }

            logger.info(f"Market brief generated in {processing_time:.2f} seconds")
            return result

        except Exception as e:
            logger.error(f"Error generating market brief: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def _retrieve_context(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve relevant context for a query using RAG.

        Args:
            query: The user's query.

        Returns:
            List of relevant context items.
        """
        try:
            # Search for relevant content
            search_response = await self.retriever_agent.search_content(
                query=query, top_k=5
            )

            if "error" in search_response:
                logger.warning(f"Error in retrieval: {search_response['error']}")
                return []  # Return empty context on error

            # Extract the search results
            results = search_response.get("data", {}).get("data", [])

            # If no results or poor quality results, try query reformulation
            # Check if results is a dictionary (error from agent) or list of dicts (actual results)
            if isinstance(results, dict) and "error" in results:
                logger.warning(f"Error in initial retrieval: {results['error']}")
                results = []  # Treat as no results

            # Ensure results is a list before trying to access scores
            if not isinstance(results, list):
                logger.error(
                    f"Unexpected format for retrieval results: {results}. Defaulting to empty list."
                )
                results = []

            # Calculate max score safely
            scores = [r.get("score", 0.0) for r in results if isinstance(r, dict)]
            max_score = max(scores) if scores else 0.0

            if not results or max_score < 0.7:  # Confidence threshold
                logger.info(
                    f"Low quality initial retrieval results (max score: {max_score:.2f}). Attempting query reformulation."
                )
                # Create a brief summary of why reformulation is needed
                reformulation_reason = (
                    "Initial search results were not specific or relevant enough."
                )
                if results:  # If there were some results, try to use them for context
                    top_results_summary_parts = []
                    for res in results[:2]:  # Take top 2 results
                        if (
                            isinstance(res, dict)
                            and "metadata" in res
                            and isinstance(res["metadata"], dict)
                        ):
                            text_preview = res["metadata"].get("text", "")[
                                :100
                            ]  # Get first 100 chars of text
                            if text_preview:
                                top_results_summary_parts.append(text_preview)
                    if top_results_summary_parts:
                        reformulation_reason = f"Initial search results (e.g., '{'; '.join(top_results_summary_parts)}...') were not specific or relevant enough."

                original_query_text = query
                reformulated_query = await self.language_agent.reformulate_query(
                    original_query_text, search_summary=reformulation_reason
                )

                if (
                    reformulated_query.lower().strip()
                    != original_query_text.lower().strip()
                ):
                    logger.info(
                        f"Retrying search with reformulated query: {reformulated_query}"
                    )
                    search_response_reformulated = (
                        await self.retriever_agent.search_content(
                            query=reformulated_query,
                            top_k=5,  # Or use original top_k if it was a parameter
                        )
                    )

                    new_results_data = search_response_reformulated.get("data", {})
                    new_results = new_results_data.get(
                        "data", []
                    )  # Actual list of results

                    if (
                        isinstance(new_results, dict) and "error" in new_results
                    ):  # Check for error in new_results dict itself
                        logger.warning(
                            f"Error in reformulated retrieval: {new_results['error']}"
                        )
                    elif not isinstance(
                        new_results, list
                    ):  # Check type of actual results list
                        logger.error(
                            f"Unexpected format for reformulated retrieval results: {new_results}. Using original results."
                        )
                    else:
                        new_scores = [
                            r.get("score", 0.0)
                            for r in new_results
                            if isinstance(r, dict)
                        ]
                        new_max_score = max(new_scores) if new_scores else 0.0
                        logger.info(
                            f"Reformulated search results max score: {new_max_score:.2f}"
                        )
                        # Simple strategy: use reformulated results if they exist, otherwise stick with original (even if poor)
                        if new_results:  # If reformulated search returned any results
                            results = new_results
                else:
                    logger.info(
                        "Query reformulation did not result in a new query. Using original results."
                    )
            return results

        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return []  # Return empty context on error

    async def _gather_data_for_query(
        self, query: str, context: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Gather necessary data based on the query type.

        Args:
            query: The user's query.
            context: Retrieved context for the query.

        Returns:
            Gathered data for query processing.
        """
        gathered_data = {}

        # Determine what type of data is needed based on the query
        query_lower = query.lower()

        # Check for portfolio-related queries
        if any(
            term in query_lower
            for term in ["portfolio", "holdings", "positions", "stocks", "investment"]
        ):
            # This would typically come from a database or user configuration
            # Using the default portfolio from Config
            portfolio = Config.DEFAULT_PORTFOLIO
            if not portfolio:
                logger.warning(
                    "Config.DEFAULT_PORTFOLIO is empty for query context. Portfolio analysis will be based on an empty portfolio."
                )

            portfolio_response = await self.analysis_agent.analyze_portfolio(portfolio)
            gathered_data["portfolio_data"] = portfolio_response.get("data", {})

        # Check for market-related queries
        if any(
            term in query_lower for term in ["market", "index", "s&p", "dow", "nasdaq"]
        ):
            market_response = await self.api_agent.get_market_summary()
            gathered_data["market_data"] = market_response.get("data", {})

        # Check for news-related queries
        if any(
            term in query_lower
            for term in ["news", "headline", "announcement", "report"]
        ):
            news_response = await self.scraping_agent.scrape_market_news()
            gathered_data["news_data"] = news_response.get("data", [])

        # Check for stock-specific queries
        stock_tickers = self._extract_stock_tickers(query)
        if stock_tickers:
            stock_data = {}
            for ticker in stock_tickers:
                stock_response = await self.api_agent.get_stock_info(ticker)
                stock_data[ticker] = stock_response.get("data", {})
            gathered_data["stock_data"] = stock_data

        # Check for risk-related queries
        if any(
            term in query_lower for term in ["risk", "exposure", "volatility", "beta"]
        ):
            # This would typically use historical data for risk calculations
            # Use Config.DEFAULT_PORTFOLIO if portfolio data hasn't been loaded yet
            # Or if the query is specifically about generic portfolio risk without prior portfolio context
            current_portfolio_for_risk = (
                portfolio
                if "portfolio_data" in gathered_data and portfolio
                else Config.DEFAULT_PORTFOLIO
            )
            if not current_portfolio_for_risk:
                logger.warning(
                    "Config.DEFAULT_PORTFOLIO is empty for risk calculation. Risk analysis might be limited."
                )

            if current_portfolio_for_risk:  # Proceed if there's any portfolio
                risk_response = await self.analysis_agent.calculate_risk(
                    tickers=[
                        item["ticker"]
                        for item in current_portfolio_for_risk
                        if "ticker" in item
                    ]  # Ensure item has 'ticker'
                )
                gathered_data["risk_data"] = risk_response.get("data", {})
            else:  # No portfolio to analyze for risk
                gathered_data["risk_data"] = {
                    "error": "No portfolio data available to calculate risk."
                }

        # Check for specific regional focus
        regions = {
            "asia": ["TSMC", "9988.HK", "005930.KS"],  # TSMC, Alibaba, Samsung
            "europe": ["ASML", "SIE.DE", "UL"],  # ASML, Siemens, Unilever
            "us": ["AAPL", "MSFT", "AMZN"],  # Apple, Microsoft, Amazon
        }

        for region, tickers in regions.items():
            if region in query_lower:
                regional_data = {}
                for ticker in tickers:
                    stock_response = await self.api_agent.get_stock_info(ticker)
                    regional_data[ticker] = stock_response.get("data", {})
                gathered_data[f"{region}_data"] = regional_data

        return gathered_data

    def _extract_stock_tickers(self, query: str) -> List[str]:
        """Extract stock tickers from a query.

        Args:
            query: The user's query.

        Returns:
            List of stock tickers found in the query.
        """
        # This is a very simplistic implementation
        # In a real implementation, you would use more sophisticated NER
        common_tickers = {
            "AAPL",
            "MSFT",
            "AMZN",
            "GOOGL",
            "BRK.B",
            "JNJ",
            "JPM",
            "V",
            "PG",
            "UNH",
            "TSMC",
            "ASML",
            "NVDA",
            "TSM",
            "INTC",
            "AMD",
            "FB",
            "META",
            "NFLX",
            "BABA",
        }

        words = query.upper().split()
        found_tickers = [
            word.strip(",.?!()")
            for word in words
            if word.strip(",.?!()") in common_tickers
        ]

        return found_tickers
