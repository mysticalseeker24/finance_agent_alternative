"""Firecrawl integration for scraping web content using the firecrawl-py SDK."""

import json # Keep for potential use, though SDK might reduce direct JSON handling
from typing import Dict, List, Any, Optional
from datetime import datetime # Keep for potential use
import asyncio

from firecrawl import FirecrawlApp # Added
from loguru import logger

from config import Config


class FirecrawlScraper:
    """Scraper using the firecrawl-py SDK for web content."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Firecrawl scraper using the SDK.

        Args:
            api_key: The Firecrawl API key (defaults to Config.FIRECRAWL_API_KEY).
        """
        self.api_key = api_key or Config.FIRECRAWL_API_KEY
        self.app = None # Initialize app attribute

        if not self.api_key:
            logger.error("Firecrawl API key not configured. FirecrawlScraper will not work.")
        else:
            try:
                self.app = FirecrawlApp(api_key=self.api_key)
                logger.info("FirecrawlApp SDK initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize FirecrawlApp SDK: {str(e)}")
                self.app = None # Ensure app is None if initialization fails

    async def scrape_url(
        self, 
        url_to_scrape: str, 
        page_options: Optional[Dict] = None, 
        extractor_options: Optional[Dict] = None,
        timeout_seconds: int = 30
    ) -> Dict[str, Any]:
        """Scrape content from a URL using the Firecrawl SDK.

        Args:
            url_to_scrape: The URL to scrape.
            page_options: Options for page loading (e.g., screenshot, headers).
                          Defaults to {"onlyMainContent": True, "includeHtml": False}.
            extractor_options: Options for LLM-based extraction (e.g., schema, model).
                               Defaults to None.
            timeout_seconds: Timeout for the API request in seconds.

        Returns:
            A dictionary containing the scraped content or an error.
        """
        if not self.app:
            return {"success": False, "url": url_to_scrape, "error": "FirecrawlApp not initialized due to missing API key or initialization failure."}

        current_page_options = page_options if page_options is not None else {"onlyMainContent": True, "includeHtml": False}
        
        logger.info(f"Requesting Firecrawl SDK scrape for URL: {url_to_scrape} with page_options: {current_page_options}, extractor_options: {extractor_options}")

        try:
            # The firecrawl-py SDK's scrape_url method is synchronous.
            # We run it in a thread to avoid blocking the asyncio event loop.
            sdk_response = await asyncio.to_thread(
                self.app.scrape_url,
                url=url_to_scrape,
                page_options=current_page_options,
                extractor_options=extractor_options if extractor_options else None,
                timeout=timeout_seconds * 1000  # Convert seconds to milliseconds for SDK
            )
            
            # The SDK returns a Pydantic model-like object or dict. 
            # It behaves like a dict, so direct .get() access should work.
            # If it were a Pydantic model and we needed a true dict:
            # sdk_response_dict = sdk_response.dict() if hasattr(sdk_response, 'dict') else sdk_response

            # Assuming sdk_response is directly usable as a dict:
            chosen_content = sdk_response.get("markdown") or sdk_response.get("content", "")
            metadata = sdk_response.get("metadata", {})
            
            logger.info(f"Successfully scraped URL via SDK: {url_to_scrape}")
            return {
                "success": True,
                "url": url_to_scrape,
                "content": chosen_content,
                "metadata": metadata,
                "raw_data": sdk_response # Store the entire SDK response
            }
        # It's good practice to catch more specific exceptions if known for the SDK.
        # For now, a general Exception is used as per the prompt's example.
        # Common exceptions from HTTP clients include connection errors, timeouts.
        # The FirecrawlApp might raise its own specific exceptions.
        except asyncio.TimeoutError: # This catches timeout for asyncio.to_thread itself
            logger.error(f"asyncio.to_thread call timed out scraping {url_to_scrape} after {timeout_seconds} seconds.")
            return {
                "success": False,
                "url": url_to_scrape,
                "error": "Operation Timeout", # Distinguish from SDK's internal timeout
                "details": f"Scraping operation timed out after {timeout_seconds} seconds.",
            }
        except Exception as e: 
            logger.error(f"Firecrawl SDK error scraping {url_to_scrape}: {type(e).__name__} - {str(e)}")
            # Check if the exception 'e' has a 'response' attribute for more details (like status code)
            details = str(e)
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_content = e.response.json() # Or e.response.text()
                    details = f"{str(e)} - Response: {error_content}"
                except ValueError: # If response is not JSON
                    details = f"{str(e)} - Response: {e.response.text}"
            
            return {
                "success": False,
                "url": url_to_scrape,
                "error": f"Firecrawl SDK Error: {type(e).__name__}",
                "details": details,
            }

    async def scrape_financial_news(
        self, sources: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Scrape financial news from main pages of multiple sources.

        Args:
            sources: List of news source identifiers (e.g., ["yahoo_finance", "cnbc"]).
                     If None, default sources are used.

        Returns:
            A list of dictionaries, each containing the source name, URL, 
            and scraped content of the main news page.
        """
        available_sources = {
            "yahoo_finance": "https://finance.yahoo.com/news/",
            "cnbc_finance": "https://www.cnbc.com/finance/", # Renamed to avoid conflict if key was just 'cnbc'
            "bloomberg_markets": "https://www.bloomberg.com/markets",
            # Add more sources as needed
        }

        sources_to_scrape_map = {}
        if sources:
            for source_key in sources:
                if source_key in available_sources:
                    sources_to_scrape_map[source_key] = available_sources[source_key]
                else:
                    logger.warning(f"Unknown news source key: {source_key}. Skipping.")
        else: # Default to all available sources
            sources_to_scrape_map = available_sources
        
        results = []
        
        # Page options tailored for news landing pages - might want full content to find links
        page_options_for_news_landing = {"onlyMainContent": False, "includeHtml": True} 

        for source_name, url in sources_to_scrape_map.items():
            logger.info(f"Scraping main news page for source: {source_name} from {url}")
            scrape_result = await self.scrape_url(url, page_options=page_options_for_news_landing)
            
            if scrape_result.get("success"):
                results.append({
                    "source_name": source_name,
                    "url": url,
                    "content": scrape_result.get("content"), # This will be markdown or text of the main page
                    "metadata": scrape_result.get("metadata", {}),
                    "raw_html": scrape_result.get("raw_data", {}).get("html") if page_options_for_news_landing.get("includeHtml") else None
                })
            else:
                logger.warning(f"Failed to scrape news source: {source_name} from {url}. Error: {scrape_result.get('error')}")
                results.append({
                    "source_name": source_name,
                    "url": url,
                    "error": scrape_result.get("error"),
                    "details": scrape_result.get("details")
                })
        
        logger.info(f"Completed scraping of {len(results)} financial news main pages.")
        return results

    async def scrape_earnings_report(self, ticker: str) -> Dict[str, Any]:
        """Scrape the main content of an earnings report page for a company.

        Args:
            ticker: The stock ticker symbol.

        Returns:
            A dictionary containing the ticker, URL, scraped content, metadata, and source.
        """
        # Construct URL for earnings information (example: Yahoo Finance)
        # This might need to be made more robust or configurable if different sources are used.
        earnings_page_url = f"https://finance.yahoo.com/quote/{ticker}/analysis" # Changed to /analysis for potentially richer text
        # earnings_page_url = f"https://finance.yahoo.com/quote/{ticker}/financials"
        # earnings_page_url = f"https://seekingalpha.com/symbol/{ticker}/earnings" # Example for another source
        
        logger.info(f"Scraping earnings report page for ticker: {ticker} from {earnings_page_url}")

        # Standard page options for content extraction
        page_options_for_earnings = {"onlyMainContent": True, "includeHtml": False}

        scrape_result = await self.scrape_url(earnings_page_url, page_options=page_options_for_earnings)

        if scrape_result.get("success"):
            return {
                "ticker": ticker,
                "url": earnings_page_url,
                "content": scrape_result.get("content"), # Markdown or text content
                "metadata": scrape_result.get("metadata", {}),
                "source": "Firecrawl" 
            }
        else:
            logger.warning(f"Failed to scrape earnings report for {ticker} from {earnings_page_url}. Error: {scrape_result.get('error')}")
            return {
                "ticker": ticker,
                "url": earnings_page_url,
                "error": scrape_result.get("error"),
                "details": scrape_result.get("details"),
                "source": "Firecrawl"
            }

    async def crawl_website(
        self, 
        url_to_crawl: str, 
        crawler_options: Optional[Dict] = None, # These are the 'params' for SDK's crawl_url
        timeout_seconds: int = 180 
    ) -> List[Dict[str, Any]]:
        """Crawl a website starting from a specific URL using the Firecrawl SDK.

        Args:
            url_to_crawl: The starting URL for the crawl.
            crawler_options: A dictionary of options to pass to the Firecrawl SDK's
                             `crawl_url` method via its `params` argument. This can include
                             options like:
                             - `includes`: List of URL patterns to include.
                             - `excludes`: List of URL patterns to exclude.
                             - `maxDepth`: Maximum crawl depth.
                             - `limit`: Maximum number of pages to crawl.
                             - `pageOptions`: Dictionary of page options for each page crawled 
                               (e.g., `onlyMainContent`, `includeHtml`).
                             - `extractorOptions`: Dictionary of extractor options for each page
                               (e.g., `extractionSchema`, `mode`).
                             - `generateTags`: Boolean, whether to generate AI-powered tags.
                             - `returnOnlyUrls`: Boolean, whether to return only URLs.
            timeout_seconds: Timeout for the entire crawl operation in seconds.

        Returns:
            A list of dictionaries. Each dictionary represents a successfully crawled page
            and contains keys like `success` (bool), `url` (str), `content` (str, markdown or text),
            `metadata` (dict), and `raw_data` (the raw SDK response for that page).
            If the crawl operation fails globally (e.g., SDK initialization error, top-level timeout),
            it returns a list containing a single error dictionary:
            `[{"success": False, "url": url_to_crawl, "error": "Error message", "details": "..."}]`.
        """
        if not self.app:
            return [{"success": False, "url": url_to_crawl, "error": "FirecrawlApp not initialized due to missing API key or initialization failure."}]

        logger.info(f"Requesting Firecrawl SDK crawl for URL: {url_to_crawl} with crawler_options: {crawler_options}")

        try:
            # SDK's crawl_url is synchronous
            crawl_results_sdk = await asyncio.to_thread(
                self.app.crawl_url,
                url=url_to_crawl,
                params=crawler_options, # Pass crawler_options as params
                timeout=timeout_seconds * 1000 # Convert seconds to milliseconds
            )
            
            processed_results = []
            if crawl_results_sdk: # Ensure it's not None or empty
                for item_data in crawl_results_sdk:
                    # Assuming item_data is a dict-like object
                    # Based on SDK documentation, the URL of the crawled page is in 'sourceURL'
                    page_url = item_data.get("sourceURL") or item_data.get("url") 
                    chosen_content = item_data.get("markdown") or item_data.get("content", "")
                    metadata = item_data.get("metadata", {})
                    
                    processed_results.append({
                        "success": True,
                        "url": page_url,
                        "content": chosen_content,
                        "metadata": metadata,
                        "raw_data": item_data 
                    })
            logger.info(f"Successfully crawled URL via SDK: {url_to_crawl}, found {len(processed_results)} pages.")
            return processed_results
            
        except asyncio.TimeoutError:
            logger.error(f"asyncio.to_thread call timed out crawling {url_to_crawl} after {timeout_seconds} seconds.")
            return [{"success": False, "url": url_to_crawl, "error": "Operation Timeout", "details": f"Crawl operation timed out after {timeout_seconds} seconds."}]
        except Exception as e:
            logger.error(f"Firecrawl SDK error crawling {url_to_crawl}: {type(e).__name__} - {str(e)}")
            details = str(e)
            # Attempt to get more details if the exception has a response attribute (e.g., from HTTP errors)
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_content = e.response.json()
                    details = f"{str(e)} - Response: {error_content}"
                except ValueError: # If response content is not JSON
                    details = f"{str(e)} - Response: {e.response.text}"
            return [{"success": False, "url": url_to_crawl, "error": f"Firecrawl SDK Crawl Error: {type(e).__name__}", "details": details}]
