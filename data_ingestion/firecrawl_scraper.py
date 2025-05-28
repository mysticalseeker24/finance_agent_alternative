"""Firecrawl integration for scraping web content using Firecrawl's public REST API."""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio # Added for potential concurrent calls in future, though not strictly used by aiohttp per call

import aiohttp # Added
from loguru import logger

from config import Config


class FirecrawlScraper:
    """Scraper using Firecrawl's public REST API for web content."""

    def __init__(self, api_url: Optional[str] = None, api_key: Optional[str] = None):
        """Initialize the Firecrawl scraper.

        Args:
            api_url: The Firecrawl API URL (defaults to Config.FIRECRAWL_API_URL).
            api_key: The Firecrawl API key (defaults to Config.FIRECRAWL_API_KEY).
        """
        self.api_url = api_url or Config.FIRECRAWL_API_URL
        self.api_key = api_key or Config.FIRECRAWL_API_KEY
        if not self.api_key:
            logger.error("Firecrawl API key not configured. FirecrawlScraper will not work.")
        if not self.api_url: # Should default in Config, but good to check
             logger.error("Firecrawl API URL not configured. FirecrawlScraper will not work.")
        # Ensure api_url does not end with /scrape, as it will be added.
        if self.api_url and self.api_url.endswith("/scrape"):
            self.api_url = self.api_url[:-7] # Remove /scrape
        if self.api_url and self.api_url.endswith("/"):
            self.api_url = self.api_url[:-1] # Remove trailing slash


        logger.info(
            f"FirecrawlScraper initialized to use API endpoint: {self.api_url}"
        )

    async def scrape_url(
        self, 
        url_to_scrape: str, 
        page_options: Optional[Dict] = None, 
        extractor_options: Optional[Dict] = None,
        timeout_seconds: int = 30 # Added timeout
    ) -> Dict[str, Any]:
        """Scrape content from a URL using Firecrawl's /scrape endpoint.

        Args:
            url_to_scrape: The URL to scrape.
            page_options: Options for page loading (e.g., screenshot, headers).
                          Defaults to {"onlyMainContent": True, "includeHtml": False}.
            extractor_options: Options for LLM-based extraction (e.g., schema, model).
                               Defaults to None.
            timeout_seconds: Timeout for the API request.

        Returns:
            A dictionary containing the scraped content or an error.
        """
        if not self.api_key or not self.api_url:
            return {"success": False, "url": url_to_scrape, "error": "Firecrawl API key or URL not configured."}

        scrape_endpoint = f"{self.api_url}/scrape"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {"url": url_to_scrape}
        payload["pageOptions"] = page_options if page_options is not None else {"onlyMainContent": True, "includeHtml": False}
        
        if extractor_options: # Only include if provided and not None/empty
            payload["extractorOptions"] = extractor_options # Corrected key from "extractor" to "extractorOptions"

        logger.info(f"Requesting Firecrawl scrape for URL: {url_to_scrape} with payload: {payload}")

        try:
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.post(scrape_endpoint, json=payload, timeout=timeout_seconds) as response:
                    if response.status == 200:
                        response_data = await response.json()
                        # Firecrawl's /scrape response structure: {"data": {"content": ..., "markdown": ..., "metadata": ...}}
                        # or sometimes directly {"content": ..., "markdown": ...} if success is implied by 200
                        
                        data_content = response_data.get("data", response_data) # Handle both structures

                        chosen_content = data_content.get("markdown", data_content.get("content", ""))
                        metadata = data_content.get("metadata", {})
                        
                        logger.info(f"Successfully scraped URL: {url_to_scrape}")
                        return {
                            "success": True,
                            "url": url_to_scrape,
                            "content": chosen_content,
                            "metadata": metadata, # Includes title, description, etc.
                            "raw_data": data_content # Include for potential downstream use
                        }
                    else:
                        error_details = await response.text()
                        logger.error(
                            f"Firecrawl API error for {url_to_scrape}: {response.status} - {error_details}"
                        )
                        return {
                            "success": False,
                            "url": url_to_scrape,
                            "error": f"Firecrawl API Error: {response.status}",
                            "details": error_details,
                        }
        except aiohttp.ClientError as e:
            logger.error(f"aiohttp.ClientError scraping {url_to_scrape}: {str(e)}")
            return {
                "success": False,
                "url": url_to_scrape,
                "error": "Client/Network Error",
                "details": str(e),
            }
        except asyncio.TimeoutError: # Specifically catch timeout
            logger.error(f"Timeout scraping {url_to_scrape} after {timeout_seconds} seconds.")
            return {
                "success": False,
                "url": url_to_scrape,
                "error": "Timeout",
                "details": f"Scraping timed out after {timeout_seconds} seconds.",
            }
        except Exception as e:
            logger.error(f"Unexpected error scraping {url_to_scrape}: {str(e)}")
            return {
                "success": False,
                "url": url_to_scrape,
                "error": "Unexpected Error",
                "details": str(e),
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


if __name__ == "__main__":
    # Example usage (requires running an asyncio event loop)
    async def main():
        scraper = FirecrawlScraper()

        # Test scrape_url
        # test_url = "https://www.cnbc.com/2024/03/10/stock-market-today-live-updates.html" # Example article
        # test_url = "https://finance.yahoo.com/news/"
        # print(f"\n--- Scraping URL: {test_url} ---")
        # # Example with custom pageOptions (e.g., to get HTML for parsing links)
        # # page_options_custom = {"includeHtml": True, "onlyMainContent": False} 
        # # url_data = await scraper.scrape_url(test_url, page_options=page_options_custom)
        # url_data = await scraper.scrape_url(test_url)
        # if url_data.get("success"):
        #     print(f"Content (first 500 chars): {url_data.get('content', '')[:500]}...")
        #     print(f"Metadata: {url_data.get('metadata')}")
        # else:
        #     print(f"Error scraping {test_url}: {url_data.get('error')} - {url_data.get('details')}")

        # Test scrape_financial_news
        print("\n--- Scraping Financial News Main Pages ---")
        news_results = await scraper.scrape_financial_news(["yahoo_finance", "bloomberg_markets"])
        for news_item in news_results:
            if news_item.get("error"):
                print(f"Error for {news_item['source_name']} ({news_item['url']}): {news_item['error']}")
            else:
                print(f"Source: {news_item['source_name']}, URL: {news_item['url']}")
                # print(f"  Content (first 200 chars): {news_item.get('content', '')[:200]}...")
                if news_item.get("raw_html"):
                     print(f"  HTML (first 300 chars): {news_item.get('raw_html', '')[:300]}...")
                else:
                     print(f"  Content (first 200 chars): {news_item.get('content', '')[:200]}...")
                print(f"  Metadata: {news_item.get('metadata')}")
            print("---")

        # Test scrape_earnings_report
        # test_ticker = "AAPL"
        # print(f"\n--- Scraping Earnings Report for {test_ticker} ---")
        # earnings_data = await scraper.scrape_earnings_report(test_ticker)
        # if earnings_data.get("error"):
        #     print(f"Error for {test_ticker}: {earnings_data['error']}")
        # else:
        #     print(f"Ticker: {earnings_data['ticker']}, URL: {earnings_data['url']}")
        #     print(f"  Content (first 500 chars): {earnings_data.get('content', '')[:500]}...")
        #     print(f"  Metadata: {earnings_data.get('metadata')}")
        # print("---")

    # Run the main async function
    # asyncio.run(main())
    # If running in an environment where asyncio loop is already running (e.g. Jupyter notebook with !python script.py)
    # then use:
    # loop = asyncio.get_event_loop()
    # if loop.is_running():
    #     loop.create_task(main())
    # else:
    #     asyncio.run(main())
    pass # Comment out pass and uncomment loop logic to run
