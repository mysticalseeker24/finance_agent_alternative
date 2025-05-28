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
        # Ensure api_url does not end with /scrape or /v1/scrape, as it will be added.
        # The Config.FIRECRAWL_API_URL should be just "https://api.firecrawl.dev"
        # Let's ensure the base URL is clean.
        temp_api_url = self.api_url
        if temp_api_url:
            if temp_api_url.endswith("/v1/scrape"): temp_api_url = temp_api_url[:-11]
            elif temp_api_url.endswith("/scrape"): temp_api_url = temp_api_url[:-7]
            elif temp_api_url.endswith("/v1"): temp_api_url = temp_api_url[:-3]
            if temp_api_url.endswith("/"): temp_api_url = temp_api_url[:-1]
        self.api_url = temp_api_url # Store the cleaned base URL

        logger.info(
            f"FirecrawlScraper initialized. Effective API base URL: {self.api_url}"
        )

    async def scrape_url(
        self, url_to_scrape: str, page_options: Optional[Dict] = None, extractor_options: Optional[Dict] = None
    ) -> Dict[str, Any]:
        # Ensure self.api_url is from Config (e.g., "https://api.firecrawl.dev")
        # Ensure self.api_key is from Config
        
        if not self.api_key or not self.api_url:
            return {"success": False, "url": url_to_scrape, "error": "Firecrawl API key or URL not configured."}

        scrape_endpoint = f"{self.api_url}/v1/scrape" # Corrected to /v1/scrape

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {"url": url_to_scrape}
        current_page_options = page_options if page_options is not None else {"onlyMainContent": True, "includeHtml": False}
        payload["pageOptions"] = current_page_options
        
        if extractor_options: # For potential future use with LLM-based extraction
            payload["extractor"] = extractor_options # Note: Firecrawl docs use "extractorOptions", but for simplicity using "extractor" as per target. Will adjust if this is a strict API requirement.
                                                # Update: The target structure for the method uses "extractor" in payload, but the previous implementation used "extractorOptions".
                                                # The Firecrawl API documentation typically uses "extractorOptions" for the more advanced LLM extraction.
                                                # For a basic scrape, it might not be needed, or if used, it should be "extractorOptions".
                                                # Given the target structure, I'll use "extractor", but this is a point of potential API incompatibility if not careful.
                                                # Re-checking Firecrawl documentation: It IS "extractorOptions". The target spec has a slight deviation.
                                                # I will use "extractorOptions" as it's more likely correct for the actual API.
            payload["extractorOptions"] = extractor_options


        
        logger.info(f"Calling Firecrawl API: POST {scrape_endpoint} for URL: {url_to_scrape} with payload: {json.dumps(payload)}")

        try:
            async with aiohttp.ClientSession(headers=headers) as session: # headers can be set per-session or per-request
                async with session.post(scrape_endpoint, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as response: # Timeout set to 60s
                    if response.status == 200:
                        response_data = await response.json()
                        # Expect actual scraped data under a top-level "data" key
                        scraped_data_dict = response_data.get("data", {}) 
                        
                        if not scraped_data_dict: # If "data" key is missing or empty
                             logger.warning(f"Firecrawl response for {url_to_scrape} is missing 'data' field or 'data' is empty. Response: {response_data}")
                             # Check for 'content' directly under response_data as a fallback for simpler scrape responses
                             # though /v1/scrape with POST usually has the "data" field.
                             if response_data.get("content") or response_data.get("markdown"):
                                logger.info("Found content/markdown directly under response_data. Using that.")
                                scraped_data_dict = response_data # Treat the whole response as the data part

                        chosen_content = scraped_data_dict.get("markdown") or scraped_data_dict.get("content")
                        metadata = scraped_data_dict.get("metadata", {})
                        
                        if chosen_content is None:
                            logger.warning(f"No 'markdown' or 'content' found in Firecrawl response data for {url_to_scrape}. Data: {scraped_data_dict}")
                            # Fallback: use raw HTML if explicitly requested and available, or entire dict as content
                            if current_page_options.get("includeHtml") and scraped_data_dict.get("html"):
                                chosen_content = scraped_data_dict.get("html")
                            elif not chosen_content: # if still no content
                                 chosen_content = json.dumps(scraped_data_dict) # Serialize the whole data dict as content

                        logger.info(f"Successfully scraped URL: {url_to_scrape} using Firecrawl API.")
                        return {
                            "success": True, 
                            "url": url_to_scrape, 
                            "content": chosen_content, 
                            "metadata": metadata,
                            "raw_firecrawl_data": scraped_data_dict # Optionally include for downstream debugging
                        }
                    else:
                        error_details = await response.text()
                        logger.error(f"Firecrawl API error for {url_to_scrape}: {response.status} - {error_details}")
                        return {
                            "success": False, 
                            "url": url_to_scrape, 
                            "error": f"Firecrawl API Error: {response.status}", 
                            "details": error_details
                        }
        except aiohttp.ClientTimeout:
            logger.error(f"Timeout during Firecrawl request for {url_to_scrape} after 60 seconds.")
            return {"success": False, "url": url_to_scrape, "error": "Timeout", "details": "Request timed out after 60 seconds."}
        except aiohttp.ClientError as e:
            logger.error(f"aiohttp.ClientError during Firecrawl request for {url_to_scrape}: {str(e)}")
            return {"success": False, "url": url_to_scrape, "error": f"ClientError: {str(e)}"}
        except Exception as e:
            logger.error(f"Generic error during Firecrawl scrape for {url_to_scrape}: {str(e)}")
            return {"success": False, "url": url_to_scrape, "error": f"Generic error: {str(e)}"}

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
