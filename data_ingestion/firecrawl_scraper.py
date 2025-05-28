"""Firecrawl integration for scraping dynamic financial web content using MCP."""

import json
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

import requests
from loguru import logger
# Attempt to import MCP client, will be used if available
try:
    from mcp_client import mcp4_puppeteer_navigate, mcp4_puppeteer_evaluate
except ImportError:
    logger.warning(
        "mcp_client not found. FirecrawlScraper will operate in simulation mode only."
    )
    mcp4_puppeteer_navigate = None
    mcp4_puppeteer_evaluate = None

from config import Config


class FirecrawlScraper:
    """Scraper using Firecrawl MCP Server for dynamic web content."""

    def __init__(self, api_url: Optional[str] = None, api_key: Optional[str] = None):
        """Initialize the Firecrawl scraper using MCP.

        Args:
            api_url: The Firecrawl API URL (defaults to Config.FIRECRAWL_API_URL).
            api_key: The Firecrawl API key (defaults to Config.FIRECRAWL_API_KEY).
        """
        self.api_url = api_url or Config.FIRECRAWL_API_URL
        self.api_key = api_key or Config.FIRECRAWL_API_KEY
        self.mcp_server_name = Config.FIRECRAWL_MCP_SERVER
        logger.info(
            f"Firecrawl MCP scraper initialized with server: {self.mcp_server_name}"
        )

    def scrape_url(
        self, url: str, selectors: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Scrape content from a URL using Firecrawl MCP integration.

        Args:
            url: The URL to scrape.
            selectors: CSS selectors to extract specific content.
                       Example: {"title": "h1", "content": ".article-body"}

        Returns:
            A dictionary containing the scraped content.
        """
        try:
            logger.info(f"Scraping URL with Firecrawl MCP: {url}")

            # Prepare the puppeteer script for MCP
            script = {"navigate": {"url": url, "waitUntil": "networkidle0"}}

            # Add selector extraction if provided
            extraction_scripts = {}
            if selectors:
                for key, selector in selectors.items():
                    extraction_scripts[key] = (
                        f"document.querySelector('{selector}')?.textContent.trim()"
                    )

            # If selectors provided, add them to the script
            if extraction_scripts:
                script["extract"] = extraction_scripts
            else:
                # Default extraction if no selectors provided
                script["extract"] = {
                    "title": "document.title",
                    "content": "document.body.textContent.trim()",
                }

            # Call MCP Puppeteer to execute the script
            mcp_result = self._call_mcp_puppeteer(url, script)

            # Process the result
            if mcp_result and "error" not in mcp_result and mcp_result.get("success", True): # MCP success
                result = {
                    "url": url,
                    "timestamp": datetime.now().isoformat(),
                    **mcp_result,
                }
                logger.info(f"Successfully scraped content from {url} using Firecrawl MCP.")
            else: # MCP failed or returned an error structure
                logger.warning(
                    f"MCP scraping failed for {url}. Result: {mcp_result}. Falling back to simulation."
                )
                # Use selector keys for fallback if available, else pass selectors dict
                selector_keys = list(selectors.keys()) if selectors else []
                result = self._trigger_simulation_fallback(url, selector_keys, selectors)

            return result

        except Exception as e:
            logger.error(f"Error scraping {url} with Firecrawl: {str(e)}")
            return {
                "url": url,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def scrape_financial_news(
        self, sources: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Scrape financial news from multiple sources.

        Args:
            sources: List of news source identifiers (e.g., ["yahoo_finance", "cnbc"]).
                    If None, all available sources are used.

        Returns:
            A list of dictionaries containing news articles.
        """
        available_sources = {
            "yahoo_finance": "https://finance.yahoo.com/news/",
            "cnbc": "https://www.cnbc.com/finance/",
            "bloomberg": "https://www.bloomberg.com/markets",
        }

        sources_to_scrape = sources or list(available_sources.keys())
        results = []

        for source in sources_to_scrape:
            if source in available_sources:
                url = available_sources[source]
                selectors = {
                    "articles": "article, .js-stream-content",  # Generic selector for articles
                    "title": "h1, h2, h3",
                    "link": "a",
                    "summary": "p",
                }

                try:
                    result = self.scrape_url(url, selectors)
                    if "error" not in result:
                        results.extend(result.get("articles", []))
                    else:
                        logger.warning(f"Error scraping {source}: {result['error']}")

                except Exception as e:
                    logger.error(f"Error processing {source}: {str(e)}")
            else:
                logger.warning(f"Unknown source: {source}")

        return results

    def scrape_earnings_report(self, ticker: str) -> Dict[str, Any]:
        """Scrape the latest earnings report for a company.

        Args:
            ticker: The stock ticker symbol.

        Returns:
            A dictionary containing earnings data.
        """
        # Construct URL for earnings information
        url = f"https://finance.yahoo.com/quote/{ticker}/earnings"

        # Define selectors for earnings data
        selectors = {
            "earnings_table": "table",
            "earnings_date": ".earnings-date",
            "eps": ".eps-actual",
            "eps_estimate": ".eps-estimate",
            "revenue": ".revenue-actual",
            "revenue_estimate": ".revenue-estimate",
        }

        try:
            # Scrape the earnings page
            result = self.scrape_url(url, selectors)

            if "error" not in result:
                # Process the result
                return {
                    "ticker": ticker,
                    "earnings_data": result,
                    "source": url,
                    "timestamp": datetime.now().isoformat(),
                }
            else:
                logger.warning(
                    f"Error scraping earnings for {ticker}: {result['error']}"
                )
                return {
                    "ticker": ticker,
                    "error": result["error"],
                    "timestamp": datetime.now().isoformat(),
                }

        except Exception as e:
            logger.error(f"Error processing earnings for {ticker}: {str(e)}")
            return {
                "ticker": ticker,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def _simulate_yahoo_finance(
        self, url: str, selector_keys: Optional[List[str]] = None, selectors: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Simulate a response from Yahoo Finance.

        Args:
            url: The URL being scraped.
            selectors: The selectors used for scraping.

        Returns:
            A simulated response dictionary.
        """
        return {
            "url": url,
            "articles": [
                {
                    "title": "Markets close higher as tech stocks rally",
                    "link": "https://finance.yahoo.com/news/markets-close-higher-tech-stocks-rally",
                    "summary": "Major indices closed higher today as technology stocks led a broad market rally...",
                    "source": "Yahoo Finance",
                    "timestamp": datetime.now().isoformat(),
                },
                {
                    "title": "Federal Reserve signals potential rate cuts",
                    "link": "https://finance.yahoo.com/news/federal-reserve-signals-potential-rate-cuts",
                    "summary": "The Federal Reserve indicated it may consider interest rate cuts later this year...",
                    "source": "Yahoo Finance",
                    "timestamp": datetime.now().isoformat(),
                },
                {
                    "title": "Asian markets mixed as investors await economic data",
                    "link": "https://finance.yahoo.com/news/asian-markets-mixed-investors-await-economic-data",
                    "summary": "Asian stock markets were mixed on Wednesday as investors awaited key economic reports...",
                    "source": "Yahoo Finance",
                    "timestamp": datetime.now().isoformat(),
                },
            ],
            "timestamp": datetime.now().isoformat(),
        }

    def _simulate_cnbc(
        self, url: str, selector_keys: Optional[List[str]] = None, selectors: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Simulate a response from CNBC.

        Args:
            url: The URL being scraped.
            selectors: The selectors used for scraping.

        Returns:
            A simulated response dictionary.
        """
        return {
            "url": url,
            "articles": [
                {
                    "title": "S&P 500 hits new record as tech shares surge",
                    "link": "https://www.cnbc.com/sp-500-hits-new-record-tech-shares-surge",
                    "summary": "The S&P 500 reached a new all-time high on Tuesday, driven by strong performance in technology stocks...",
                    "source": "CNBC",
                    "timestamp": datetime.now().isoformat(),
                },
                {
                    "title": "Treasury yields fall as inflation concerns ease",
                    "link": "https://www.cnbc.com/treasury-yields-fall-inflation-concerns-ease",
                    "summary": "Treasury yields declined on Tuesday as new data suggested inflation pressures may be moderating...",
                    "source": "CNBC",
                    "timestamp": datetime.now().isoformat(),
                },
                {
                    "title": "Asian tech stocks face pressure amid regulatory concerns",
                    "link": "https://www.cnbc.com/asian-tech-stocks-face-pressure-regulatory-concerns",
                    "summary": "Technology stocks in Asian markets faced selling pressure as regulators announced new oversight measures...",
                    "source": "CNBC",
                    "timestamp": datetime.now().isoformat(),
                },
            ],
            "timestamp": datetime.now().isoformat(),
        }

    def _simulate_bloomberg(
        self, url: str, selector_keys: Optional[List[str]] = None, selectors: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Simulate a response from Bloomberg.

        Args:
            url: The URL being scraped.
            selectors: The selectors used for scraping.

        Returns:
            A simulated response dictionary.
        """
        return {
            "url": url,
            "articles": [
                {
                    "title": "Global Markets Rally as Central Banks Signal Support",
                    "link": "https://www.bloomberg.com/markets/global-markets-rally-central-banks",
                    "summary": "Global equity markets rallied after major central banks signaled continued support for economic recovery...",
                    "source": "Bloomberg",
                    "timestamp": datetime.now().isoformat(),
                },
                {
                    "title": "Asian Tech Stocks See Selloff on Regulatory Concerns",
                    "link": "https://www.bloomberg.com/markets/asian-tech-stocks-selloff",
                    "summary": "Technology stocks across Asia experienced a broad selloff as regulators in multiple countries announced new oversight measures...",
                    "source": "Bloomberg",
                    "timestamp": datetime.now().isoformat(),
                },
                {
                    "title": "Earnings Surprises: Tech Companies Outperform Expectations",
                    "link": "https://www.bloomberg.com/markets/earnings-surprises-tech-companies",
                    "summary": "Several major technology companies reported quarterly earnings that significantly exceeded analyst expectations...",
                    "source": "Bloomberg",
                    "timestamp": datetime.now().isoformat(),
                },
            ],
            "timestamp": datetime.now().isoformat(),
        }

    def _call_mcp_puppeteer(self, url: str, script: Dict[str, Any]) -> Dict[str, Any]:
        """Call the MCP Puppeteer server to execute web scraping scripts.

        Args:
            url: The URL to scrape
            script: The script containing navigation and extraction instructions

        Returns:
            Dictionary containing the extracted data or error information
        """
        if not mcp4_puppeteer_navigate or not mcp4_puppeteer_evaluate:
            logger.warning("MCP client not available. Triggering simulation fallback for _call_mcp_puppeteer.")
            # Pass selector keys if available from the script, else an empty list
            selector_keys = list(script.get("extract", {}).keys()) if "extract" in script else []
            return self._trigger_simulation_fallback(url, selector_keys)

        try:
            logger.info(f"Calling MCP Puppeteer for URL: {url}")
            navigate_options = script.get("navigate", {})
            navigate_result = self._mcp_puppeteer_navigate(url, navigate_options)

            if navigate_result and navigate_result.get("success"):
                if "extract" in script:
                    extracted_data = {}
                    errors_in_extraction = False
                    for key, js_script in script["extract"].items():
                        eval_result = self._mcp_puppeteer_evaluate(js_script)
                        if eval_result is None:  # None indicates error
                            logger.warning(f"MCP script evaluation failed for selector key: {key} on {url}")
                            errors_in_extraction = True
                            break
                        extracted_data[key] = eval_result
                    
                    if errors_in_extraction:
                        logger.warning(f"Errors occurred during MCP script evaluation for {url}. Falling back.")
                        selector_keys = list(script.get("extract", {}).keys())
                        return self._trigger_simulation_fallback(url, selector_keys)
                    else:
                        logger.info(f"MCP extraction successful for {url}.")
                        return {"success": True, **extracted_data} # Ensure success flag
                else:
                    # Navigation successful, no extraction scripts
                    return {
                        "success": True,
                        "message": f"Navigation to {url} successful, no extraction specified.",
                        "page_content": navigate_result.get("page_content", "") # Include page content if available
                    }
            else:
                logger.warning(f"MCP navigation failed for {url}. Result: {navigate_result}. Falling back.")
                selector_keys = list(script.get("extract", {}).keys()) if "extract" in script else []
                return self._trigger_simulation_fallback(url, selector_keys)

        except Exception as e:
            logger.error(f"Error in MCP Puppeteer execution for {url}: {str(e)}")
            selector_keys = list(script.get("extract", {}).keys()) if "extract" in script else []
            return self._trigger_simulation_fallback(url, selector_keys, error_message=str(e))

    def _trigger_simulation_fallback(self, url: str, selector_keys: Optional[List[str]] = None, selectors_dict: Optional[Dict[str, str]] = None, error_message: Optional[str] = None) -> Dict[str, Any]:
        """Triggers the appropriate simulation based on URL and returns a structured error if no simulation matches."""
        logger.info(f"Triggering simulation fallback for {url}. Selector keys: {selector_keys}. Error: {error_message}")
        # The 'selectors' argument for simulation methods was originally the dict,
        # but selector_keys might be more useful for some fallback logic if needed.
        # For now, we pass the original selectors_dict if available, else None.
        
        simulated_data = None
        if "finance.yahoo.com" in url:
            simulated_data = self._simulate_yahoo_finance(url, selectors=selectors_dict)
        elif "cnbc.com" in url:
            simulated_data = self._simulate_cnbc(url, selectors=selectors_dict)
        elif "bloomberg.com" in url:
            simulated_data = self._simulate_bloomberg(url, selectors=selectors_dict)
        
        if simulated_data:
            # Ensure the simulation result includes a note about it being a fallback
            simulated_data["fallback_simulation"] = True
            simulated_data["original_mcp_error"] = error_message if error_message else "MCP call failed or not available."
            return simulated_data
        else:
            # Default generic error if no specific simulation matches
            return {
                "url": url,
                "error": error_message or "MCP call failed and no specific simulation available.",
                "timestamp": datetime.now().isoformat(),
                "title": f"Simulated error for {url}",
                "content": "MCP call failed, and no specific simulation fallback was found for this URL.",
                "fallback_simulation": True,
                "success": False # Explicitly mark as not successful
            }

    def _mcp_puppeteer_navigate(
        self, url: str, options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Navigate to a URL using MCP Puppeteer.

        Args:
            url: The URL to navigate to
            options: Additional navigation options (e.g., waitUntil)

        Returns:
            Response from the navigation operation, including success status.
        """
        if not mcp4_puppeteer_navigate:
            logger.warning("mcp4_puppeteer_navigate not available. Cannot perform live navigation.")
            return {"error": "MCP client (navigate) not available", "success": False}
        
        options = options or {}
        logger.info(f"Attempting MCP navigation to URL: {url} with options: {options}")
        
        try:
            nav_options = {"url": url, "allowDangerous": False}
            nav_options.update(options) # Merge provided options
            
            # Assuming mcp4_puppeteer_navigate might take an API key,
            # but current structure relies on global MCP config or client-side env vars.
            # If self.api_key is needed, it should be passed here.
            # e.g., result = mcp4_puppeteer_navigate(nav_options, api_key=self.api_key)
            result = mcp4_puppeteer_navigate(nav_options)

            if result and result.get("success"):
                logger.info(f"MCP navigation to {url} successful.")
                # Return structure compatible with _call_mcp_puppeteer expectations
                return {"success": True, "page_content": result.get("content")}
            else:
                logger.error(f"MCP navigation to {url} failed. Raw MCP Result: {result}")
                return {"error": f"MCP navigation failed. Details: {result}", "success": False}
        except Exception as e:
            logger.error(f"Exception during MCP navigation for {url}: {str(e)}")
            return {"error": str(e), "success": False}

    def _mcp_puppeteer_evaluate(self, script: str) -> Optional[Any]:
        """Evaluate JavaScript in the browser context using MCP Puppeteer.

        Args:
            script: JavaScript to execute

        Returns:
            Result of the JavaScript evaluation, or None if an error occurs.
        """
        if not mcp4_puppeteer_evaluate:
            logger.warning("mcp4_puppeteer_evaluate not available. Cannot perform live evaluation.")
            return None # Indicates error

        logger.info(f"Attempting MCP script evaluation: {script[:100]}...")
        try:
            # Similar to navigate, if API key needed per call, it would be passed here.
            result = mcp4_puppeteer_evaluate({"script": script})

            if result and result.get("success"): # Adapt based on actual mcp_client result structure
                logger.info(f"MCP script evaluation successful.")
                return result.get("result") # This is the data extracted by the script
            else:
                logger.error(f"MCP script evaluation failed. Raw MCP Result: {result}")
                return None # Indicates error
        except Exception as e:
            logger.error(f"Exception during MCP script evaluation: {str(e)}")
            return None # Indicates error


if __name__ == "__main__":
    # Example usage
    scraper = FirecrawlScraper()
    news = scraper.scrape_financial_news(["yahoo_finance"])
    for article in news:
        print(f"Title: {article['title']}")
        print(f"Source: {article['source']}")
        print("---")
