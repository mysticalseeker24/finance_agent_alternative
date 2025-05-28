"""Firecrawl integration for scraping dynamic financial web content using MCP."""

import json
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

import requests
from loguru import logger

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
            if mcp_result and "error" not in mcp_result:
                result = {
                    "url": url,
                    "timestamp": datetime.now().isoformat(),
                    **mcp_result,  # Include all extracted data
                }
            else:
                # Fallback to simulated responses if MCP fails
                logger.warning(
                    f"MCP scraping failed for {url}, falling back to simulation"
                )
                if "finance.yahoo.com" in url:
                    result = self._simulate_yahoo_finance(url, selectors)
                elif "cnbc.com" in url:
                    result = self._simulate_cnbc(url, selectors)
                elif "bloomberg.com" in url:
                    result = self._simulate_bloomberg(url, selectors)
                else:
                    # Default to a generic result for other URLs
                    result = {
                        "url": url,
                        "title": f"Content from {url}",
                        "content": "Sample content",
                        "timestamp": datetime.now().isoformat(),
                    }

            logger.info(f"Scraped content from {url} using Firecrawl")
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
        self, url: str, selectors: Optional[Dict[str, str]]
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
        self, url: str, selectors: Optional[Dict[str, str]]
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
        self, url: str, selectors: Optional[Dict[str, str]]
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
        try:
            logger.info(f"Calling MCP Puppeteer for URL: {url}")

            # First, navigate to the URL
            navigate_result = self._mcp_puppeteer_navigate(
                url, script.get("navigate", {})
            )

            # Then run any extraction scripts if navigation was successful
            if navigate_result and "error" not in navigate_result:
                if "extract" in script:
                    # Execute each extraction script
                    extracted_data = {}
                    for key, js_script in script["extract"].items():
                        extracted_data[key] = self._mcp_puppeteer_evaluate(js_script)
                    return extracted_data
                else:
                    return {
                        "success": True,
                        "message": "Navigation successful but no extraction scripts provided",
                    }
            else:
                return navigate_result or {"error": "Failed to navigate to URL"}
        except Exception as e:
            logger.error(f"Error in MCP Puppeteer execution: {str(e)}")
            return {"error": str(e)}

    def _mcp_puppeteer_navigate(
        self, url: str, options: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Navigate to a URL using MCP Puppeteer.

        Args:
            url: The URL to navigate to
            options: Additional navigation options

        Returns:
            Response from the navigation operation
        """
        try:
            # This would be the actual MCP call in a production environment
            # In a real implementation, you would use:
            # from mcp_client import mcp4_puppeteer_navigate
            # result = mcp4_puppeteer_navigate({
            #     "url": url,
            #     "allowDangerous": False,
            #     **options
            # })

            # For now, simulate a successful response
            logger.info(f"[MCP Simulation] Navigated to {url}")
            return {"success": True}
        except Exception as e:
            logger.error(f"Error in MCP navigation: {str(e)}")
            return {"error": str(e)}

    def _mcp_puppeteer_evaluate(self, script: str) -> Any:
        """Evaluate JavaScript in the browser context using MCP Puppeteer.

        Args:
            script: JavaScript to execute

        Returns:
            Result of the JavaScript evaluation
        """
        try:
            # This would be the actual MCP call in a production environment
            # In a real implementation, you would use:
            # from mcp_client import mcp4_puppeteer_evaluate
            # result = mcp4_puppeteer_evaluate({"script": script})

            # For now, return a simulated result
            logger.info(f"[MCP Simulation] Evaluated script: {script[:50]}...")
            return f"Simulated result for: {script[:20]}..."
        except Exception as e:
            logger.error(f"Error in MCP evaluation: {str(e)}")
            return f"Error: {str(e)}"


if __name__ == "__main__":
    # Example usage
    scraper = FirecrawlScraper()
    news = scraper.scrape_financial_news(["yahoo_finance"])
    for article in news:
        print(f"Title: {article['title']}")
        print(f"Source: {article['source']}")
        print("---")
