"""Scraping Agent for extracting financial data from websites."""

from typing import Dict, List, Any, Optional
import asyncio

from loguru import logger

from agents.base_agent import BaseAgent
from data_ingestion.scrapy_spiders import run_spider, FinancialNewsSpider, SECFilingsSpider
from data_ingestion.firecrawl_scraper import FirecrawlScraper


class ScrapingAgent(BaseAgent):
    """Agent for scraping financial data from websites."""
    
    def __init__(self):
        """Initialize the Scraping agent."""
        super().__init__("Scraping Agent")
        self.firecrawl_scraper = FirecrawlScraper()
        logger.info("Scraping Agent initialized with Firecrawl scraper")
    
    async def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a scraping request.
        
        Args:
            request: The request containing operation and parameters.
                    Expected format:
                    {
                        "operation": "scrape_news"|"scrape_sec_filings"|"scrape_earnings"|"scrape_url",
                        "parameters": {...}  # Operation-specific parameters
                    }
            
        Returns:
            The scraped data result.
        """
        operation = request.get("operation")
        parameters = request.get("parameters", {})
        
        if not operation:
            return {"error": "No operation specified"}
        
        # Use the appropriate scraping method based on the operation
        if operation == "scrape_news":
            sources = parameters.get("sources")
            use_firecrawl = parameters.get("use_firecrawl", True)
            
            if use_firecrawl:
                # Use Firecrawl for dynamic content
                news = await asyncio.to_thread(self.firecrawl_scraper.scrape_financial_news, sources)
                return {"data": news, "source": "Firecrawl"}
            else:
                # Use Scrapy for static content
                news = await asyncio.to_thread(run_spider, FinancialNewsSpider, sources=sources)
                return {"data": news, "source": "Scrapy"}
        
        elif operation == "scrape_sec_filings":
            tickers = parameters.get("tickers")
            if not tickers:
                return {"error": "No tickers specified for scrape_sec_filings operation"}
            
            filings = await asyncio.to_thread(run_spider, SECFilingsSpider, tickers=tickers)
            return {"data": filings, "source": "Scrapy"}
        
        elif operation == "scrape_earnings":
            ticker = parameters.get("ticker")
            if not ticker:
                return {"error": "No ticker specified for scrape_earnings operation"}
            
            earnings = await asyncio.to_thread(self.firecrawl_scraper.scrape_earnings_report, ticker)
            return {"data": earnings, "source": "Firecrawl"}
        
        elif operation == "scrape_url":
            url = parameters.get("url")
            selectors = parameters.get("selectors")
            
            if not url:
                return {"error": "No URL specified for scrape_url operation"}
            
            content = await asyncio.to_thread(self.firecrawl_scraper.scrape_url, url, selectors)
            return {"data": content, "source": "Firecrawl"}
        
        else:
            return {"error": f"Unknown operation: {operation}"}
    
    async def scrape_news(self, sources: Optional[List[str]] = None, use_firecrawl: bool = True) -> Dict[str, Any]:
        """Scrape financial news from various sources.
        
        Args:
            sources: List of news sources to scrape.
            use_firecrawl: Whether to use Firecrawl for dynamic content.
            
        Returns:
            Scraped news data.
        """
        request = {
            "operation": "scrape_news",
            "parameters": {"sources": sources, "use_firecrawl": use_firecrawl}
        }
        return await self.run(request)
    
    async def scrape_sec_filings(self, tickers: List[str]) -> Dict[str, Any]:
        """Scrape SEC filings for specified tickers.
        
        Args:
            tickers: List of stock ticker symbols.
            
        Returns:
            Scraped SEC filings data.
        """
        request = {
            "operation": "scrape_sec_filings",
            "parameters": {"tickers": tickers}
        }
        return await self.run(request)
    
    async def scrape_earnings(self, ticker: str) -> Dict[str, Any]:
        """Scrape earnings report for a company.
        
        Args:
            ticker: The stock ticker symbol.
            
        Returns:
            Scraped earnings data.
        """
        request = {
            "operation": "scrape_earnings",
            "parameters": {"ticker": ticker}
        }
        return await self.run(request)
    
    async def scrape_url(self, url: str, selectors: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Scrape content from a specific URL.
        
        Args:
            url: The URL to scrape.
            selectors: CSS selectors to extract specific content.
            
        Returns:
            Scraped content.
        """
        request = {
            "operation": "scrape_url",
            "parameters": {"url": url, "selectors": selectors}
        }
        return await self.run(request)
    
    async def scrape_market_news(self) -> Dict[str, Any]:
        """Scrape the latest market news from multiple sources.
        
        Returns:
            Combined market news from various sources.
        """
        # Scrape news from multiple sources
        sources = ["yahoo_finance", "cnbc", "bloomberg"]
        scrapy_result = await self.scrape_news(sources, use_firecrawl=False)
        firecrawl_result = await self.scrape_news(sources, use_firecrawl=True)
        
        # Combine results
        combined_news = []
        
        if "data" in scrapy_result:
            combined_news.extend(scrapy_result["data"])
        
        if "data" in firecrawl_result:
            combined_news.extend(firecrawl_result["data"])
        
        # Remove duplicates based on title
        seen_titles = set()
        unique_news = []
        
        for item in combined_news:
            title = item.get("title")
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_news.append(item)
        
        return {"data": unique_news}
