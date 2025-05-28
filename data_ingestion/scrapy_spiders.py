"""Scrapy spiders for financial data extraction."""

import os
from typing import Dict, List, Any, Iterable, Optional
from datetime import datetime

import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from loguru import logger


class FinancialNewsSpider(scrapy.Spider):
    """Spider for scraping financial news from various sources."""

    name = "financial_news"

    def __init__(self, sources: Optional[List[str]] = None, *args, **kwargs):
        """Initialize the spider.

        Args:
            sources: List of news sources to scrape. If None, all available sources are used.
        """
        super().__init__(*args, **kwargs)

        # Define available sources with their URLs and parsing methods
        self.available_sources = {
            "yahoo_finance": {
                "url": "https://finance.yahoo.com/news/",
                "parser": self.parse_yahoo_finance,
            },
            "cnbc": {"url": "https://www.cnbc.com/finance/", "parser": self.parse_cnbc},
            "bloomberg": {
                "url": "https://www.bloomberg.com/markets",
                "parser": self.parse_bloomberg,
            },
        }

        # Set sources to crawl
        self.sources_to_crawl = sources or list(self.available_sources.keys())

        # Validate sources
        for source in self.sources_to_crawl:
            if source not in self.available_sources:
                logger.warning(f"Unknown source: {source}. It will be skipped.")
                self.sources_to_crawl.remove(source)

    def start_requests(self) -> Iterable[scrapy.Request]:
        """Generate initial requests for the spider.

        Returns:
            An iterable of scrapy.Request objects.
        """
        for source in self.sources_to_crawl:
            source_info = self.available_sources[source]
            yield scrapy.Request(
                url=source_info["url"],
                callback=source_info["parser"],
                meta={"source": source},
            )

    def parse_yahoo_finance(
        self, response: scrapy.http.Response
    ) -> Iterable[Dict[str, Any]]:
        """Parse Yahoo Finance news articles.

        Args:
            response: The HTTP response from the request.

        Returns:
            An iterable of article items.
        """
        # Extract article listings
        articles = response.css("div.Ov\\(h\\).Pend\\(44px\\).Pstart\\(25px\\)")

        for article in articles:
            title = article.css("h3.Mb\\(5px\\)::text").get()
            url = article.css("a::attr(href)").get()
            summary = article.css("p::text").get()

            # Clean up the data
            if title:
                title = title.strip()

            if url and not url.startswith("http"):
                url = f"https://finance.yahoo.com{url}"

            if summary:
                summary = summary.strip()

            # Yield the article data
            if title and url:
                yield {
                    "title": title,
                    "url": url,
                    "summary": summary,
                    "source": "Yahoo Finance",
                    "timestamp": datetime.now().isoformat(),
                    "category": "news",
                }

    def parse_cnbc(self, response: scrapy.http.Response) -> Iterable[Dict[str, Any]]:
        """Parse CNBC news articles.

        Args:
            response: The HTTP response from the request.

        Returns:
            An iterable of article items.
        """
        # Extract article listings
        articles = response.css("div.Card-titleContainer")

        for article in articles:
            title = article.css("a.Card-title::text").get()
            url = article.css("a.Card-title::attr(href)").get()

            # Clean up the data
            if title:
                title = title.strip()

            # Yield the article data
            if title and url:
                yield {
                    "title": title,
                    "url": url,
                    "summary": None,  # CNBC doesn't show summaries in listings
                    "source": "CNBC",
                    "timestamp": datetime.now().isoformat(),
                    "category": "news",
                }

    def parse_bloomberg(
        self, response: scrapy.http.Response
    ) -> Iterable[Dict[str, Any]]:
        """Parse Bloomberg news articles.

        Args:
            response: The HTTP response from the request.

        Returns:
            An iterable of article items.
        """
        # Extract article listings
        articles = response.css("article.story-package-module__story")

        for article in articles:
            title = article.css("h3.story-package-module__headline::text").get()
            url = article.css("a.story-package-module__headline::attr(href)").get()
            summary = article.css("p.story-package-module__summary::text").get()

            # Clean up the data
            if title:
                title = title.strip()

            if url and not url.startswith("http"):
                url = f"https://www.bloomberg.com{url}"

            if summary:
                summary = summary.strip()

            # Yield the article data
            if title and url:
                yield {
                    "title": title,
                    "url": url,
                    "summary": summary,
                    "source": "Bloomberg",
                    "timestamp": datetime.now().isoformat(),
                    "category": "news",
                }


class SECFilingsSpider(scrapy.Spider):
    """Spider for scraping SEC filings from EDGAR database."""

    name = "sec_filings"
    start_urls = ["https://www.sec.gov/edgar/searchedgar/companysearch.html"]

    def __init__(self, tickers: Optional[List[str]] = None, *args, **kwargs):
        """Initialize the spider.

        Args:
            tickers: List of stock tickers to search for. If None, no filings are retrieved.
        """
        super().__init__(*args, **kwargs)
        self.tickers = tickers or []

    def parse(self, response: scrapy.http.Response) -> Iterable[scrapy.Request]:
        """Parse the SEC EDGAR search page.

        Args:
            response: The HTTP response from the request.

        Returns:
            An iterable of scrapy.Request objects for company pages.
        """
        # For each ticker, navigate to the company's SEC filings page
        for ticker in self.tickers:
            # In a real implementation, we would use the SEC's API or navigate through forms
            # For simplicity, we'll use a direct URL pattern (this is just an example)
            # The actual implementation would need to use the EDGAR API or more complex navigation
            company_url = f"https://www.sec.gov/cgi-bin/browse-edgar?CIK={ticker}&owner=exclude&action=getcompany"
            yield scrapy.Request(
                url=company_url,
                callback=self.parse_company_page,
                meta={"ticker": ticker},
            )

    def parse_company_page(
        self, response: scrapy.http.Response
    ) -> Iterable[Dict[str, Any]]:
        """Parse a company's SEC filings page.

        Args:
            response: The HTTP response from the request.

        Returns:
            An iterable of filing items.
        """
        ticker = response.meta.get("ticker")

        # Extract filings table
        filings = response.css("tr")

        for filing in filings[1:]:  # Skip header row
            columns = filing.css("td")
            if len(columns) >= 5:
                filing_type = columns[0].css("::text").get()
                filing_date = columns[3].css("::text").get()
                filing_url = columns[1].css("a::attr(href)").get()

                # Clean up the data
                if filing_type:
                    filing_type = filing_type.strip()

                if filing_date:
                    filing_date = filing_date.strip()

                if filing_url and not filing_url.startswith("http"):
                    filing_url = f"https://www.sec.gov{filing_url}"

                # Yield the filing data
                if filing_type and filing_date and filing_url:
                    yield {
                        "ticker": ticker,
                        "filing_type": filing_type,
                        "filing_date": filing_date,
                        "filing_url": filing_url,
                        "source": "SEC EDGAR",
                        "timestamp": datetime.now().isoformat(),
                        "category": "filing",
                    }


class ScrapyPipeline:
    """Pipeline for processing and storing scraped data."""

    def __init__(self, output_dir: str = "scraped_data"):
        """Initialize the pipeline.

        Args:
            output_dir: Directory to store scraped data.
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def process_item(
        self, item: Dict[str, Any], spider: scrapy.Spider
    ) -> Dict[str, Any]:
        """Process a scraped item.

        Args:
            item: The scraped item.
            spider: The spider that scraped the item.

        Returns:
            The processed item.
        """
        # Add any additional processing here
        logger.info(
            f"Processed item: {item['title'] if 'title' in item else item['filing_type']}"
        )
        return item


def run_spider(spider_cls: scrapy.Spider, **kwargs) -> List[Dict[str, Any]]:
    """Run a spider and return the scraped items.

    Args:
        spider_cls: The spider class to run.
        **kwargs: Additional arguments to pass to the spider.

    Returns:
        A list of scraped items.
    """
    # Create a settings dictionary
    settings = get_project_settings()
    settings.set("ITEM_PIPELINES", {"__main__.ScrapyPipeline": 300})
    settings.set("LOG_LEVEL", "INFO")

    # List to store items
    items = []

    # Define a pipeline that captures items
    class CaptureItemsPipeline:
        def process_item(self, item, spider):
            items.append(item)
            return item

    # Update settings to use our pipeline
    settings.set("ITEM_PIPELINES", {"__main__.CaptureItemsPipeline": 300})

    # Create and run the crawler
    process = CrawlerProcess(settings)
    process.crawl(spider_cls, **kwargs)
    process.start()  # This will block until crawling is finished

    return items


if __name__ == "__main__":
    # Example usage
    news_items = run_spider(FinancialNewsSpider, sources=["yahoo_finance"])
    for item in news_items:
        print(f"Title: {item['title']}")
        print(f"URL: {item['url']}")
        print(f"Source: {item['source']}")
        print("---")
