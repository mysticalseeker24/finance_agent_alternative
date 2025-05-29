"""YFinance client for fetching market data."""

import time
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta

import yfinance as yf
import pandas as pd
import redis
from loguru import logger
import json  # Added for cache handling
import requests  # To catch requests.exceptions.RequestException
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import re  # Added for regex parsing

from config import Config
from data_ingestion.firecrawl_scraper import FirecrawlScraper  # Added import


class YFinanceClient:
    """Client for fetching market data from Yahoo Finance."""

    def __init__(self):
        """Initialize the YFinance client."""
        self.cache = redis.Redis(
            host=Config.REDIS_HOST,
            port=Config.REDIS_PORT,
            decode_responses=True,  # Keep True for direct string operations
            socket_timeout=5,
        )
        self.firecrawl_scraper = FirecrawlScraper()  # Added FirecrawlScraper instance
        logger.info("YFinance client initialized with FirecrawlScraper")

    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get data from cache.

        Args:
            key: The cache key.

        Returns:
            The cached data if available, None otherwise.
        """
        try:
            data_str = self.cache.get(key)
            if data_str:
                return json.loads(data_str)
            return None
        except redis.RedisError as e:
            logger.warning(f"Redis error when getting from cache: {e}")
        except json.JSONDecodeError as e:
            logger.warning(
                f"JSON decode error when getting from cache for key {key}: {e}"
            )
            return None

    def _set_in_cache(self, key: str, value: Any, expiry: int = 3600) -> bool:
        """Set data in cache.

        Args:
            key: The cache key.
            value: The data to cache.
            expiry: The cache expiry time in seconds (default: 1 hour).

        Returns:
            True if successful, False otherwise.
        """
        try:
            self.cache.set(key, json.dumps(value), ex=expiry)
            return True
        except redis.RedisError as e:
            logger.warning(f"Redis error when setting in cache: {e}")
            return False
        except TypeError as e:  # Handle cases where value is not JSON serializable
            logger.error(
                f"TypeError when setting cache for key {key}: {e}. Value: {value}"
            )
            return False

    @retry(
        wait=wait_exponential(
            multiplier=1, min=2, max=10
        ),  # Exponential backoff: 2s, 4s, 8s ... up to 10s
        stop=stop_after_attempt(3),  # Retry up to 3 times
        retry=retry_if_exception_type(
            (requests.exceptions.RequestException, ConnectionError)
        ),  # Retry on specific network errors
    )
    def _fetch_stock_info_with_retry(self, ticker_obj):
        # Helper to encapsulate the part that needs retrying
        logger.debug(
            f"Fetching stock info for {ticker_obj.ticker} (attempt {self._fetch_stock_info_with_retry.retry.statistics.get('attempt_number', 1)})"
        )
        return ticker_obj.info

    def get_stock_info(self, ticker: str) -> Dict[str, Any]:
        """Get stock information.

        Args:
            ticker: The stock ticker symbol.

        Returns:
            A dictionary of stock information.
        """
        cache_key = f"stock_info:{ticker}"
        cached_data = self._get_from_cache(cache_key)

        if cached_data:
            logger.debug(f"Retrieved stock info for {ticker} from cache")
            return cached_data

        try:
            stock = yf.Ticker(ticker)
            info = self._fetch_stock_info_with_retry(stock)  # New call via retry helper

            # Extract relevant information
            stock_info = {
                "symbol": ticker,
                "name": info.get("shortName", ""),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "country": info.get("country", ""),
                "market_cap": info.get("marketCap", 0),
                "price": info.get("currentPrice", 0),
                "change_percent": info.get("regularMarketChangePercent", 0),
                "pe_ratio": info.get("trailingPE", 0),
                "dividend_yield": info.get("dividendYield", 0),
                "fifty_two_week_high": info.get("fiftyTwoWeekHigh", 0),
                "fifty_two_week_low": info.get("fiftyTwoWeekLow", 0),
                "timestamp": datetime.now().isoformat(),
            }

            # Cache the data for 1 hour
            self._set_in_cache(cache_key, stock_info, 3600)

            logger.info(f"Retrieved stock info for {ticker} from YFinance")
            return stock_info

        except (
            Exception
        ) as e:  # This will catch tenacity's RetryError if all retries fail
            logger.error(
                f"Error getting stock info for {ticker} after retries: {str(e)}"
            )
            return {
                "symbol": ticker,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def get_historical_data(
        self, ticker: str, period: str = "1mo", interval: str = "1d"
    ) -> Union[pd.DataFrame, Dict[str, Any]]:
        """Get historical stock data.

        Args:
            ticker: The stock ticker symbol.
            period: The period to fetch data for (default: 1 month).
                   Valid values: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
            interval: The data interval (default: 1 day).
                      Valid values: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo

        Returns:
            A dataframe of historical data or an error dictionary.
        """
        cache_key = f"historical:{ticker}:{period}:{interval}"
        cached_data = self._get_from_cache(cache_key)

        if cached_data:
            logger.debug(f"Retrieved historical data for {ticker} from cache")
            return pd.DataFrame.from_dict(cached_data)

        try:
            stock = yf.Ticker(ticker)
            data = self._fetch_historical_data_with_retry(
                stock, period, interval
            )  # New call

            # Reset index to make Date a column
            data = data.reset_index()

            # Convert to dict for caching
            data_dict = data.to_dict(orient="records")

            # Cache the data (TTL depends on the interval)
            if interval in ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"]:
                expiry = 300  # 5 minutes for intraday data
            else:
                expiry = 3600  # 1 hour for daily or longer data

            self._set_in_cache(cache_key, data_dict, expiry)

            logger.info(f"Retrieved historical data for {ticker} from YFinance")
            return data

        except Exception as e:  # This will catch tenacity's RetryError
            logger.error(
                f"Error getting historical data for {ticker} after retries: {str(e)}"
            )
            return {
                "symbol": ticker,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def get_market_summary(self) -> Dict[str, Any]:
        """Get a summary of major market indices.

        Returns:
            A dictionary containing major market indices data.
        """
        cache_key = "market_summary"
        cached_data = self._get_from_cache(cache_key)

        if cached_data:
            logger.debug("Retrieved market summary from cache")
            return cached_data

        # List of major indices to track
        indices = [
            "^GSPC",  # S&P 500
            "^DJI",  # Dow Jones Industrial Average
            "^IXIC",  # NASDAQ Composite
            "^FTSE",  # FTSE 100
            "^N225",  # Nikkei 225
            "^HSI",  # Hang Seng Index
            "^NSEI",  # NIFTY 50
        ]

        try:
            result = {}
            for index_ticker_str in indices:
                stock_obj = yf.Ticker(index_ticker_str)
                index_data = self._fetch_stock_info_with_retry(
                    stock_obj
                )  # Reusing helper

                result[index_ticker_str] = {
                    "name": index_data.get("shortName", ""),
                    "price": index_data.get("regularMarketPrice", 0),
                    "change": index_data.get("regularMarketChange", 0),
                    "change_percent": index_data.get("regularMarketChangePercent", 0),
                    "previous_close": index_data.get("regularMarketPreviousClose", 0),
                }

            # Add timestamp
            result["timestamp"] = datetime.now().isoformat()

            # Cache for 5 minutes
            self._set_in_cache(cache_key, result, 300)

            logger.info("Retrieved market summary from YFinance")
            return result

        except Exception as e:  # This will catch tenacity's RetryError
            logger.error(
                f"Error getting market summary after retries on one of the indices: {str(e)}"
            )
            return {"error": str(e), "timestamp": datetime.now().isoformat()}

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(
            (requests.exceptions.RequestException, ConnectionError)
        ),
    )
    def _fetch_historical_data_with_retry(self, ticker_obj, period, interval):
        # Helper to encapsulate the part that needs retrying
        logger.debug(
            f"Fetching historical data for {ticker_obj.ticker} (period: {period}, interval: {interval}, attempt: {self._fetch_historical_data_with_retry.retry.statistics.get('attempt_number', 1)})"
        )
        return ticker_obj.history(period=period, interval=interval)

    # Note: The get_earnings_calendar method using FirecrawlScraper is not modified
    # as it relies on a different fetching mechanism (FirecrawlScraper handles its own retries/fallbacks).
    # The original mock get_earnings_calendar is also left as is, if it were still active.
    # The focus here is on yfinance direct calls.

    def get_earnings_calendar(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get upcoming earnings releases.

        Args:
            days: Number of days to look ahead (default: 7).

        Returns:
            A list of dictionaries containing earnings release information.
        """
        cache_key = f"earnings_calendar:{days}"
        cached_data = self._get_from_cache(cache_key)

        if cached_data:
            logger.debug(f"Retrieved earnings calendar for {days} days from cache")
            return cached_data

        try:
            # Get today's date and the end date
            today = datetime.now().date()
            end_date = today + timedelta(days=days)

            # Format dates as strings
            start_str = today.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")

            # Use yfinance to get the earnings calendar
            # Note: This is a placeholder as yfinance doesn't directly provide earnings calendar
            # In a real implementation, we might need to use web scraping or another API
            # For now, we'll return a mock response

            # Mock data
            earnings = [
                {
                    "symbol": "AAPL",
                    "company_name": "Apple Inc.",
                    "date": (today + timedelta(days=2)).isoformat(),
                    "eps_estimate": 1.45,
                    "eps_actual": None,
                    "time": "AMC",  # After Market Close
                },
                {
                    "symbol": "MSFT",
                    "company_name": "Microsoft Corporation",
                    "date": (today + timedelta(days=3)).isoformat(),
                    "eps_estimate": 2.31,
                    "eps_actual": None,
                    "time": "BMO",  # Before Market Open
                },
                # Add more mock data as needed
            ]

            # Cache for 6 hours
            self._set_in_cache(cache_key, earnings, 21600)

            logger.info(f"Retrieved earnings calendar for {days} days")
            return earnings

        except Exception as e:
            logger.error(
                f"Error getting earnings calendar: {str(e)}"
            )  # This refers to the live scraped one
            return [{"error": str(e), "timestamp": datetime.now().isoformat()}]

    def _parse_yahoo_earnings_page_from_content(
        self, page_content: str
    ) -> List[Dict[str, Any]]:
        """Parse scraped Yahoo Finance earnings calendar page content (markdown/text).

        Args:
            page_content: The markdown or text content of the earnings calendar page.

        Returns:
            A list of parsed earnings event dictionaries.
        """
        parsed_events = []
        if not page_content:
            logger.warning(
                "No page content provided to _parse_yahoo_earnings_page_from_content."
            )
            return []

        # Regex to capture relevant data. This is highly dependent on Yahoo's page structure
        # and Firecrawl's markdown conversion. It needs to be robust.
        # Example line from Yahoo (might be in a markdown table):
        # "Symbol Company EPS Estimate Earnings Call Time" - Header
        # "AAPL Apple Inc 1.50 Before Market Open" - Data Row
        # This will try to find list items or table rows in markdown.
        # A more complex approach might involve parsing markdown tables if Firecrawl produces them.

        # This regex is illustrative and will likely need significant refinement.
        # It looks for patterns like:
        # Optional markdown list/table chars: ^\s*[-*|]?\s*
        # Symbol: ([A-Z]{1,5})
        # Company Name: (.*?)\s*\(?\1\)?\s*  (captures name, optionally followed by symbol in parens)
        # Earnings Date: (Today|Tomorrow|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},\s+\d{4}|\d{1,2}/\d{1,2}/\d{4})
        # Earnings Time: (Before Market Open|After Market Close|Time Not Supplied|During Market Hours|\d{1,2}:\d{2}\s*(?:AM|PM)\s*(?:ET|EST|EDT|CT|CST|CDT|MT|MST|MDT|PT|PST|PDT)?)
        # EPS Estimate: (?:Est\.:\s*|Estimate:\s*|EPS Estimate:\s*)?(-?\d+\.\d{2}|N/A|-)
        # This is very optimistic. A simpler line-by-line processing might be more robust initially.

        # Let's try a simpler approach: iterate line by line and look for patterns.
        # Yahoo's calendar often has a list-like structure for each day's earnings.
        # Firecrawl's `onlyMainContent: True` should give us a cleaner text version.

        current_year = datetime.now().year
        lines = page_content.splitlines()

        # General pattern to identify a line that might contain an earnings event.
        # This is very broad and will need refinement based on actual Firecrawl output.
        # It looks for a ticker, then some text (company), then a date-like string, then time-like string, then optional EPS.
        # Example target text that Firecrawl might produce from a list item:
        # "AAPL Apple Inc. May 1 AMC Estimate: 1.50"
        # "TSLA Tesla, Inc. Apr 23 After Market Close Estimate: 0.65"
        # "NVDA NVIDIA Corporation Today Before Market Open -" (Estimate might be missing or '-')

        # Regex to capture SYMBOL, COMPANY NAME, DATE, TIME, EPS_ESTIMATE
        # This pattern is complex and assumes a certain ordering and format.
        # It's broken down for readability.
        pattern_str = (
            r"^(?P<symbol>[A-Z]{1,5}(?:\.[A-Z])?)\s+"  # Symbol (e.g., AAPL, BRK.B)
            r"(?P<company_name>.+?)\s+"  # Company Name (non-greedy)
            # Date (handles "Month Day", "Month Day, Year", "Today", "Tomorrow")
            r"(?P<earnings_date>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}(?:,\s*\d{4})?|Today|Tomorrow)\s*"
            # Time (flexible for various Yahoo formats)
            r"(?P<earnings_time>(?:Before Market Open|After Market Close|During Market Hours|Time Not Supplied|TAS|BMO|AMC|\d{1,2}:\d{2}\s*(?:AM|PM)\s*(?:[A-Z]{2,3}T)?))"
            r"(?:\s*Estimate:\s*(?P<eps_estimate>-?\d+\.\d{2}|-|N/A))?.*$"  # Optional EPS Estimate
        )
        # A simpler pattern if the above is too strict, focusing on symbol and date:
        # pattern_str = r"^(?P<symbol>[A-Z]{1,5}(?:\.[A-Z])?)\s+(?P<company_name>.+?)\s+(?P<earnings_date>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}(?:,\s*\d{4})?|Today|Tomorrow)"

        event_pattern = re.compile(pattern_str, re.IGNORECASE)

        for line in lines:
            line = line.strip()
            if not line:
                continue

            match = event_pattern.search(line)
            if match:
                data = match.groupdict()

                symbol = data.get("symbol", "N/A").upper()
                company_name = data.get("company_name", "N/A").strip()
                # Clean up company name if symbol is appended e.g. "Apple Inc. AAPL"
                if company_name.endswith(symbol):
                    company_name = company_name[: -len(symbol)].strip()
                if company_name.endswith(f"({symbol})"):  # e.g. "Apple Inc. (AAPL)"
                    company_name = company_name[: -len(symbol) - 2].strip()

                raw_date_str = data.get("earnings_date", "N/A").strip()
                earnings_date_iso = "N/A"

                if raw_date_str.lower() == "today":
                    earnings_date_iso = datetime.now().strftime("%Y-%m-%d")
                elif raw_date_str.lower() == "tomorrow":
                    earnings_date_iso = (datetime.now() + timedelta(days=1)).strftime(
                        "%Y-%m-%d"
                    )
                else:
                    try:
                        # Check if year is missing, e.g., "May 01"
                        if "," not in raw_date_str:
                            date_obj = datetime.strptime(
                                f"{raw_date_str} {current_year}", "%b %d %Y"
                            )
                        else:  # e.g. "May 01, 2024"
                            date_obj = datetime.strptime(raw_date_str, "%b %d, %Y")
                        earnings_date_iso = date_obj.strftime("%Y-%m-%d")
                    except ValueError as ve:
                        logger.debug(
                            f"Could not parse date '{raw_date_str}' for {symbol}: {ve}"
                        )
                        earnings_date_iso = (
                            raw_date_str  # Keep original if parsing fails
                        )

                earnings_time = data.get("earnings_time", "Time Not Supplied").strip()
                if not earnings_time:
                    earnings_time = "Time Not Supplied"

                # Normalize time abbreviations
                if earnings_time.upper() == "BMO":
                    earnings_time = "Before Market Open"
                if earnings_time.upper() == "AMC":
                    earnings_time = "After Market Close"
                if earnings_time.upper() == "TAS":
                    earnings_time = "Time Not Supplied"  # Or "During Market Hours"

                eps_estimate_raw = data.get("eps_estimate")
                eps_estimate = None
                if eps_estimate_raw and eps_estimate_raw.strip() not in [
                    "-",
                    "N/A",
                    "",
                ]:
                    try:
                        eps_estimate = float(eps_estimate_raw.strip())
                    except ValueError:
                        logger.debug(
                            f"Could not parse EPS estimate '{eps_estimate_raw}' for {symbol}"
                        )

                event = {
                    "symbol": symbol,
                    "company_name": company_name,
                    "earnings_date": earnings_date_iso,
                    "earnings_time": earnings_time,
                    "eps_estimate": eps_estimate,
                    "reported_eps": None,  # Not available from calendar view
                    "market_cap": None,  # Not available from calendar view
                    "source": "Yahoo Finance (via Firecrawl text parse)",
                }
                # Avoid duplicates if multiple lines match for the same core info (less likely with strict regex)
                # This simple check might not be sufficient for very messy data.
                is_duplicate = False
                for pe in parsed_events:
                    if (
                        pe["symbol"] == event["symbol"]
                        and pe["earnings_date"] == event["earnings_date"]
                    ):
                        is_duplicate = True
                        break
                if not is_duplicate:
                    parsed_events.append(event)
            # else:
            # logger.debug(f"Line did not match earnings event pattern: {line[:100]}")

        if not parsed_events:
            logger.warning(
                "Parsing Yahoo Finance earnings calendar from page content (markdown/text) yielded no events. "
                "This may be due to changes in page structure or Firecrawl output format. "
                "The regex patterns may need adjustment."
            )
        else:
            logger.info(
                f"Successfully parsed {len(parsed_events)} earnings events from page content."
            )
        return parsed_events

    async def get_earnings_calendar(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get upcoming earnings releases by scraping Yahoo Finance.

        Args:
            days: Number of days to look ahead. Note: Current scraping implementation
                  fetches the default view of Yahoo's calendar (often current week)
                  and may not precisely honor this 'days' parameter without
                  more complex date-based navigation during scraping.

        Returns:
            A list of dictionaries containing earnings release information.
        """
        cache_key = f"live_earnings_calendar:default_view_approx_{days}d_textparse"  # Updated cache key
        cached_data = self._get_from_cache(cache_key)

        if cached_data:
            logger.debug(
                f"Retrieved earnings calendar (text_parse default view) from cache. Requested days: {days}"
            )
            return cached_data

        logger.info(
            f"Fetching live earnings calendar (text_parse). Requested days: {days}. "
            "Note: Scraping fetches Yahoo's default calendar view (typically current week)."
        )

        EARNINGS_CALENDAR_URL = "https://finance.yahoo.com/calendar/earnings"

        # No selectors needed, FirecrawlScraper will get main content
        try:
            # Default pageOptions in FirecrawlScraper are {"onlyMainContent": True, "includeHtml": False}
            # which should provide markdown or main text content.
            scraped_page_response = await self.firecrawl_scraper.scrape_url(
                url_to_scrape=EARNINGS_CALENDAR_URL
            )

            if not scraped_page_response.get("success"):
                error_msg = scraped_page_response.get("error", "Unknown error")
                logger.error(
                    f"Failed to scrape earnings calendar page: {error_msg} - Details: {scraped_page_response.get('details')}"
                )
                self._set_in_cache(
                    cache_key,
                    [{"error": "Scraping failed.", "details": error_msg}],
                    300,
                )
                return [
                    {
                        "error": "Failed to retrieve earnings calendar page via Firecrawl.",
                        "details": error_msg,
                        "timestamp": datetime.now().isoformat(),
                    }
                ]

            page_content = scraped_page_response.get("content")
            if not page_content:
                logger.warning(
                    f"No content received from earnings calendar page scrape at {EARNINGS_CALENDAR_URL}."
                )
                self._set_in_cache(cache_key, [], 300)  # Cache empty result for 5 mins
                return []

            parsed_earnings = self._parse_yahoo_earnings_page_from_content(page_content)

            if not parsed_earnings:
                logger.warning("No earnings events parsed from scraped page content.")
                self._set_in_cache(cache_key, [], 300)  # Cache empty result for 5 mins
                return []

            self._set_in_cache(cache_key, parsed_earnings, 21600)  # Cache for 6 hours
            logger.info(
                f"Retrieved and parsed live earnings calendar with {len(parsed_earnings)} events using text parsing."
            )
            return parsed_earnings

        except Exception as e:
            logger.exception(
                f"General error in get_earnings_calendar (text_parse): {str(e)}"
            )
            self._set_in_cache(cache_key, [{"error": str(e)}], 300)
            return [{"error": str(e), "timestamp": datetime.now().isoformat()}]

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(
            (requests.exceptions.RequestException, ConnectionError)
        ),
    )
    def _fetch_earnings_dates_with_retry(self, ticker_obj):
        # Helper to encapsulate the yfinance call that needs retrying
        logger.debug(
            f"Fetching earnings dates for {ticker_obj.ticker} (attempt {self._fetch_earnings_dates_with_retry.retry.statistics.get('attempt_number', 1)})"
        )
        earnings_data = ticker_obj.earnings_dates
        if earnings_data is None or earnings_data.empty:
            logger.warning(
                f"No earnings dates data returned by yfinance for {ticker_obj.ticker}"
            )
            # Return an empty DataFrame with expected columns to prevent downstream errors
            # This also helps in caching a "no data" result rather than erroring out or returning None
            return pd.DataFrame(columns=["EPS Estimate", "Reported EPS"])
        return earnings_data

    def get_earnings_history(self, ticker: str) -> Union[pd.DataFrame, Dict[str, Any]]:
        """Get historical earnings data (dates, EPS estimate, reported EPS).

        Args:
            ticker: The stock ticker symbol.

        Returns:
            A DataFrame with earnings history or an error dictionary.
            The DataFrame index is usually the earnings date.
        """
        cache_key = f"earnings_history:{ticker}"
        cached_data = self._get_from_cache(cache_key)

        if cached_data:
            if (
                isinstance(cached_data, dict) and "error" in cached_data
            ):  # If error was cached
                logger.debug(
                    f"Retrieved error for earnings history for {ticker} from cache"
                )
                return cached_data
            if (
                isinstance(cached_data, list) and not cached_data
            ):  # Empty list for no data
                logger.debug(
                    f"Retrieved empty earnings history for {ticker} from cache (no data available)"
                )
                return pd.DataFrame(columns=["EPS Estimate", "Reported EPS"])
            logger.debug(f"Retrieved earnings history for {ticker} from cache")
            # Ensure data from cache is DataFrame (it's stored as list of dicts)
            df = pd.DataFrame(cached_data)
            if (
                "Earnings Date" in df.columns
            ):  # Assuming 'Earnings Date' was stored if index was date
                df = df.set_index("Earnings Date")
                df.index = pd.to_datetime(df.index)
            return df

        try:
            stock = yf.Ticker(ticker)
            # earnings_dates might return None or an empty DataFrame
            data = self._fetch_earnings_dates_with_retry(stock)

            if data is None or data.empty:
                logger.warning(
                    f"No earnings history data found for {ticker} after retries."
                )
                # Cache an empty list to signify no data, for 24 hours
                self._set_in_cache(cache_key, [], 86400)
                return pd.DataFrame(
                    columns=["EPS Estimate", "Reported EPS"]
                )  # Return empty DataFrame

            # Convert Timestamp objects in index to strings for JSON serialization if index is DatetimeIndex
            # And store the index as a column before converting to dict for caching
            if isinstance(data.index, pd.DatetimeIndex):
                data_to_cache = data.reset_index()
                # Convert datetime objects to ISO format strings
                date_column = data_to_cache.columns[
                    0
                ]  # Usually 'Earnings Date' or similar
                data_to_cache[date_column] = data_to_cache[date_column].dt.strftime(
                    "%Y-%m-%d %H:%M:%S%z"
                )
            else:  # If index is not datetime, just reset it
                data_to_cache = data.reset_index()

            # Convert to list of dicts for caching
            data_dict_list = data_to_cache.to_dict(orient="records")

            # Cache the data for 24 hours
            self._set_in_cache(cache_key, data_dict_list, 86400)

            logger.info(f"Retrieved earnings history for {ticker} from YFinance")
            return data

        except Exception as e:  # This will catch tenacity's RetryError or other issues
            logger.error(
                f"Error getting earnings history for {ticker} after retries: {str(e)}"
            )
            error_result = {"error": str(e), "timestamp": datetime.now().isoformat()}
            self._set_in_cache(cache_key, error_result, 3600)  # Cache error for 1 hour
            return error_result
