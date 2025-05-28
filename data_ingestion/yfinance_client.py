"""YFinance client for fetching market data."""

import time
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta

import yfinance as yf
import pandas as pd
import redis
from loguru import logger
import json # Added for cache handling
import requests # To catch requests.exceptions.RequestException
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config import Config
from data_ingestion.firecrawl_scraper import FirecrawlScraper # Added import


class YFinanceClient:
    """Client for fetching market data from Yahoo Finance."""

    def __init__(self):
        """Initialize the YFinance client."""
        self.cache = redis.Redis(
            host=Config.REDIS_HOST,
            port=Config.REDIS_PORT,
            decode_responses=True, # Keep True for direct string operations
            socket_timeout=5,
        )
        self.firecrawl_scraper = FirecrawlScraper() # Added FirecrawlScraper instance
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
            logger.warning(f"JSON decode error when getting from cache for key {key}: {e}")
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
        except TypeError as e: # Handle cases where value is not JSON serializable
            logger.error(f"TypeError when setting cache for key {key}: {e}. Value: {value}")
            return False

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10), # Exponential backoff: 2s, 4s, 8s ... up to 10s
        stop=stop_after_attempt(3), # Retry up to 3 times
        retry=retry_if_exception_type((requests.exceptions.RequestException, ConnectionError)) # Retry on specific network errors
    )
    def _fetch_stock_info_with_retry(self, ticker_obj):
        # Helper to encapsulate the part that needs retrying
        logger.debug(f"Fetching stock info for {ticker_obj.ticker} (attempt {self._fetch_stock_info_with_retry.retry.statistics.get('attempt_number', 1)})")
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
            info = self._fetch_stock_info_with_retry(stock) # New call via retry helper

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

        except Exception as e: # This will catch tenacity's RetryError if all retries fail
            logger.error(f"Error getting stock info for {ticker} after retries: {str(e)}")
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
            data = self._fetch_historical_data_with_retry(stock, period, interval) # New call

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

        except Exception as e: # This will catch tenacity's RetryError
            logger.error(f"Error getting historical data for {ticker} after retries: {str(e)}")
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
                index_data = self._fetch_stock_info_with_retry(stock_obj) # Reusing helper

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

        except Exception as e: # This will catch tenacity's RetryError
            logger.error(f"Error getting market summary after retries on one of the indices: {str(e)}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((requests.exceptions.RequestException, ConnectionError))
    )
    def _fetch_historical_data_with_retry(self, ticker_obj, period, interval):
        # Helper to encapsulate the part that needs retrying
        logger.debug(f"Fetching historical data for {ticker_obj.ticker} (period: {period}, interval: {interval}, attempt: {self._fetch_historical_data_with_retry.retry.statistics.get('attempt_number', 1)})")
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
            logger.error(f"Error getting earnings calendar: {str(e)}") # This refers to the live scraped one
            return [{"error": str(e), "timestamp": datetime.now().isoformat()}]

    def _parse_yahoo_earnings_page(
        self, earnings_rows: Optional[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Parse scraped Yahoo Finance earnings data.

        Args:
            earnings_rows: A list of dictionaries, where each dictionary represents a row
                           of earnings data scraped by FirecrawlScraper. Expected keys
                           within each dict are based on the selectors used (e.g., 'symbol',
                           'company_name', 'earnings_date', 'earnings_time', 'eps_estimate').

        Returns:
            A list of parsed earnings event dictionaries.
        """
        if not earnings_rows:
            logger.warning("No earnings rows data provided to _parse_yahoo_earnings_page.")
            return []

        parsed_events = []
        for idx, row in enumerate(earnings_rows):
            if not isinstance(row, dict):
                logger.warning(f"Skipping row {idx} in earnings data as it's not a dictionary: {row}")
                continue
            try:
                symbol = row.get("symbol", "N/A")
                company_name = row.get("company_name", "N/A")
                
                # Date parsing and normalization
                raw_date = row.get("earnings_date")
                earnings_date_str = "N/A"
                if raw_date:
                    try:
                        # Yahoo earnings calendar often has dates like "May 23, 2024" or "Tomorrow"
                        # This needs robust parsing. For now, assume it might be directly usable or needs simple parsing.
                        # A more robust solution would use dateutil.parser
                        # For simplicity, if it contains " AM" or " PM" (time info), strip it for date.
                        if isinstance(raw_date, str):
                            if " AM" in raw_date or " PM" in raw_date: # Handles cases like "May 23 AM"
                                raw_date_parts = raw_date.split(" ")
                                if len(raw_date_parts) > 1: # e.g. "May 23"
                                     # This part is tricky without knowing the exact format from selectors.
                                     # Assuming for now the selector gives a date string that might need parsing
                                     # Example: "May 23" -> needs year.
                                     # Let's assume for now the selector gives something more direct or already processed.
                                     # If the date selector is specific enough, it might give "YYYY-MM-DD" or "Mon DD, YYYY"
                                     # For now, we'll assume it's a string that might be parsable.
                                     # This is a placeholder for more robust date parsing.
                                     # A common format on Yahoo is like "May 23, 2024".
                                    pass # Needs more context on actual raw_date format from scraper
                        
                        # Assuming raw_date is in a format like "Month Day, Year" or can be parsed
                        # This is a simplification. Real parsing would be more complex.
                        # If Firecrawl selectors give clean text, this becomes easier.
                        # For now, we'll just use it as is if it's a string.
                        if isinstance(raw_date, str):
                            # Attempt to parse common Yahoo date format, e.g., "May 23, 2024"
                            # Or it might be something like "Today", "Tomorrow"
                            # This part is highly dependent on the actual text scraped.
                            # For now, let's assume it's a string we can use or placeholder.
                            # A better approach would be to parse it into a datetime object and reformat.
                            # Example: datetime.strptime(raw_date, "%b %d, %Y").strftime("%Y-%m-%d")
                            # But this requires knowing the exact format from scraping.
                            earnings_date_str = raw_date # Placeholder - requires actual scraped format
                        elif isinstance(raw_date, dict) and 'date' in raw_date: # if scraper returns structured date
                            earnings_date_str = raw_date['date']
                        else: # Fallback
                            earnings_date_str = str(raw_date) if raw_date else "N/A"

                    except Exception as date_e:
                        logger.warning(f"Could not parse date for row {idx}: {raw_date}. Error: {date_e}")
                        earnings_date_str = "N/A"
                
                earnings_time = row.get("earnings_time", "Time Not Supplied")
                if not earnings_time: earnings_time = "Time Not Supplied" # Ensure it's not empty

                eps_estimate_raw = row.get("eps_estimate")
                eps_estimate = None
                if eps_estimate_raw and eps_estimate_raw not in ["-", "N/A"]:
                    try:
                        eps_estimate = float(eps_estimate_raw)
                    except ValueError:
                        logger.warning(f"Could not parse EPS estimate '{eps_estimate_raw}' for {symbol}")
                        eps_estimate = None

                # Reported EPS and Market Cap are typically not on the main calendar page per row
                # Defaulting them.
                event = {
                    "symbol": symbol,
                    "company_name": company_name,
                    "earnings_date": earnings_date_str, # This needs to be YYYY-MM-DD
                    "earnings_time": earnings_time,
                    "eps_estimate": eps_estimate,
                    "reported_eps": None, # Usually N/A for future events
                    "market_cap": row.get("market_cap"), # If available from selectors
                    "source": "Yahoo Finance (via Firecrawl)"
                }
                parsed_events.append(event)
            except Exception as e:
                logger.error(f"Error parsing earnings row {idx}: {row}. Error: {str(e)}")
                continue
        
        logger.info(f"Successfully parsed {len(parsed_events)} earnings events.")
        return parsed_events

    def get_earnings_calendar(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get upcoming earnings releases by scraping Yahoo Finance.

        Args:
            days: Number of days to look ahead. Note: Current scraping implementation
                  fetches the default view of Yahoo's calendar (often current week)
                  and may not precisely honor this 'days' parameter without
                  more complex date-based navigation during scraping.

        Returns:
            A list of dictionaries containing earnings release information.
        """
        cache_key = f"live_earnings_calendar:default_view_approx_{days}d" # Cache key reflects default view
        cached_data = self._get_from_cache(cache_key)

        if cached_data:
            logger.debug(f"Retrieved earnings calendar (default view) from cache. Requested days: {days}")
            return cached_data

        logger.info(f"Fetching live earnings calendar. Requested days: {days}. "
                    "Note: Scraping fetches Yahoo's default calendar view (typically current week).")
        
        EARNINGS_CALENDAR_URL = "https://finance.yahoo.com/calendar/earnings"
        
        # These selectors need to be robust and verified against Yahoo's current structure.
        # Assuming FirecrawlScraper's scrape_url returns a dictionary where keys are
        # the selector names, and values are the extracted data (e.g., list of strings or dicts).
        earnings_selectors = {
            "rows": "div[data-test='cal-table'] ul li", # Top-level selector for each earnings event row
            # Relative selectors within each "row" element. Firecrawl should handle this.
            # If Firecrawl returns structured data per row, these are keys in that structure.
            # If Firecrawl returns flat lists, parsing logic needs to map them.
            # Assuming the former: each item in "rows" is a dict with these keys.
            "symbol": "a[data-test='symbol']",
            "company_name": "span[aria-label='Company']",
            "earnings_date": "span[aria-label='Earnings Date']", # This text needs careful parsing
            "earnings_time": "span[aria-label='Earnings Call Time']", # e.g., "Before Market", "After Market"
            "eps_estimate": "span[aria-label='EPS Estimate']",
            # "market_cap": "span[aria-label='Market Cap']", # If available and desired
        }

        try:
            scraped_page_data = self.firecrawl_scraper.scrape_url(
                url=EARNINGS_CALENDAR_URL, 
                selectors=earnings_selectors
            )

            # Check for errors from FirecrawlScraper itself or if data is missing
            if not scraped_page_data or scraped_page_data.get("error") or scraped_page_data.get("success") is False:
                error_msg = scraped_page_data.get('error', 'No data returned or success=false')
                logger.error(f"Failed to scrape earnings calendar: {error_msg}")
                # Cache the error indication to prevent rapid retries
                self._set_in_cache(cache_key, [{"error": "Scraping failed.", "details": error_msg}], 300) # Cache error for 5 mins
                return [{"error": "Failed to retrieve earnings calendar data via scraping.", "details": error_msg, "timestamp": datetime.now().isoformat()}]

            # The actual data from Firecrawl based on selectors.
            # If "rows" selector worked, this should be a list of dicts.
            # Each dict corresponds to one <li> element, with keys like "symbol", "company_name" etc.
            earnings_data_rows = scraped_page_data.get("rows") 

            if not earnings_data_rows:
                logger.warning(f"No 'rows' data found in scraped earnings calendar from {EARNINGS_CALENDAR_URL}. Scraped data: {scraped_page_data}")
                # Cache an empty list to prevent re-scraping on immediate retries for a short period
                self._set_in_cache(cache_key, [], 300) # Cache empty result for 5 mins
                return []

            parsed_earnings = self._parse_yahoo_earnings_page(earnings_data_rows)

            if not parsed_earnings:
                 logger.warning("No earnings events parsed from scraped data, though rows were present.")
                 self._set_in_cache(cache_key, [], 300) # Cache empty result for 5 mins
                 return []

            self._set_in_cache(cache_key, parsed_earnings, 21600)  # Cache for 6 hours
            logger.info(f"Retrieved and parsed live earnings calendar with {len(parsed_earnings)} events.")
            return parsed_earnings

        except Exception as e:
            logger.exception(f"General error in get_earnings_calendar: {str(e)}") # Use logger.exception for stack trace
            # Cache the error indication
            self._set_in_cache(cache_key, [{"error": str(e)}], 300)
            return [{"error": str(e), "timestamp": datetime.now().isoformat()}]

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((requests.exceptions.RequestException, ConnectionError))
    )
    def _fetch_earnings_dates_with_retry(self, ticker_obj):
        # Helper to encapsulate the yfinance call that needs retrying
        logger.debug(f"Fetching earnings dates for {ticker_obj.ticker} (attempt {self._fetch_earnings_dates_with_retry.retry.statistics.get('attempt_number', 1)})")
        earnings_data = ticker_obj.earnings_dates
        if earnings_data is None or earnings_data.empty:
            logger.warning(f"No earnings dates data returned by yfinance for {ticker_obj.ticker}")
            # Return an empty DataFrame with expected columns to prevent downstream errors
            # This also helps in caching a "no data" result rather than erroring out or returning None
            return pd.DataFrame(columns=['EPS Estimate', 'Reported EPS']) 
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
            if isinstance(cached_data, dict) and "error" in cached_data: # If error was cached
                 logger.debug(f"Retrieved error for earnings history for {ticker} from cache")
                 return cached_data
            if isinstance(cached_data, list) and not cached_data: # Empty list for no data
                logger.debug(f"Retrieved empty earnings history for {ticker} from cache (no data available)")
                return pd.DataFrame(columns=['EPS Estimate', 'Reported EPS'])
            logger.debug(f"Retrieved earnings history for {ticker} from cache")
            # Ensure data from cache is DataFrame (it's stored as list of dicts)
            df = pd.DataFrame(cached_data)
            if 'Earnings Date' in df.columns: # Assuming 'Earnings Date' was stored if index was date
                df = df.set_index('Earnings Date')
                df.index = pd.to_datetime(df.index)
            return df


        try:
            stock = yf.Ticker(ticker)
            # earnings_dates might return None or an empty DataFrame
            data = self._fetch_earnings_dates_with_retry(stock)

            if data is None or data.empty:
                logger.warning(f"No earnings history data found for {ticker} after retries.")
                # Cache an empty list to signify no data, for 24 hours
                self._set_in_cache(cache_key, [], 86400) 
                return pd.DataFrame(columns=['EPS Estimate', 'Reported EPS']) # Return empty DataFrame

            # Convert Timestamp objects in index to strings for JSON serialization if index is DatetimeIndex
            # And store the index as a column before converting to dict for caching
            if isinstance(data.index, pd.DatetimeIndex):
                data_to_cache = data.reset_index()
                # Convert datetime objects to ISO format strings
                date_column = data_to_cache.columns[0] # Usually 'Earnings Date' or similar
                data_to_cache[date_column] = data_to_cache[date_column].dt.strftime('%Y-%m-%d %H:%M:%S%z')
            else: # If index is not datetime, just reset it
                data_to_cache = data.reset_index()

            # Convert to list of dicts for caching
            data_dict_list = data_to_cache.to_dict(orient="records")
            
            # Cache the data for 24 hours
            self._set_in_cache(cache_key, data_dict_list, 86400)

            logger.info(f"Retrieved earnings history for {ticker} from YFinance")
            return data

        except Exception as e: # This will catch tenacity's RetryError or other issues
            logger.error(f"Error getting earnings history for {ticker} after retries: {str(e)}")
            error_result = {"error": str(e), "timestamp": datetime.now().isoformat()}
            self._set_in_cache(cache_key, error_result, 3600) # Cache error for 1 hour
            return error_result
