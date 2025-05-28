"""YFinance client for fetching market data."""

import time
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta

import yfinance as yf
import pandas as pd
import redis
from loguru import logger

from config import Config


class YFinanceClient:
    """Client for fetching market data from Yahoo Finance."""
    
    def __init__(self):
        """Initialize the YFinance client."""
        self.cache = redis.Redis(
            host=Config.REDIS_HOST,
            port=Config.REDIS_PORT,
            decode_responses=True,
            socket_timeout=5,
        )
        logger.info("YFinance client initialized")
    
    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get data from cache.
        
        Args:
            key: The cache key.
            
        Returns:
            The cached data if available, None otherwise.
        """
        try:
            data = self.cache.get(key)
            if data:
                import json
                return json.loads(data)
            return None
        except redis.RedisError as e:
            logger.warning(f"Redis error when getting from cache: {e}")
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
            import json
            self.cache.set(key, json.dumps(value), ex=expiry)
            return True
        except redis.RedisError as e:
            logger.warning(f"Redis error when setting in cache: {e}")
            return False
    
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
            info = stock.info
            
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
        
        except Exception as e:
            logger.error(f"Error getting stock info for {ticker}: {str(e)}")
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
            data = stock.history(period=period, interval=interval)
            
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
        
        except Exception as e:
            logger.error(f"Error getting historical data for {ticker}: {str(e)}")
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
            "^GSPC",    # S&P 500
            "^DJI",     # Dow Jones Industrial Average
            "^IXIC",    # NASDAQ Composite
            "^FTSE",    # FTSE 100
            "^N225",    # Nikkei 225
            "^HSI",     # Hang Seng Index
            "^NSEI",    # NIFTY 50
        ]
        
        try:
            result = {}
            for index in indices:
                index_data = yf.Ticker(index).info
                result[index] = {
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
        
        except Exception as e:
            logger.error(f"Error getting market summary: {str(e)}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}
    
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
            logger.error(f"Error getting earnings calendar: {str(e)}")
            return [{"error": str(e), "timestamp": datetime.now().isoformat()}]
