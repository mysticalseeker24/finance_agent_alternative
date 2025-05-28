"""API Agent for fetching market data using YFinance."""

from typing import Dict, List, Any, Optional
import asyncio
import aiohttp # Added

from loguru import logger

from agents.base_agent import BaseAgent
from data_ingestion.yfinance_client import YFinanceClient


class APIAgent(BaseAgent):
    """Agent for fetching market data from APIs."""

    def __init__(self):
        """Initialize the API agent."""
        super().__init__("API Agent")
        self.yfinance_client = YFinanceClient()
        logger.info("API Agent initialized with YFinance client")

    async def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a market data request.

        Args:
            request: The request containing operation and parameters.
                    Expected format:
                    {
                        "operation": "get_stock_info"|"get_historical_data"|"get_market_summary"|"get_earnings_calendar",
                        "parameters": {...}  # Operation-specific parameters
                    }

        Returns:
            The market data result.
        """
        operation = request.get("operation")
        parameters = request.get("parameters", {})

        if not operation:
            return {"error": "No operation specified"}

        # Use asyncio to run blocking YFinance operations in a thread pool
        if operation == "get_stock_info":
            ticker = parameters.get("ticker")
            if not ticker:
                return {"error": "No ticker specified for get_stock_info operation"}

            stock_info = await asyncio.to_thread(
                self.yfinance_client.get_stock_info, ticker
            )
            return {"data": stock_info}

        elif operation == "get_historical_data":
            ticker = parameters.get("ticker")
            period = parameters.get("period", "1mo")
            interval = parameters.get("interval", "1d")

            if not ticker:
                return {
                    "error": "No ticker specified for get_historical_data operation"
                }

            historical_data = await asyncio.to_thread(
                self.yfinance_client.get_historical_data, ticker, period, interval
            )

            # Convert DataFrame to list of dictionaries if it's a DataFrame
            if hasattr(historical_data, "to_dict"):
                historical_data = historical_data.to_dict(orient="records")

            return {"data": historical_data}

        elif operation == "get_market_summary":
            market_summary = await asyncio.to_thread(
                self.yfinance_client.get_market_summary
            )
            return {"data": market_summary}

        elif operation == "get_earnings_calendar":
            days = parameters.get("days", 7)
            earnings_calendar = await asyncio.to_thread(
                self.yfinance_client.get_earnings_calendar, days
            )
            return {"data": earnings_calendar}

        elif operation == "get_earnings_history":
            ticker = parameters.get("ticker")
            if not ticker:
                return {"error": "No ticker specified for get_earnings_history operation"}
            
            earnings_history_data = await asyncio.to_thread(
                self.yfinance_client.get_earnings_history, ticker
            )
            
            # Convert DataFrame to list of dictionaries if it's a DataFrame
            # yfinance_client.get_earnings_history already handles data conversion for cache
            # and should return a DataFrame or an error dict.
            if hasattr(earnings_history_data, "to_dict"):
                # If index is DatetimeIndex, reset it to include in dicts
                if hasattr(earnings_history_data.index, 'strftime'):
                    earnings_history_data_serializable = earnings_history_data.reset_index()
                    # Convert datetime objects in the new index column to string
                    date_column_name = earnings_history_data_serializable.columns[0] # typically 'index' or 'Earnings Date'
                    earnings_history_data_serializable[date_column_name] = earnings_history_data_serializable[date_column_name].dt.strftime('%Y-%m-%d')

                else: # If index is not datetime, just reset
                    earnings_history_data_serializable = earnings_history_data.reset_index()

                earnings_history_data = earnings_history_data_serializable.to_dict(orient="records")

            return {"data": earnings_history_data}

        elif operation == "fetch_generic_json":
            url = parameters.get("url")
            if not url:
                return {"error": "No URL specified for fetch_generic_json operation"}
            
            params = parameters.get("params")
            headers = parameters.get("headers")
            
            # Call the actual implementation method
            # The self.run() wrapper in BaseAgent will handle adding agent name, processing time etc.
            # So, the 'process' method should directly return what the core logic returns.
            return await self.fetch_generic_json_api(url, params, headers)

        else:
            return {"error": f"Unknown operation: {operation}"}

    async def fetch_generic_json_api(
        self, url: str, params: Optional[Dict] = None, headers: Optional[Dict] = None
    ) -> Dict[str, Any]:
        # This method will be wrapped by self.run() by the caller or by adding an operation
        # For direct calling, it would be:
        # request_details = {"operation": "fetch_generic_json", "parameters": {"url": url, "params": params, "headers": headers}}
        # return await self.run(request_details) 
        # However, to make it directly callable and also usable via 'process', we'll have it do the work
        # and the 'process' method will call this.

        logger.info(f"Fetching generic JSON API from URL: {url} with params: {params}")
        try:
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.get(url, params=params) as response:
                    response.raise_for_status()  # Raises an ClientResponseError for bad statuses (4xx or 5xx)
                    data = await response.json()
                    logger.info(f"Successfully fetched JSON data from {url}")
                    # The structure here should be consistent with how other methods return data for the process method
                    # Typically, the actual data is under a "data" key in the successful response.
                    return {"data": data, "url": url, "status_code": response.status}
        except aiohttp.ClientResponseError as e:
            logger.error(f"API Error fetching {url}: {e.status} - {e.message} - {e.headers}")
            return {"error": f"API Error: {e.status} - {e.message}", "url": url, "status_code": e.status}
        except aiohttp.ClientError as e: # More general client errors (e.g., connection error)
            logger.error(f"ClientError fetching {url}: {str(e)}")
            return {"error": f"ClientError: {str(e)}", "url": url, "status_code": "N/A"} # Status code might not be available
        except Exception as e: # Catch other exceptions like JSONDecodeError if response is not valid JSON
            logger.error(f"Generic error fetching {url}: {str(e)}")
            return {"error": f"Generic error: {str(e)}", "url": url, "status_code": "N/A"}


    async def get_stock_info(self, ticker: str) -> Dict[str, Any]:
        """Get stock information.

        Args:
            ticker: The stock ticker symbol.

        Returns:
            Stock information.
        """
        request = {"operation": "get_stock_info", "parameters": {"ticker": ticker}}
        return await self.run(request)

    async def get_historical_data(
        self, ticker: str, period: str = "1mo", interval: str = "1d"
    ) -> Dict[str, Any]:
        """Get historical stock data.

        Args:
            ticker: The stock ticker symbol.
            period: The period to fetch data for.
            interval: The data interval.

        Returns:
            Historical stock data.
        """
        request = {
            "operation": "get_historical_data",
            "parameters": {"ticker": ticker, "period": period, "interval": interval},
        }
        return await self.run(request)

    async def get_market_summary(self) -> Dict[str, Any]:
        """Get a summary of major market indices.

        Returns:
            Market summary data.
        """
        request = {"operation": "get_market_summary", "parameters": {}}
        return await self.run(request)

    async def get_earnings_calendar(self, days: int = 7) -> Dict[str, Any]:
        """Get upcoming earnings releases.

        Args:
            days: Number of days to look ahead.

        Returns:
            Earnings calendar data.
        """
        request = {"operation": "get_earnings_calendar", "parameters": {"days": days}}
        return await self.run(request)

    async def get_earnings_history(self, ticker: str) -> Dict[str, Any]:
        """Get historical earnings data for a stock.

        Args:
            ticker: The stock ticker symbol.

        Returns:
            Historical earnings data.
        """
        request = {"operation": "get_earnings_history", "parameters": {"ticker": ticker}}
        return await self.run(request)

    async def get_portfolio_data(
        self, portfolio: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Get data for a portfolio of stocks.

        Args:
            portfolio: List of portfolio items with ticker and weight.
                      Example: [{"ticker": "AAPL", "weight": 0.25}, ...]

        Returns:
            Portfolio data.
        """
        results = {}
        tasks = []

        # Create tasks for each stock in the portfolio
        for item in portfolio:
            ticker = item.get("ticker")
            if ticker:
                task = asyncio.create_task(self.get_stock_info(ticker))
                tasks.append((ticker, task))

        # Wait for all tasks to complete
        for ticker, task in tasks:
            result = await task
            results[ticker] = result.get("data", {"error": "Failed to fetch data"})

        return {"data": results, "portfolio": portfolio}
