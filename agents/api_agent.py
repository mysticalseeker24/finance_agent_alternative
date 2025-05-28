"""API Agent for fetching market data using YFinance."""

from typing import Dict, List, Any, Optional
import asyncio

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
            
            stock_info = await asyncio.to_thread(self.yfinance_client.get_stock_info, ticker)
            return {"data": stock_info}
        
        elif operation == "get_historical_data":
            ticker = parameters.get("ticker")
            period = parameters.get("period", "1mo")
            interval = parameters.get("interval", "1d")
            
            if not ticker:
                return {"error": "No ticker specified for get_historical_data operation"}
            
            historical_data = await asyncio.to_thread(
                self.yfinance_client.get_historical_data,
                ticker,
                period,
                interval
            )
            
            # Convert DataFrame to list of dictionaries if it's a DataFrame
            if hasattr(historical_data, "to_dict"):
                historical_data = historical_data.to_dict(orient="records")
            
            return {"data": historical_data}
        
        elif operation == "get_market_summary":
            market_summary = await asyncio.to_thread(self.yfinance_client.get_market_summary)
            return {"data": market_summary}
        
        elif operation == "get_earnings_calendar":
            days = parameters.get("days", 7)
            earnings_calendar = await asyncio.to_thread(self.yfinance_client.get_earnings_calendar, days)
            return {"data": earnings_calendar}
        
        else:
            return {"error": f"Unknown operation: {operation}"}
    
    async def get_stock_info(self, ticker: str) -> Dict[str, Any]:
        """Get stock information.
        
        Args:
            ticker: The stock ticker symbol.
            
        Returns:
            Stock information.
        """
        request = {
            "operation": "get_stock_info",
            "parameters": {"ticker": ticker}
        }
        return await self.run(request)
    
    async def get_historical_data(self, ticker: str, period: str = "1mo", interval: str = "1d") -> Dict[str, Any]:
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
            "parameters": {"ticker": ticker, "period": period, "interval": interval}
        }
        return await self.run(request)
    
    async def get_market_summary(self) -> Dict[str, Any]:
        """Get a summary of major market indices.
        
        Returns:
            Market summary data.
        """
        request = {
            "operation": "get_market_summary",
            "parameters": {}
        }
        return await self.run(request)
    
    async def get_earnings_calendar(self, days: int = 7) -> Dict[str, Any]:
        """Get upcoming earnings releases.
        
        Args:
            days: Number of days to look ahead.
            
        Returns:
            Earnings calendar data.
        """
        request = {
            "operation": "get_earnings_calendar",
            "parameters": {"days": days}
        }
        return await self.run(request)
    
    async def get_portfolio_data(self, portfolio: List[Dict[str, Any]]) -> Dict[str, Any]:
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
        
        return {
            "data": results,
            "portfolio": portfolio
        }
