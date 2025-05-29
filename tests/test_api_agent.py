"""Unit tests for the API Agent."""

import asyncio
import unittest
from unittest import mock

import pandas as pd
import pytest

from agents.api_agent import APIAgent


class TestAPIAgent(unittest.TestCase):
    """Test cases for the API Agent."""

    def setUp(self):
        """Set up test environment before each test."""
        self.agent = APIAgent()

    def tearDown(self):
        """Clean up after each test."""
        self.agent = None

    @mock.patch("agents.api_agent.YFinanceClient")
    def test_agent_initialization(self, mock_yfinance_client):
        """Test that the agent initializes correctly."""
        agent = APIAgent()
        self.assertEqual(agent.name, "API Agent")
        mock_yfinance_client.assert_called_once()

    @pytest.mark.asyncio
    @mock.patch.object(APIAgent, "run")
    async def test_get_stock_info(self, mock_run):
        """Test the get_stock_info method."""
        # Set up the mock return value
        mock_run.return_value = {
            "data": {
                "symbol": "AAPL",
                "name": "Apple Inc.",
                "price": 150.0,
                "change_percent": 1.5,
                "timestamp": "2025-01-01T00:00:00",
            },
            "agent": "API Agent",
            "processing_time": 0.1,
            "timestamp": "2025-01-01T00:00:00",
        }

        # Call the method under test
        result = await self.agent.get_stock_info("AAPL")

        # Verify the method was called with the right parameters
        expected_request = {
            "operation": "get_stock_info",
            "parameters": {"ticker": "AAPL"},
        }
        mock_run.assert_called_once_with(expected_request)

        # Verify the result
        self.assertEqual(result["data"]["symbol"], "AAPL")
        self.assertEqual(result["data"]["name"], "Apple Inc.")

    @pytest.mark.asyncio
    @mock.patch.object(APIAgent, "run")
    async def test_get_market_summary(self, mock_run):
        """Test the get_market_summary method."""
        # Set up the mock return value
        mock_run.return_value = {
            "data": {
                "^GSPC": {"name": "S&P 500", "price": 4000.0, "change_percent": 0.5},
                "^DJI": {"name": "Dow Jones", "price": 33000.0, "change_percent": 0.3},
                "timestamp": "2025-01-01T00:00:00",
            },
            "agent": "API Agent",
            "processing_time": 0.2,
            "timestamp": "2025-01-01T00:00:00",
        }

        # Call the method under test
        result = await self.agent.get_market_summary()

        # Verify the method was called with the right parameters
        expected_request = {"operation": "get_market_summary", "parameters": {}}
        mock_run.assert_called_once_with(expected_request)

        # Verify the result
        self.assertIn("^GSPC", result["data"])
        self.assertEqual(result["data"]["^GSPC"]["name"], "S&P 500")

    @pytest.mark.asyncio
    @mock.patch.object(APIAgent, "run")
    async def test_get_portfolio_data(self, mock_run):
        """Test the get_portfolio_data method."""
        # Set up the portfolio and mocked responses
        portfolio = [
            {"ticker": "AAPL", "weight": 0.5},
            {"ticker": "MSFT", "weight": 0.5},
        ]

        # Mock the get_stock_info method to return predefined data
        self.agent.get_stock_info = mock.AsyncMock()
        self.agent.get_stock_info.side_effect = [
            {"data": {"symbol": "AAPL", "price": 150.0}},
            {"data": {"symbol": "MSFT", "price": 300.0}},
        ]

        # Call the method under test
        result = await self.agent.get_portfolio_data(portfolio)

        # Verify get_stock_info was called for each stock
        self.assertEqual(self.agent.get_stock_info.call_count, 2)
        self.agent.get_stock_info.assert_any_call("AAPL")
        self.agent.get_stock_info.assert_any_call("MSFT")

        # Verify the result contains data for both stocks
        self.assertIn("AAPL", result["data"])
        self.assertIn("MSFT", result["data"])
        self.assertEqual(result["portfolio"], portfolio)


# Allow running the tests from command line
if __name__ == "__main__":
    unittest.main()
