"""Integration tests for the Orchestrator component."""

import unittest
import asyncio
from unittest import mock

import pytest

from orchestrator.orchestrator import Orchestrator
from agents.api_agent import APIAgent
from agents.retriever_agent import RetrieverAgent
from agents.language_agent import LanguageAgent


class TestOrchestrator(unittest.TestCase):
    """Integration tests for the Orchestrator component."""

    def setUp(self):
        """Set up test environment before each test."""
        self.orchestrator = Orchestrator()

    def tearDown(self):
        """Clean up after each test."""
        self.orchestrator = None

    def test_orchestrator_initialization(self):
        """Test that the orchestrator initializes correctly with all agents."""
        self.assertEqual(len(self.orchestrator.agents), 6)
        agent_names = [agent.name for agent in self.orchestrator.agents.values()]
        expected_agents = [
            "API Agent",
            "Scraping Agent",
            "Retriever Agent",
            "Analysis Agent",
            "Language Agent",
            "Voice Agent",
        ]
        for agent in expected_agents:
            self.assertIn(agent, agent_names)

    @pytest.mark.asyncio
    @mock.patch.object(APIAgent, "get_stock_info")
    @mock.patch.object(LanguageAgent, "generate_response")
    async def test_process_stock_query(
        self, mock_generate_response, mock_get_stock_info
    ):
        """Test processing a stock query through the orchestrator."""
        # Set up mocks
        mock_get_stock_info.return_value = {
            "data": {
                "symbol": "AAPL",
                "name": "Apple Inc.",
                "price": 180.95,
                "change_percent": 0.69,
            }
        }

        mock_generate_response.return_value = {
            "data": {
                "response": "Apple Inc. (AAPL) is currently trading at $180.95, up 0.69% today."
            }
        }

        # Process a query
        query = "What is the current price of Apple stock?"
        result = await self.orchestrator.process_query(query)

        # Verify API agent was called correctly
        mock_get_stock_info.assert_called_once_with("AAPL")

        # Verify Language agent was called to generate a response
        self.assertTrue(mock_generate_response.called)

        # Verify the result contains the expected response
        self.assertIn("Apple", result["data"]["response"])
        self.assertIn("$180.95", result["data"]["response"])

    @pytest.mark.asyncio
    @mock.patch.object(APIAgent, "get_market_summary")
    @mock.patch.object(APIAgent, "get_portfolio_data")
    @mock.patch.object(LanguageAgent, "generate_narrative")
    async def test_generate_market_brief(
        self, mock_generate_narrative, mock_get_portfolio_data, mock_get_market_summary
    ):
        """Test generating a market brief through the orchestrator."""
        # Set up mocks
        mock_get_market_summary.return_value = {
            "data": {
                "^GSPC": {"name": "S&P 500", "price": 4984.35, "change_percent": 0.25},
                "^DJI": {
                    "name": "Dow Jones",
                    "price": 38456.78,
                    "change_percent": -0.09,
                },
            }
        }

        mock_get_portfolio_data.return_value = {
            "data": {
                "AAPL": {"price": 180.95, "change_percent": 0.69},
                "MSFT": {"price": 350.45, "change_percent": 1.25},
            },
            "portfolio": [
                {"ticker": "AAPL", "weight": 0.5},
                {"ticker": "MSFT", "weight": 0.5},
            ],
        }

        mock_generate_narrative.return_value = {
            "data": {
                "narrative": "Good morning. Here's your market brief for today. The S&P 500 is up 0.25% at 4,984.35, while the Dow Jones is down slightly by 0.09% at 38,456.78. Your portfolio is up 0.97% today, outperforming the broader market."
            }
        }

        # Generate a market brief
        portfolio = [
            {"ticker": "AAPL", "weight": 0.5},
            {"ticker": "MSFT", "weight": 0.5},
        ]

        result = await self.orchestrator.generate_market_brief(portfolio)

        # Verify API agent methods were called correctly
        mock_get_market_summary.assert_called_once()
        mock_get_portfolio_data.assert_called_once_with(portfolio)

        # Verify Language agent was called to generate a narrative
        self.assertTrue(mock_generate_narrative.called)

        # Verify the result contains the expected narrative
        self.assertIn("market brief", result["data"]["narrative"])
        self.assertIn("S&P 500", result["data"]["narrative"])
        self.assertIn("portfolio", result["data"]["narrative"])

    @pytest.mark.asyncio
    @mock.patch.object(RetrieverAgent, "search_content")
    @mock.patch.object(APIAgent, "get_stock_info")
    @mock.patch.object(LanguageAgent, "generate_response")
    async def test_complex_query_with_retrieval(
        self, mock_generate_response, mock_get_stock_info, mock_search_content
    ):
        """Test processing a complex query requiring vector retrieval."""
        # Set up mocks
        mock_search_content.return_value = {
            "data": {
                "data": [
                    {
                        "id": "1",
                        "score": 0.95,
                        "metadata": {
                            "text": "Apple reported strong Q2 2025 earnings, with revenue of $94.8 billion."
                        },
                    }
                ]
            }
        }

        mock_get_stock_info.return_value = {
            "data": {
                "symbol": "AAPL",
                "name": "Apple Inc.",
                "price": 180.95,
                "change_percent": 0.69,
                "pe_ratio": 29.76,
            }
        }

        mock_generate_response.return_value = {
            "data": {
                "response": "Apple Inc. (AAPL) is currently trading at $180.95, up 0.69% today. The company has a P/E ratio of 29.76, which is above the sector average. In their latest earnings report, Apple reported strong Q2 2025 results with revenue of $94.8 billion."
            }
        }

        # Process a complex query
        query = "What are Apple's current financials and recent earnings?"
        result = await self.orchestrator.process_query(query)

        # Verify Retriever agent was called to search for content
        mock_search_content.assert_called_once()

        # Verify API agent was called to get stock information
        mock_get_stock_info.assert_called_once_with("AAPL")

        # Verify Language agent was called with combined context
        self.assertTrue(mock_generate_response.called)

        # Verify the result contains information from both sources
        self.assertIn("$180.95", result["data"]["response"])
        self.assertIn("P/E ratio", result["data"]["response"])
        self.assertIn("$94.8 billion", result["data"]["response"])


# Allow running the tests from command line
if __name__ == "__main__":
    unittest.main()
