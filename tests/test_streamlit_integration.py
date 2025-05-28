"""Integration tests for the Streamlit application."""

import unittest
from unittest import mock
import json

import pytest
import requests


class TestStreamlitIntegration(unittest.TestCase):
    """Integration tests for the Streamlit application's API interactions."""

    def setUp(self):
        """Set up test environment before each test."""
        # Mock base URL for API calls
        self.api_base_url = "http://localhost:8000"

    @mock.patch("requests.get")
    def test_market_summary_api_integration(self, mock_get):
        """Test the integration with the market summary API endpoint."""
        # Mock API response
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {
                "^GSPC": {"name": "S&P 500", "price": 4984.35, "change_percent": 0.25},
                "^DJI": {
                    "name": "Dow Jones",
                    "price": 38456.78,
                    "change_percent": -0.09,
                },
                "^IXIC": {
                    "name": "NASDAQ Composite",
                    "price": 16789.54,
                    "change_percent": 0.47,
                },
            },
            "agent": "API Agent",
            "processing_time": 0.152,
            "timestamp": "2025-05-28T14:30:05",
        }
        mock_get.return_value = mock_response

        # Simulate the API call made by Streamlit
        from streamlit_app.app import fetch_market_summary

        summary = fetch_market_summary(self.api_base_url)

        # Verify the request was made correctly
        mock_get.assert_called_once_with(
            f"{self.api_base_url}/api/v1/market/summary",
            headers={"Content-Type": "application/json"},
            timeout=10,
        )

        # Verify the data was processed correctly
        self.assertEqual(len(summary), 3)
        self.assertEqual(summary["^GSPC"]["price"], 4984.35)
        self.assertEqual(summary["^DJI"]["change_percent"], -0.09)

    @mock.patch("requests.post")
    def test_query_processing_api_integration(self, mock_post):
        """Test the integration with the query processing API endpoint."""
        # Mock API response
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {
                "query": "What is the current price of Apple stock?",
                "response": "Apple Inc. (AAPL) is currently trading at $180.95, up 0.69% today.",
                "sources": [
                    {"type": "market_data", "description": "Current stock price"}
                ],
                "audio_url": None,
            },
            "agent": "Orchestrator",
            "processing_time": 1.567,
            "timestamp": "2025-05-28T14:30:40",
        }
        mock_post.return_value = mock_response

        # Simulate the API call made by Streamlit
        from streamlit_app.app import process_query

        query = "What is the current price of Apple stock?"
        use_voice = False
        result = process_query(self.api_base_url, query, use_voice)

        # Verify the request was made correctly
        mock_post.assert_called_once_with(
            f"{self.api_base_url}/api/v1/query",
            headers={"Content-Type": "application/json"},
            json={"query": query, "use_voice": use_voice},
            timeout=30,
        )

        # Verify the data was processed correctly
        self.assertEqual(
            result["response"],
            "Apple Inc. (AAPL) is currently trading at $180.95, up 0.69% today.",
        )
        self.assertIsNone(result["audio_url"])

    @mock.patch("requests.post")
    def test_market_brief_api_integration(self, mock_post):
        """Test the integration with the market brief API endpoint."""
        # Mock API response
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {
                "brief": {
                    "market_summary": "Good morning. Here's your market brief for Wednesday, May 28, 2025.",
                    "key_events": "Federal Reserve minutes released yesterday indicate officials are considering rate cuts.",
                    "portfolio_update": "Your portfolio is up 0.38% today, outperforming the S&P 500 by 0.13%.",
                    "outlook": "Market sentiment remains cautiously optimistic with expectations for Fed rate cuts.",
                },
                "audio_url": "/api/v1/audio/brief_20250528_081500.mp3",
            },
            "agent": "Orchestrator",
            "processing_time": 3.245,
            "timestamp": "2025-05-28T14:31:00",
        }
        mock_post.return_value = mock_response

        # Simulate the API call made by Streamlit
        from streamlit_app.app import generate_market_brief

        portfolio = [
            {"ticker": "AAPL", "weight": 0.5},
            {"ticker": "MSFT", "weight": 0.5},
        ]
        use_voice = True
        brief = generate_market_brief(self.api_base_url, portfolio, use_voice)

        # Verify the request was made correctly
        mock_post.assert_called_once_with(
            f"{self.api_base_url}/api/v1/market/brief",
            headers={"Content-Type": "application/json"},
            json={
                "include_portfolio": True,
                "portfolio": portfolio,
                "use_voice": use_voice,
            },
            timeout=60,
        )

        # Verify the data was processed correctly
        self.assertIn("market_summary", brief["brief"])
        self.assertIn("key_events", brief["brief"])
        self.assertIn("portfolio_update", brief["brief"])
        self.assertEqual(brief["audio_url"], "/api/v1/audio/brief_20250528_081500.mp3")

    @mock.patch("requests.post")
    def test_voice_processing_api_integration(self, mock_post):
        """Test the integration with the voice processing API endpoints."""
        # Mock API response for text-to-speech
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {
                "audio_url": "/api/v1/audio/tts_20250528_143115.mp3",
                "duration_seconds": 4.5,
            },
            "agent": "Voice Agent",
            "processing_time": 0.987,
            "timestamp": "2025-05-28T14:31:15",
        }
        mock_post.return_value = mock_response

        # Simulate the API call made by Streamlit
        from streamlit_app.app import convert_text_to_speech

        text = "Apple stock is currently trading at $180.95, up 0.69% today."
        audio_url = convert_text_to_speech(self.api_base_url, text)

        # Verify the request was made correctly
        mock_post.assert_called_once_with(
            f"{self.api_base_url}/api/v1/voice/tts",
            headers={"Content-Type": "application/json"},
            json={"text": text, "voice_id": "default"},
            timeout=30,
        )

        # Verify the data was processed correctly
        self.assertEqual(audio_url, "/api/v1/audio/tts_20250528_143115.mp3")


# Allow running the tests from command line
if __name__ == "__main__":
    unittest.main()
