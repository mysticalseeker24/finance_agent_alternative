"""Unit tests for the Retriever Agent."""

import unittest
import asyncio
from unittest import mock

import numpy as np
import pytest

from agents.retriever_agent import RetrieverAgent


class TestRetrieverAgent(unittest.TestCase):
    """Test cases for the Retriever Agent."""

    def setUp(self):
        """Set up test environment before each test."""
        self.agent = RetrieverAgent(index_name="test-index")

    def tearDown(self):
        """Clean up after each test."""
        self.agent = None

    def test_agent_initialization(self):
        """Test that the agent initializes with the correct parameters."""
        agent = RetrieverAgent(index_name="custom-index")
        self.assertEqual(agent.name, "Retriever Agent")
        self.assertEqual(agent.index_name, "custom-index")
        self.assertFalse(agent.initialized)

    @pytest.mark.asyncio
    @mock.patch("agents.retriever_agent.pinecone")
    @mock.patch("agents.retriever_agent.SentenceTransformer")
    async def test_initialize(self, mock_sentence_transformer, mock_pinecone):
        """Test the initialize method."""
        # Configure mocks
        mock_model = mock.MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sentence_transformer.return_value = mock_model

        mock_pinecone.list_indexes.return_value = []
        mock_index = mock.MagicMock()
        mock_pinecone.Index.return_value = mock_index

        # Call the method
        result = await self.agent.initialize()

        # Verify initialization
        self.assertTrue(result)
        self.assertTrue(self.agent.initialized)
        mock_sentence_transformer.assert_called_once_with("all-MiniLM-L6-v2")
        mock_pinecone.init.assert_called_once()
        mock_pinecone.list_indexes.assert_called_once()
        mock_pinecone.create_index.assert_called_once_with(
            name="test-index", dimension=384, metric="cosine"
        )
        mock_pinecone.Index.assert_called_once_with("test-index")

    @pytest.mark.asyncio
    @mock.patch.object(RetrieverAgent, "run")
    async def test_index_content(self, mock_run):
        """Test the index_content method."""
        # Set up mock return value
        mock_run.return_value = {
            "data": {"indexed_count": 2},
            "agent": "Retriever Agent",
            "processing_time": 0.5,
            "timestamp": "2025-01-01T00:00:00",
        }

        # Test data
        texts = ["This is a test document", "Another test document"]
        metadata = [{"source": "test1"}, {"source": "test2"}]
        namespace = "test-namespace"

        # Call the method
        result = await self.agent.index_content(texts, metadata, namespace)

        # Verify the method was called with the right parameters
        expected_request = {
            "operation": "index",
            "parameters": {
                "texts": texts,
                "metadata": metadata,
                "namespace": namespace,
            },
        }
        mock_run.assert_called_once_with(expected_request)

        # Verify the result
        self.assertEqual(result["data"]["indexed_count"], 2)

    @pytest.mark.asyncio
    @mock.patch.object(RetrieverAgent, "run")
    async def test_search_content(self, mock_run):
        """Test the search_content method."""
        # Set up mock return value
        mock_results = [
            {"id": "1", "score": 0.95, "metadata": {"text": "Test document 1"}},
            {"id": "2", "score": 0.85, "metadata": {"text": "Test document 2"}},
        ]
        mock_run.return_value = {
            "data": {"data": mock_results},
            "agent": "Retriever Agent",
            "processing_time": 0.3,
            "timestamp": "2025-01-01T00:00:00",
        }

        # Call the method
        result = await self.agent.search_content("test query", "test-namespace", 5)

        # Verify the method was called with the right parameters
        expected_request = {
            "operation": "search",
            "parameters": {
                "query": "test query",
                "namespace": "test-namespace",
                "top_k": 5,
            },
        }
        mock_run.assert_called_once_with(expected_request)

        # Verify the result
        self.assertEqual(len(result["data"]["data"]), 2)
        self.assertEqual(result["data"]["data"][0]["score"], 0.95)

    @pytest.mark.asyncio
    @mock.patch.object(RetrieverAgent, "_search")
    async def test_process_search(self, mock_search):
        """Test the process method with search operation."""
        # Configure the agent for testing
        self.agent.initialized = True
        self.agent.embedding_model = mock.MagicMock()

        # Set up mock return value
        mock_search.return_value = [
            {"id": "1", "score": 0.95, "text": "Test document 1"},
            {"id": "2", "score": 0.85, "text": "Test document 2"},
        ]

        # Create a search request
        request = {
            "operation": "search",
            "parameters": {
                "query": "test query",
                "namespace": "test-namespace",
                "top_k": 2,
            },
        }

        # Call the method
        result = await self.agent.process(request)

        # Verify the result
        self.assertEqual(result["data"], mock_search.return_value)
        mock_search.assert_called_once_with("test query", "test-namespace", 2)


# Allow running the tests from command line
if __name__ == "__main__":
    unittest.main()
