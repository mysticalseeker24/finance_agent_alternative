"""Retriever Agent for managing vector store and RAG functionality."""

from typing import Dict, List, Any, Optional, Union
import json
import asyncio
from datetime import datetime

import pinecone
from sentence_transformers import SentenceTransformer
from loguru import logger

from agents.base_agent import BaseAgent
from config import Config


class RetrieverAgent(BaseAgent):
    """Agent for managing vector store and retrieval operations."""

    def __init__(self, index_name: str = "finance-data"):
        """Initialize the Retriever agent.

        Args:
            index_name: Name of the Pinecone index to use.
        """
        super().__init__("Retriever Agent")
        self.index_name = index_name
        self.embedding_model = None
        self.index = None
        self.initialized = False
        logger.info("Retriever Agent initialized")

    async def initialize(self) -> bool:
        """Initialize the Pinecone client and embedding model.

        Returns:
            True if initialization is successful, False otherwise.
        """
        try:
            # Initialize embedding model
            model_name = Config.EMBEDDING_MODEL
            self.embedding_model = await asyncio.to_thread(
                SentenceTransformer, model_name
            )
            logger.info(f"Initialized embedding model: {model_name}")

            # Initialize Pinecone
            api_key = Config.PINECONE_API_KEY
            environment = Config.PINECONE_ENVIRONMENT

            if not api_key or not environment:
                logger.error("Pinecone API key or environment not set")
                return False

            await asyncio.to_thread(
                pinecone.init, api_key=api_key, environment=environment
            )
            logger.info("Initialized Pinecone client")

            # Check if index exists, create if it doesn't
            indexes = await asyncio.to_thread(pinecone.list_indexes)

            if self.index_name not in indexes:
                logger.info(f"Creating Pinecone index: {self.index_name}")
                # Specify the dimension based on the embedding model
                embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
                await asyncio.to_thread(
                    pinecone.create_index,
                    name=self.index_name,
                    dimension=embedding_dim,
                    metric="cosine",
                )

            # Connect to the index
            self.index = await asyncio.to_thread(pinecone.Index, self.index_name)
            logger.info(f"Connected to Pinecone index: {self.index_name}")

            self.initialized = True
            return True

        except Exception as e:
            logger.error(f"Error initializing Retriever Agent: {str(e)}")
            return False

    async def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a retrieval request.

        Args:
            request: The request containing operation and parameters.
                    Expected format:
                    {
                        "operation": "index"|"search"|"delete"|"stats",
                        "parameters": {...}  # Operation-specific parameters
                    }

        Returns:
            The retrieval result.
        """
        # Ensure the agent is initialized
        if not self.initialized:
            success = await self.initialize()
            if not success:
                return {"error": "Failed to initialize Retriever Agent"}

        operation = request.get("operation")
        parameters = request.get("parameters", {})

        if not operation:
            return {"error": "No operation specified"}

        # Execute the requested operation
        if operation == "index":
            texts = parameters.get("texts", [])
            metadata_list = parameters.get("metadata", [])
            namespace = parameters.get("namespace", "default")

            if not texts:
                return {"error": "No texts provided for indexing"}

            if metadata_list and len(texts) != len(metadata_list):
                return {"error": "Number of texts and metadata items must match"}

            result = await self._index_texts(texts, metadata_list, namespace)
            return {"data": result}

        elif operation == "search":
            query = parameters.get("query")
            namespace = parameters.get("namespace", "default")
            top_k = parameters.get("top_k", 5)

            if not query:
                return {"error": "No query provided for search"}

            results = await self._search(query, namespace, top_k)
            return {"data": results}

        elif operation == "delete":
            ids = parameters.get("ids", [])
            namespace = parameters.get("namespace", "default")
            delete_all = parameters.get("delete_all", False)

            if not ids and not delete_all:
                return {"error": "No ids provided for deletion and delete_all is False"}

            success = await self._delete(ids, namespace, delete_all)
            return {"success": success}

        elif operation == "stats":
            result = await self._get_stats()
            return {"data": result}

        else:
            return {"error": f"Unknown operation: {operation}"}

    async def _index_texts(
        self,
        texts: List[str],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
        namespace: str = "default",
    ) -> Dict[str, Any]:
        """Index texts in the vector store.

        Args:
            texts: List of texts to index.
            metadata_list: List of metadata dictionaries for each text.
            namespace: The namespace to use.

        Returns:
            Result of the indexing operation.
        """
        try:
            # Generate embeddings for the texts
            embeddings = await asyncio.to_thread(self.embedding_model.encode, texts)

            # Prepare vectors for upsert
            vectors = []
            for i, (text, embedding) in enumerate(zip(texts, embeddings)):
                # Create a unique ID for the vector
                vector_id = f"vec_{datetime.now().timestamp()}_{i}"

                # Prepare metadata
                metadata = {"text": text}
                if metadata_list and i < len(metadata_list):
                    metadata.update(metadata_list[i])

                # Add vector to the list
                vectors.append(
                    {
                        "id": vector_id,
                        "values": embedding.tolist(),
                        "metadata": metadata,
                    }
                )

            # Upsert vectors in batches (Pinecone has limits on batch size)
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i : i + batch_size]
                await asyncio.to_thread(
                    self.index.upsert, vectors=batch, namespace=namespace
                )

            logger.info(f"Indexed {len(texts)} texts in namespace '{namespace}'")
            return {"indexed_count": len(texts)}

        except Exception as e:
            logger.error(f"Error indexing texts: {str(e)}")
            return {"error": str(e)}

    async def _search(
        self, query: str, namespace: str = "default", top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search the vector store for relevant content.

        Args:
            query: The query text.
            namespace: The namespace to search in.
            top_k: Number of results to return.

        Returns:
            List of search results with metadata and scores.
        """
        try:
            # Generate embedding for the query
            query_embedding = await asyncio.to_thread(
                self.embedding_model.encode, query
            )

            # Search the index
            results = await asyncio.to_thread(
                self.index.query,
                namespace=namespace,
                top_k=top_k,
                include_metadata=True,
                vector=query_embedding.tolist(),
            )

            # Process and format the results
            formatted_results = []
            for match in results.get("matches", []):
                formatted_results.append(
                    {
                        "id": match.get("id"),
                        "score": match.get("score"),
                        "metadata": match.get("metadata", {}),
                        "text": match.get("metadata", {}).get("text", ""),
                    }
                )

            logger.info(f"Found {len(formatted_results)} results for query: {query}")
            return formatted_results

        except Exception as e:
            logger.error(f"Error searching: {str(e)}")
            return [{"error": str(e)}]

    async def _delete(
        self,
        ids: List[str] = None,
        namespace: str = "default",
        delete_all: bool = False,
    ) -> bool:
        """Delete vectors from the index.

        Args:
            ids: List of vector IDs to delete.
            namespace: The namespace to delete from.
            delete_all: Whether to delete all vectors in the namespace.

        Returns:
            True if successful, False otherwise.
        """
        try:
            if delete_all:
                await asyncio.to_thread(
                    self.index.delete, delete_all=True, namespace=namespace
                )
                logger.info(f"Deleted all vectors in namespace '{namespace}'")
            else:
                await asyncio.to_thread(self.index.delete, ids=ids, namespace=namespace)
                logger.info(f"Deleted {len(ids)} vectors in namespace '{namespace}'")

            return True

        except Exception as e:
            logger.error(f"Error deleting vectors: {str(e)}")
            return False

    async def _get_stats(self) -> Dict[str, Any]:
        """Get statistics for the index.

        Returns:
            Statistics for the index.
        """
        try:
            stats = await asyncio.to_thread(self.index.describe_index_stats)
            logger.info(f"Retrieved stats for index {self.index_name}")
            return stats

        except Exception as e:
            logger.error(f"Error getting index stats: {str(e)}")
            return {"error": str(e)}

    async def index_content(
        self,
        texts: List[str],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
        namespace: str = "default",
    ) -> Dict[str, Any]:
        """Index content in the vector store.

        Args:
            texts: List of texts to index.
            metadata_list: List of metadata dictionaries for each text.
            namespace: The namespace to use.

        Returns:
            Result of the indexing operation.
        """
        request = {
            "operation": "index",
            "parameters": {
                "texts": texts,
                "metadata": metadata_list,
                "namespace": namespace,
            },
        }
        return await self.run(request)

    async def search_content(
        self, query: str, namespace: str = "default", top_k: int = 5
    ) -> Dict[str, Any]:
        """Search for relevant content.

        Args:
            query: The query text.
            namespace: The namespace to search in.
            top_k: Number of results to return.

        Returns:
            Search results.
        """
        request = {
            "operation": "search",
            "parameters": {"query": query, "namespace": namespace, "top_k": top_k},
        }
        return await self.run(request)

    async def delete_content(
        self,
        ids: List[str] = None,
        namespace: str = "default",
        delete_all: bool = False,
    ) -> Dict[str, Any]:
        """Delete content from the vector store.

        Args:
            ids: List of vector IDs to delete.
            namespace: The namespace to delete from.
            delete_all: Whether to delete all vectors in the namespace.

        Returns:
            Result of the delete operation.
        """
        request = {
            "operation": "delete",
            "parameters": {
                "ids": ids,
                "namespace": namespace,
                "delete_all": delete_all,
            },
        }
        return await self.run(request)

    async def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics for the index.

        Returns:
            Index statistics.
        """
        request = {"operation": "stats", "parameters": {}}
        return await self.run(request)
