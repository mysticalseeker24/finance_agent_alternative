"""Retriever Agent for managing vector store and RAG functionality."""

from typing import Dict, List, Any, Optional, Union
import json
import asyncio
from datetime import datetime
import hashlib # Added
import numpy as np # Added

import pinecone
from sentence_transformers import SentenceTransformer
import redis # Added
from loguru import logger

from agents.base_agent import BaseAgent
from config import Config


class RetrieverAgent(BaseAgent):
    """Agent for managing vector store and retrieval operations.
    
    Conceptual Usage of Secondary Embedding Model:
    If self.secondary_embedding_model is loaded, it could be used for:
    1. Re-ranking: Retrieve initial results with the primary model, then use the secondary
       model to calculate new similarity scores for these results and re-rank them.
    2. Query Expansion: Generate an alternative query embedding using the secondary model.
       This might be useful if the primary model struggles with certain query types.
       (Note: This would require a strategy to query the Pinecone index, especially if dimensions differ).
    3. Advanced RAG strategies: E.g., using one model for dense retrieval and another
       for different types of semantic matching if supported by a more complex retrieval flow.
    Actual implementation of these strategies is not currently in place but the model can be loaded.
    """

    def __init__(self, index_name: str = "finance-data"):
        """Initialize the Retriever agent.

        Args:
            index_name: Name of the Pinecone index to use.
        """
        super().__init__("Retriever Agent")
        self.index_name = index_name
        self.embedding_model = None
        self.embedding_model_name = Config.EMBEDDING_MODEL # Added
        self.secondary_embedding_model = None 
        self.secondary_model_name = None 
        self.index = None
        self.initialized = False
        
        # Initialize Redis cache
        try:
            self.cache = redis.Redis(
                host=Config.REDIS_HOST, 
                port=Config.REDIS_PORT, 
                decode_responses=False # Store bytes for embeddings
            )
            self.cache.ping() # Check connection
            logger.info(f"RetrieverAgent connected to Redis cache at {Config.REDIS_HOST}:{Config.REDIS_PORT}")
        except redis.exceptions.ConnectionError as e:
            logger.warning(f"RetrieverAgent failed to connect to Redis cache: {e}. Caching will be disabled.")
            self.cache = None
        
        logger.info("Retriever Agent initialized")

    async def _get_cached_embedding(self, text: str) -> Optional[np.ndarray]:
        if not self.cache:
            return None
        try:
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            cache_key = f"embedding:{self.embedding_model_name}:{text_hash}"
            cached_bytes = await asyncio.to_thread(self.cache.get, cache_key)
            if cached_bytes:
                embedding = np.frombuffer(cached_bytes, dtype=np.float32)
                logger.debug(f"Embedding cache hit for text: {text[:50]}...")
                return embedding
            logger.debug(f"Embedding cache miss for text: {text[:50]}...")
            return None
        except Exception as e:
            logger.warning(f"Error getting cached embedding for '{text[:50]}...': {e}")
            return None

    async def _cache_embedding(self, text: str, embedding: np.ndarray):
        if not self.cache:
            return
        try:
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            cache_key = f"embedding:{self.embedding_model_name}:{text_hash}"
            embedding_bytes = embedding.astype(np.float32).tobytes()
            # Cache for 7 days
            await asyncio.to_thread(self.cache.set, cache_key, embedding_bytes, ex=60 * 60 * 24 * 7)
            logger.debug(f"Cached embedding for text: {text[:50]}...")
        except Exception as e:
            logger.warning(f"Error caching embedding for '{text[:50]}...': {e}")


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

            # Initialize secondary embedding model if specified
            self.secondary_model_name = Config.SECONDARY_EMBEDDING_MODEL
            if self.secondary_model_name:
                try:
                    logger.info(f"Attempting to initialize secondary embedding model: {self.secondary_model_name}")
                    self.secondary_embedding_model = await asyncio.to_thread(
                        SentenceTransformer, self.secondary_model_name
                    )
                    secondary_dim = self.secondary_embedding_model.get_sentence_embedding_dimension()
                    logger.info(f"Initialized secondary embedding model: {self.secondary_model_name} (Dimensions: {secondary_dim})")
                    
                    primary_dim = self.embedding_model.get_sentence_embedding_dimension()
                    if secondary_dim != primary_dim:
                        logger.warning(
                            f"Secondary embedding model '{self.secondary_model_name}' has {secondary_dim} dimensions, "
                            f"which differs from the primary model's {primary_dim} dimensions. "
                            "It cannot be used to directly query the primary Pinecone index."
                        )
                except Exception as e:
                    logger.error(f"Failed to initialize secondary embedding model '{self.secondary_model_name}': {e}")
                    self.secondary_embedding_model = None # Ensure it's None if init fails

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
            # Process texts to get embeddings (cached or new)
            final_embeddings = []
            texts_to_encode_map = {} # To map original index to text needing encoding
            
            for i, text_content in enumerate(texts):
                cached_embedding = await self._get_cached_embedding(text_content)
                if cached_embedding is not None:
                    final_embeddings.append((i, cached_embedding)) # Store with original index
                else:
                    texts_to_encode_map[i] = text_content
            
            texts_to_encode_list = [texts_to_encode_map[i] for i in sorted(texts_to_encode_map.keys())]
            
            if texts_to_encode_list:
                logger.info(f"Generating embeddings for {len(texts_to_encode_list)} new texts out of {len(texts)} total.")
                new_embeddings_array = await asyncio.to_thread(self.embedding_model.encode, texts_to_encode_list)
                
                current_new_embedding_idx = 0
                for original_idx in sorted(texts_to_encode_map.keys()):
                    text_for_cache = texts_to_encode_map[original_idx]
                    new_embedding = new_embeddings_array[current_new_embedding_idx]
                    final_embeddings.append((original_idx, new_embedding))
                    await self._cache_embedding(text_for_cache, new_embedding)
                    current_new_embedding_idx += 1
            
            # Sort final_embeddings by original index to maintain order
            final_embeddings.sort(key=lambda x: x[0])
            ordered_embeddings_array = [emb for _, emb in final_embeddings]


            # Prepare vectors for upsert
            vectors = []
            for i, (text, embedding) in enumerate(zip(texts, ordered_embeddings_array)):
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
            # Generate embedding for the query, using cache
            query_embedding = await self._get_cached_embedding(query)
            if query_embedding is None:
                logger.debug(f"Query embedding cache miss for: {query[:50]}...")
                query_embedding_array = await asyncio.to_thread(
                    self.embedding_model.encode, query
                )
                # Ensure it's a numpy array even if encode returns a list (though it usually returns ndarray)
                query_embedding = np.array(query_embedding_array) if not isinstance(query_embedding_array, np.ndarray) else query_embedding_array
                await self._cache_embedding(query, query_embedding)
            else:
                logger.debug(f"Query embedding cache hit for: {query[:50]}...")


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
