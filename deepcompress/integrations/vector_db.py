"""
Vector database integrations (Pinecone, Weaviate).
"""

from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from deepcompress.core.config import DeepCompressConfig
from deepcompress.exceptions import VectorDBError


class VectorDBClient:
    """
    Multi-provider vector database client.

    Supports:
    - Pinecone (serverless)
    - Weaviate (cloud/self-hosted)
    """

    def __init__(self, config: DeepCompressConfig) -> None:
        self.config = config
        self.provider = config.vector_db_provider
        self._client: Any = None
        self._index: Any = None

    async def initialize(self) -> None:
        """Initialize vector DB client."""
        if self.provider == "pinecone":
            await self._init_pinecone()
        elif self.provider == "weaviate":
            await self._init_weaviate()
        elif self.provider == "none":
            return
        else:
            raise VectorDBError(
                f"Unsupported vector DB provider: {self.provider}",
                details={"supported": ["pinecone", "weaviate", "none"]},
            )

    async def _init_pinecone(self) -> None:
        """Initialize Pinecone client."""
        try:
            from pinecone import Pinecone

            self._client = Pinecone(api_key=self.config.vector_db_api_key)

            if self.config.vector_db_index_name not in self._client.list_indexes().names():
                self._client.create_index(
                    name=self.config.vector_db_index_name,
                    dimension=1536,
                    metric="cosine",
                    spec={
                        "serverless": {
                            "cloud": "aws",
                            "region": self.config.vector_db_environment,
                        }
                    },
                )

            self._index = self._client.Index(self.config.vector_db_index_name)

        except ImportError:
            raise VectorDBError(
                "Pinecone not installed. Install with: pip install edc[vector-db]"
            )

    async def _init_weaviate(self) -> None:
        """Initialize Weaviate client."""
        try:
            import weaviate

            self._client = weaviate.Client(
                url=self.config.vector_db_environment,
                auth_client_secret=weaviate.AuthApiKey(
                    api_key=self.config.vector_db_api_key
                ),
            )

        except ImportError:
            raise VectorDBError(
                "Weaviate not installed. Install with: pip install edc[vector-db]"
            )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def upsert(
        self,
        document_id: str,
        embedding: list[float],
        metadata: dict[str, Any],
    ) -> None:
        """
        Upsert document embedding to vector DB.

        Args:
            document_id: Unique document ID
            embedding: Embedding vector (1536-dim for OpenAI)
            metadata: Document metadata
        """
        if self._client is None:
            await self.initialize()

        try:
            if self.provider == "pinecone":
                await self._upsert_pinecone(document_id, embedding, metadata)
            elif self.provider == "weaviate":
                await self._upsert_weaviate(document_id, embedding, metadata)

        except Exception as e:
            raise VectorDBError(
                f"Failed to upsert document: {document_id}",
                details={"error": str(e)},
            )

    async def _upsert_pinecone(
        self,
        document_id: str,
        embedding: list[float],
        metadata: dict[str, Any],
    ) -> None:
        """Upsert to Pinecone."""
        self._index.upsert(
            vectors=[
                {
                    "id": document_id,
                    "values": embedding,
                    "metadata": metadata,
                }
            ]
        )

    async def _upsert_weaviate(
        self,
        document_id: str,
        embedding: list[float],
        metadata: dict[str, Any],
    ) -> None:
        """Upsert to Weaviate."""
        self._client.data_object.create(
            data_object=metadata,
            class_name="Document",
            uuid=document_id,
            vector=embedding,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def query(
        self,
        embedding: list[float],
        top_k: int = 10,
        filters: dict[str, Any]  or None = None,
    ) -> list[dict[str, Any]]:
        """
        Query similar documents.

        Args:
            embedding: Query embedding vector
            top_k: Number of results to return
            filters: Optional metadata filters

        Returns:
            List of similar documents with scores
        """
        if self._client is None:
            await self.initialize()

        try:
            if self.provider == "pinecone":
                return await self._query_pinecone(embedding, top_k, filters)
            elif self.provider == "weaviate":
                return await self._query_weaviate(embedding, top_k, filters)
            else:
                return []

        except Exception as e:
            raise VectorDBError(
                "Failed to query vector DB",
                details={"error": str(e)},
            )

    async def _query_pinecone(
        self,
        embedding: list[float],
        top_k: int,
        filters: dict[str, Any]  or None,
    ) -> list[dict[str, Any]]:
        """Query Pinecone."""
        results = self._index.query(
            vector=embedding,
            top_k=top_k,
            filter=filters,
            include_metadata=True,
        )

        return [
            {
                "id": match.id,
                "score": match.score,
                "metadata": match.metadata,
            }
            for match in results.matches
        ]

    async def _query_weaviate(
        self,
        embedding: list[float],
        top_k: int,
        filters: dict[str, Any]  or None,
    ) -> list[dict[str, Any]]:
        """Query Weaviate."""
        query = self._client.query.get("Document", ["*"]).with_near_vector(
            {"vector": embedding}
        ).with_limit(top_k)

        if filters:
            query = query.with_where(filters)

        results = query.do()

        return [
            {
                "id": item["_additional"]["id"],
                "score": item["_additional"]["distance"],
                "metadata": item,
            }
            for item in results["data"]["Get"]["Document"]
        ]

    async def delete(self, document_id: str) -> None:
        """
        Delete document from vector DB.

        Args:
            document_id: Document ID to delete
        """
        if self._client is None:
            await self.initialize()

        try:
            if self.provider == "pinecone":
                self._index.delete(ids=[document_id])
            elif self.provider == "weaviate":
                self._client.data_object.delete(document_id, class_name="Document")

        except Exception as e:
            raise VectorDBError(
                f"Failed to delete document: {document_id}",
                details={"error": str(e)},
            )

