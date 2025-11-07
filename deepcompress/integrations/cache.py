"""
Redis cache integration for document compression results.
"""

import pickle
from typing import Any

import redis.asyncio as aioredis
from tenacity import retry, stop_after_attempt, wait_exponential

from deepcompress.core.config import DeepCompressConfig
from deepcompress.exceptions import CacheError


class CacheManager:
    """
    Redis-based cache manager for compressed documents.

    Features:
    - Async Redis operations
    - Automatic TTL management
    - Pickle serialization for complex objects
    - Retry logic with exponential backoff
    """

    def __init__(self, config: DeepCompressConfig) -> None:
        self.config = config
        self._client: aioredis.Redis  or None = None
        self._connected = False

    async def connect(self) -> None:
        """Establish Redis connection."""
        if self._connected:
            return

        try:
            self._client = await aioredis.from_url(
                self.config.cache_url,
                encoding="utf-8",
                decode_responses=False,
            )
            await self._client.ping()
            self._connected = True
        except Exception as e:
            raise CacheError(
                "Failed to connect to Redis",
                details={"url": self.config.cache_url, "error": str(e)},
            )

    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._connected = False

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def get(self, key: str) -> Any  or None:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        if not self._connected:
            await self.connect()

        try:
            data = await self._client.get(key)
            if data:
                return pickle.loads(data)
            return None
        except Exception as e:
            raise CacheError(
                f"Failed to get key: {key}",
                details={"error": str(e)},
            )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def set(
        self,
        key: str,
        value: Any,
        ttl: int  or None = None,
    ) -> None:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses config default if None)
        """
        if not self._connected:
            await self.connect()

        ttl = ttl or self.config.cache_ttl

        try:
            data = pickle.dumps(value)
            await self._client.setex(key, ttl, data)
        except Exception as e:
            raise CacheError(
                f"Failed to set key: {key}",
                details={"error": str(e)},
            )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def delete(self, key: str) -> None:
        """
        Delete key from cache.

        Args:
            key: Cache key
        """
        if not self._connected:
            await self.connect()

        try:
            await self._client.delete(key)
        except Exception as e:
            raise CacheError(
                f"Failed to delete key: {key}",
                details={"error": str(e)},
            )

    async def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.

        Args:
            key: Cache key

        Returns:
            True if key exists
        """
        if not self._connected:
            await self.connect()

        try:
            return await self._client.exists(key) > 0
        except Exception as e:
            raise CacheError(
                f"Failed to check key existence: {key}",
                details={"error": str(e)},
            )

    async def clear(self) -> None:
        """Clear all keys from cache (use with caution)."""
        if not self._connected:
            await self.connect()

        try:
            await self._client.flushdb()
        except Exception as e:
            raise CacheError(
                "Failed to clear cache",
                details={"error": str(e)},
            )

    async def get_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        if not self._connected:
            await self.connect()

        try:
            info = await self._client.info("stats")
            return {
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
                "hit_rate": self._calculate_hit_rate(info),
                "keys": await self._client.dbsize(),
            }
        except Exception as e:
            raise CacheError(
                "Failed to get cache stats",
                details={"error": str(e)},
            )

    def _calculate_hit_rate(self, info: dict[str, Any]) -> float:
        """Calculate cache hit rate."""
        hits = info.get("keyspace_hits", 0)
        misses = info.get("keyspace_misses", 0)
        total = hits + misses
        return hits / total if total > 0 else 0.0

