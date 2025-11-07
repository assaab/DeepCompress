"""
Prometheus metrics collection.
"""

from typing import Any

from prometheus_client import Counter, Gauge, Histogram, Summary


class MetricsCollector:
    """
    Prometheus metrics collector for EDC.

    Tracks:
    - Processing volume (documents, pages, tokens)
    - Latency (p50, p95, p99)
    - Error rates
    - Cost savings
    - Cache hit rates
    """

    def __init__(self) -> None:
        self.documents_processed = Counter(
            "edc_documents_processed_total",
            "Total documents processed",
            ["status"],
        )

        self.pages_processed = Counter(
            "edc_pages_processed_total",
            "Total pages processed",
        )

        self.tokens_saved = Counter(
            "edc_tokens_saved_total",
            "Total tokens saved",
        )

        self.cost_saved = Counter(
            "edc_cost_saved_usd_total",
            "Total cost saved in USD",
        )

        self.processing_latency = Histogram(
            "edc_processing_latency_seconds",
            "Document processing latency",
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        )

        self.compression_ratio = Summary(
            "edc_compression_ratio",
            "Token compression ratio",
        )

        self.cache_hit_rate = Gauge(
            "edc_cache_hit_rate",
            "Cache hit rate",
        )

        self.gpu_memory_usage = Gauge(
            "edc_gpu_memory_usage_bytes",
            "GPU memory usage",
            ["device"],
        )

        self.active_workers = Gauge(
            "edc_active_workers",
            "Number of active workers",
        )

    def record_document_processed(
        self,
        status: str,
        pages: int,
        tokens_saved: int,
        cost_saved: float,
        compression_ratio: float,
        latency_seconds: float,
    ) -> None:
        """
        Record document processing metrics.

        Args:
            status: Processing status (success, error)
            pages: Number of pages
            tokens_saved: Tokens saved
            cost_saved: Cost saved in USD
            compression_ratio: Compression ratio
            latency_seconds: Processing latency in seconds
        """
        self.documents_processed.labels(status=status).inc()
        self.pages_processed.inc(pages)
        self.tokens_saved.inc(tokens_saved)
        self.cost_saved.inc(cost_saved)
        self.compression_ratio.observe(compression_ratio)
        self.processing_latency.observe(latency_seconds)

    def update_cache_hit_rate(self, hit_rate: float) -> None:
        """Update cache hit rate metric."""
        self.cache_hit_rate.set(hit_rate)

    def update_gpu_memory(self, device: str, memory_bytes: int) -> None:
        """Update GPU memory usage metric."""
        self.gpu_memory_usage.labels(device=device).set(memory_bytes)

    def update_active_workers(self, count: int) -> None:
        """Update active worker count."""
        self.active_workers.set(count)

    def get_metrics(self) -> dict[str, Any]:
        """
        Get current metrics snapshot.

        Returns:
            Dictionary with metric values
        """
        return {
            "documents_processed": self.documents_processed._value.get(),
            "pages_processed": self.pages_processed._value.get(),
            "tokens_saved": self.tokens_saved._value.get(),
            "cost_saved_usd": self.cost_saved._value.get(),
            "cache_hit_rate": self.cache_hit_rate._value.get(),
            "active_workers": self.active_workers._value.get(),
        }

