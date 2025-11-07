"""
DeepCompress worker for distributed processing.
"""

import argparse
import asyncio
import signal
from typing import Any

from prometheus_client import start_http_server

from deepcompress.core.compressor import DocumentCompressor
from deepcompress.core.config import DeepCompressConfig
from deepcompress.integrations.cache import CacheManager
from deepcompress.utils.logging import get_logger, setup_logging
from deepcompress.utils.metrics import MetricsCollector


class DeepCompressWorker:
    """
    DeepCompress worker for processing documents.

    Features:
    - Health checks
    - Graceful shutdown
    - Metrics export
    - GPU monitoring
    """

    def __init__(self, config: DeepCompressConfig) -> None:
        self.config = config
        self.compressor = DocumentCompressor(config)
        self.cache_manager = CacheManager(config) if config.cache_enabled else None
        self.metrics = MetricsCollector()
        self.logger = get_logger("deepcompress.worker", config.log_level)
        self.running = False

    async def start(self) -> None:
        """Start worker."""
        self.logger.info("Starting DeepCompress worker", device=self.config.ocr_device)

        start_http_server(self.config.prometheus_port)
        self.logger.info(
            "Prometheus metrics available",
            port=self.config.prometheus_port,
        )

        if self.cache_manager:
            await self.cache_manager.connect()
            self.logger.info("Connected to cache", url=self.config.cache_url)

        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        self.running = True
        self.metrics.update_active_workers(1)

        self.logger.info("Worker started successfully")

        while self.running:
            await asyncio.sleep(1)

    async def stop(self) -> None:
        """Stop worker."""
        self.logger.info("Stopping EDC worker")

        self.running = False
        self.metrics.update_active_workers(0)

        if self.cache_manager:
            await self.cache_manager.disconnect()

        self.logger.info("Worker stopped")

    def _handle_shutdown(self, signum: int, frame: Any) -> None:
        """Handle shutdown signal."""
        self.logger.info("Received shutdown signal", signal=signum)
        asyncio.create_task(self.stop())

    async def health_check(self) -> dict[str, Any]:
        """
        Perform health check.

        Returns:
            Health status dictionary
        """
        status = {
            "status": "healthy" if self.running else "stopped",
            "device": self.config.ocr_device,
            "cache_enabled": self.config.cache_enabled,
        }

        if self.cache_manager:
            try:
                stats = await self.cache_manager.get_stats()
                status["cache_status"] = "connected"
                status["cache_hit_rate"] = stats["hit_rate"]
            except Exception as e:
                status["cache_status"] = "error"
                status["cache_error"] = str(e)

        try:
            import torch

            if torch.cuda.is_available():
                device_id = int(self.config.ocr_device.split(":")[-1])
                status["gpu_available"] = True
                status["gpu_memory_allocated"] = torch.cuda.memory_allocated(device_id)
                status["gpu_memory_reserved"] = torch.cuda.memory_reserved(device_id)
            else:
                status["gpu_available"] = False
        except ImportError:
            status["gpu_available"] = False

        return status


def main() -> None:
    """Main entry point for worker."""
    parser = argparse.ArgumentParser(description="DeepCompress Worker")
    parser.add_argument(
        "--port",
        type=int,
        default=9090,
        help="Prometheus metrics port",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for OCR processing",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level",
    )

    args = parser.parse_args()

    config = DeepCompressConfig()
    config.prometheus_port = args.port
    config.ocr_device = args.device
    config.log_level = args.log_level

    setup_logging(level=config.log_level, structured=config.structured_logging)

    worker = DeepCompressWorker(config)

    try:
        asyncio.run(worker.start())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger = get_logger("deepcompress.worker")
        logger.error("Worker failed", error=str(e))
        raise


if __name__ == "__main__":
    main()

