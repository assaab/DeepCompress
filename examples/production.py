"""
Production Deployment Example

Full production workflow with vector DB indexing and LLM analysis.
"""

import asyncio
from typing import Any

from deepcompress import compress_and_analyze
from deepcompress.core.config import DeepCompressConfig
from deepcompress.integrations.cache import CacheManager
from deepcompress.integrations.llm import LLMClient
from deepcompress.integrations.vector_db import VectorDBClient
from deepcompress.utils.logging import get_logger, setup_logging


async def get_pending_applications() -> list[dict[str, Any]]:
    """
    Fetch pending loan applications from database.

    Returns:
        List of application dictionaries
    """
    return [
        {"id": "app_001", "s3_path": "s3://loan-docs/app_001.pdf"},
        {"id": "app_002", "s3_path": "s3://loan-docs/app_002.pdf"},
    ]


async def update_application_risk_score(
    app_id: str,
    result: Any,
) -> None:
    """
    Update application risk score in database.

    Args:
        app_id: Application ID
        result: Compression result with LLM analysis
    """
    print(f"Updated risk score for {app_id}")


async def process_all_applications() -> None:
    """Process all pending applications."""
    setup_logging(level="INFO", structured=True)
    logger = get_logger("deepcompress.production")

    logger.info("Starting production batch processing")

    config = DeepCompressConfig()
    cache_manager = CacheManager(config)
    vector_db = VectorDBClient(config)
    llm_client = LLMClient(config.llm_provider, config)

    await cache_manager.connect()
    await vector_db.initialize()
    await llm_client.initialize()

    logger.info("Services initialized")

    applications = await get_pending_applications()
    logger.info(f"Processing {len(applications)} applications")

    query = """
    Extract and analyze:
    1. Total monthly income from all sources
    2. Total monthly debt obligations
    3. Debt-to-income ratio
    4. Employment stability
    5. Credit risk factors
    6. Recommend approval/rejection with confidence score
    """

    tasks = []
    for app in applications:
        task = compress_and_analyze(
            file=app["s3_path"],
            query=query,
            llm=config.llm_provider,
            cache=True,
            scrub_pii=True,
            config=config,
        )
        tasks.append((app["id"], task))

    for app_id, task in tasks:
        try:
            trace_id = logger.set_trace_id()
            logger.info(
                "Processing application",
                app_id=app_id,
                trace_id=trace_id,
            )

            result = await task

            embedding = await llm_client.embed(result.optimized_text)
            await vector_db.upsert(
                document_id=result.document_id,
                embedding=embedding,
                metadata={
                    "application_id": app_id,
                    "compressed_text": result.optimized_text,
                    "original_tokens": result.original_tokens,
                    "compressed_tokens": result.compressed_tokens,
                    "processing_date": "2024-11-07",
                },
            )

            await update_application_risk_score(app_id, result)

            logger.info(
                "Application processed successfully",
                app_id=app_id,
                tokens_saved=result.tokens_saved,
                cost_saved=result.cost_saved_usd,
            )

        except Exception as e:
            logger.error(
                "Application processing failed",
                app_id=app_id,
                error=str(e),
            )

    cache_stats = await cache_manager.get_stats()
    logger.info(
        "Batch processing complete",
        total_applications=len(applications),
        cache_hit_rate=cache_stats["hit_rate"],
    )

    await cache_manager.disconnect()


if __name__ == "__main__":
    asyncio.run(process_all_applications())

