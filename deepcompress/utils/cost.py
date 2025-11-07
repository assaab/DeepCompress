"""
Cost calculation and ROI utilities.
"""

from typing import Any


def calculate_savings(
    pages_per_month: int,
    avg_tokens_per_page: int = 5000,
    target_llm: str = "gpt-4o",
    gpu_cost_per_month: float = 4000.0,
) -> dict[str, Any]:
    """
    Calculate cost savings and ROI for EDC.

    Args:
        pages_per_month: Number of pages processed per month
        avg_tokens_per_page: Average tokens per page before compression
        target_llm: Target LLM for cost calculation
        gpu_cost_per_month: Monthly GPU infrastructure cost

    Returns:
        Dictionary with cost analysis

    Example:
        >>> savings = calculate_savings(
        ...     pages_per_month=250000,
        ...     avg_tokens_per_page=5000,
        ...     target_llm="gpt-4o"
        ... )
        >>> print(f"Monthly savings: ${savings['monthly_savings']:.2f}")
        >>> print(f"ROI: {savings['roi_percent']:.0f}%")
    """
    pricing = _get_llm_pricing(target_llm)

    original_tokens = pages_per_month * avg_tokens_per_page

    compressed_tokens_ocr = pages_per_month * 200
    compressed_tokens_toon = pages_per_month * 80

    cost_without_edc = _calculate_llm_cost(original_tokens, pricing)

    cost_with_edc_ocr = _calculate_llm_cost(compressed_tokens_ocr, pricing)
    cost_with_edc_toon = _calculate_llm_cost(compressed_tokens_toon, pricing)

    total_cost_with_edc = cost_with_edc_toon + gpu_cost_per_month

    monthly_savings = cost_without_edc - total_cost_with_edc
    annual_savings = monthly_savings * 12

    one_time_costs = 30000 + 80000
    payback_months = one_time_costs / monthly_savings if monthly_savings > 0 else 0

    three_year_roi = (annual_savings * 3 - one_time_costs) / one_time_costs * 100

    return {
        "pages_per_month": pages_per_month,
        "original_tokens": original_tokens,
        "compressed_tokens_ocr": compressed_tokens_ocr,
        "compressed_tokens_toon": compressed_tokens_toon,
        "cost_without_edc": cost_without_edc,
        "cost_with_edc_ocr": cost_with_edc_ocr,
        "cost_with_edc_toon": cost_with_edc_toon,
        "gpu_cost": gpu_cost_per_month,
        "total_cost_with_edc": total_cost_with_edc,
        "monthly_savings": monthly_savings,
        "annual_savings": annual_savings,
        "payback_months": payback_months,
        "three_year_roi_percent": three_year_roi,
        "compression_ratio_ocr": avg_tokens_per_page / 200,
        "compression_ratio_toon": avg_tokens_per_page / 80,
    }


def _get_llm_pricing(llm: str) -> dict[str, float]:
    """
    Get LLM pricing per 1K tokens.

    Returns:
        Dictionary with input/output pricing
    """
    pricing_map = {
        "gpt-4o": {
            "input": 0.005,
            "output": 0.015,
        },
        "gpt-4-turbo": {
            "input": 0.01,
            "output": 0.03,
        },
        "gpt-3.5-turbo": {
            "input": 0.0005,
            "output": 0.0015,
        },
        "claude-3-opus": {
            "input": 0.015,
            "output": 0.075,
        },
        "claude-3-sonnet": {
            "input": 0.003,
            "output": 0.015,
        },
        "claude-3-haiku": {
            "input": 0.00025,
            "output": 0.00125,
        },
    }

    return pricing_map.get(llm, pricing_map["gpt-4o"])


def _calculate_llm_cost(tokens: int, pricing: dict[str, float]) -> float:
    """
    Calculate LLM cost for token count.

    Assumes 2x input usage (embedding + query) and 10% output tokens.

    Args:
        tokens: Total token count
        pricing: Pricing dictionary

    Returns:
        Total cost in USD
    """
    input_cost = (tokens / 1000) * pricing["input"] * 2

    output_tokens = tokens * 0.1
    output_cost = (output_tokens / 1000) * pricing["output"]

    return input_cost + output_cost


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text.

    Uses approximation: 1 token = 4 characters.

    Args:
        text: Text to estimate

    Returns:
        Estimated token count
    """
    return len(text) // 4


def calculate_cost_per_document(
    pages: int,
    tokens_per_page: int = 5000,
    llm: str = "gpt-4o",
) -> dict[str, float]:
    """
    Calculate cost per document.

    Args:
        pages: Number of pages
        tokens_per_page: Tokens per page
        llm: LLM provider

    Returns:
        Cost breakdown dictionary
    """
    pricing = _get_llm_pricing(llm)

    original_tokens = pages * tokens_per_page
    compressed_tokens_ocr = pages * 200
    compressed_tokens_toon = pages * 80

    cost_without_edc = _calculate_llm_cost(original_tokens, pricing)
    cost_with_edc_ocr = _calculate_llm_cost(compressed_tokens_ocr, pricing)
    cost_with_edc_toon = _calculate_llm_cost(compressed_tokens_toon, pricing)

    gpu_cost_per_page = 0.0007

    return {
        "cost_without_edc": cost_without_edc,
        "cost_with_edc_ocr": cost_with_edc_ocr,
        "cost_with_edc_toon": cost_with_edc_toon + (pages * gpu_cost_per_page),
        "savings_ocr": cost_without_edc - cost_with_edc_ocr,
        "savings_toon": cost_without_edc - (cost_with_edc_toon + pages * gpu_cost_per_page),
    }

