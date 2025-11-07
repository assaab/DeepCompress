"""
CLI tools for DeepCompress.
"""

import argparse
import sys

from deepcompress.utils.cost import calculate_savings


def calculate_savings_cli() -> None:
    """CLI for cost savings calculator."""
    parser = argparse.ArgumentParser(
        description="Calculate DeepCompress cost savings and ROI"
    )
    parser.add_argument(
        "--pages",
        type=int,
        required=True,
        help="Pages per month",
    )
    parser.add_argument(
        "--tokens-per-page",
        type=int,
        default=5000,
        help="Average tokens per page (default: 5000)",
    )
    parser.add_argument(
        "--llm",
        type=str,
        default="gpt-4o",
        choices=["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo", "claude-3-opus", "claude-3-sonnet"],
        help="Target LLM (default: gpt-4o)",
    )
    parser.add_argument(
        "--gpu-cost",
        type=float,
        default=4000.0,
        help="Monthly GPU cost (default: $4000)",
    )

    args = parser.parse_args()

    result = calculate_savings(
        pages_per_month=args.pages,
        avg_tokens_per_page=args.tokens_per_page,
        target_llm=args.llm,
        gpu_cost_per_month=args.gpu_cost,
    )

    print("\n" + "=" * 60)
    print("DEEPCOMPRESS COST SAVINGS ANALYSIS")
    print("=" * 60)
    print(f"\nInput:")
    print(f"  Pages per month:        {result['pages_per_month']:,}")
    print(f"  Tokens per page:        {args.tokens_per_page:,}")
    print(f"  Target LLM:             {args.llm}")
    print(f"\nCost Breakdown:")
    print(f"  Without DeepCompress:            ${result['cost_without_deepcompress']:,.2f}/month")
    print(f"  With DeepCompress (OCR only):    ${result['cost_with_deepcompress_ocr']:,.2f}/month")
    print(f"  With DeepCompress (OCR + TOON):  ${result['cost_with_deepcompress_toon']:,.2f}/month")
    print(f"  GPU Infrastructure:              ${result['gpu_cost']:,.2f}/month")
    print(f"  Total with DeepCompress:         ${result['total_cost_with_deepcompress']:,.2f}/month")
    print(f"\nSavings:")
    print(f"  Monthly savings:        ${result['monthly_savings']:,.2f}")
    print(f"  Annual savings:         ${result['annual_savings']:,.2f}")
    print(f"  Payback period:         {result['payback_months']:.1f} months")
    print(f"  3-year ROI:             {result['three_year_roi_percent']:.0f}%")
    print(f"\nCompression:")
    print(f"  OCR compression:        {result['compression_ratio_ocr']:.1f}x")
    print(f"  TOON compression:       {result['compression_ratio_toon']:.1f}x")
    print(f"  Original tokens:        {result['original_tokens']:,}")
    print(f"  Compressed (TOON):      {result['compressed_tokens_toon']:,}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "calculate-savings":
        calculate_savings_cli()
    else:
        print("Available commands: calculate-savings")
        sys.exit(1)

