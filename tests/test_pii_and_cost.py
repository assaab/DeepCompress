from deepcompress.processing.pii import PIIScrubber
from deepcompress.utils.cost import calculate_cost_per_document, calculate_savings


def test_pii_scrubber_detects_and_scrubs_defaults():
    text = "Email ada@example.com, phone (555) 123-4567, SSN 123-45-6789."
    scrubber = PIIScrubber()

    detected = scrubber.detect(text)
    scrubbed = scrubber.scrub(text)

    assert detected["email"] == ["ada@example.com"]
    assert detected["phone"] == ["(555) 123-4567"]
    assert detected["ssn"] == ["123-45-6789"]
    assert "[REDACTED_EMAIL]" in scrubbed
    assert "[REDACTED_PHONE]" in scrubbed
    assert "[REDACTED_SSN]" in scrubbed


def test_cost_calculators_return_positive_savings():
    savings = calculate_savings(
        pages_per_month=100,
        avg_tokens_per_page=1000,
        gpu_cost_per_month=0,
    )
    per_document = calculate_cost_per_document(pages=2, tokens_per_page=1000)

    assert savings["original_tokens"] == 100000
    assert savings["monthly_savings"] > 0
    assert savings["compression_ratio_toon"] == 12.5
    assert per_document["cost_without_edc"] > per_document["cost_with_edc_toon"]
