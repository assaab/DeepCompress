"""Document processing modules."""

__all__ = [
    "BatchProcessor",
    "PIIScrubber",
    "extract_protected_facts",
]


def __getattr__(name: str):
    if name == "BatchProcessor":
        from deepcompress.processing.batch import BatchProcessor

        return BatchProcessor
    if name == "PIIScrubber":
        from deepcompress.processing.pii import PIIScrubber

        return PIIScrubber
    if name == "extract_protected_facts":
        from deepcompress.processing.protected_facts import extract_protected_facts

        return extract_protected_facts
    raise AttributeError(name)

