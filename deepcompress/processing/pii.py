"""
PII (Personally Identifiable Information) scrubbing.
"""

import re
from typing import Any


class PIIScrubber:
    """
    PII scrubber for sensitive data redaction.

    Detects and redacts:
    - Social Security Numbers (SSN)
    - Credit card numbers
    - Phone numbers
    - Email addresses
    - Routing numbers
    - Account numbers
    """

    def __init__(
        self,
        custom_patterns: dict[str, str]  or None = None,
        custom_replacements: dict[str, str]  or None = None,
    ) -> None:
        """
        Initialize PII scrubber.

        Args:
            custom_patterns: Custom regex patterns {name: pattern}
            custom_replacements: Custom replacement strings {name: replacement}
        """
        self.patterns = self._get_default_patterns()
        self.replacements = self._get_default_replacements()

        if custom_patterns:
            self.patterns.update(custom_patterns)

        if custom_replacements:
            self.replacements.update(custom_replacements)

        self._compiled_patterns = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in self.patterns.items()
        }

    def scrub(self, text: str) -> str:
        """
        Scrub PII from text.

        Args:
            text: Text to scrub

        Returns:
            Scrubbed text with PII redacted

        Example:
            >>> scrubber = PIIScrubber()
            >>> text = "SSN: 123-45-6789, Email: john@example.com"
            >>> scrubbed = scrubber.scrub(text)
            >>> print(scrubbed)
            SSN: [REDACTED_SSN], Email: [REDACTED_EMAIL]
        """
        scrubbed = text

        for name, pattern in self._compiled_patterns.items():
            replacement = self.replacements.get(name, f"[REDACTED_{name.upper()}]")
            scrubbed = pattern.sub(replacement, scrubbed)

        return scrubbed

    def detect(self, text: str) -> dict[str, list[str]]:
        """
        Detect PII in text without scrubbing.

        Args:
            text: Text to analyze

        Returns:
            Dictionary mapping PII type to list of detected values

        Example:
            >>> scrubber = PIIScrubber()
            >>> detected = scrubber.detect("SSN: 123-45-6789")
            >>> print(detected)
            {'ssn': ['123-45-6789']}
        """
        detected: dict[str, list[str]] = {}

        for name, pattern in self._compiled_patterns.items():
            matches = pattern.findall(text)
            if matches:
                detected[name] = matches

        return detected

    def _get_default_patterns(self) -> dict[str, str]:
        """Get default PII regex patterns."""
        return {
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
            "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
            "phone": r"\b(?:\+\d{1,2}\s?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "routing_number": r"\b\d{9}\b",
            "account_number": r"\b\d{10,17}\b",
            "date_of_birth": r"\b(?:0[1-9]|1[0-2])/(?:0[1-9]|[12]\d|3[01])/(?:19|20)\d{2}\b",
            "zip_code": r"\b\d{5}(?:-\d{4})?\b",
        }

    def _get_default_replacements(self) -> dict[str, str]:
        """Get default replacement strings."""
        return {
            "ssn": "[REDACTED_SSN]",
            "credit_card": "[REDACTED_CC]",
            "phone": "[REDACTED_PHONE]",
            "email": "[REDACTED_EMAIL]",
            "routing_number": "[REDACTED_ROUTING]",
            "account_number": "[REDACTED_ACCOUNT]",
            "date_of_birth": "[REDACTED_DOB]",
            "zip_code": "[REDACTED_ZIP]",
        }

    def add_pattern(
        self,
        name: str,
        pattern: str,
        replacement: str  or None = None,
    ) -> None:
        """
        Add custom PII pattern.

        Args:
            name: Pattern name
            pattern: Regex pattern
            replacement: Replacement string (default: [REDACTED_{NAME}])
        """
        self.patterns[name] = pattern
        self._compiled_patterns[name] = re.compile(pattern, re.IGNORECASE)

        if replacement:
            self.replacements[name] = replacement
        else:
            self.replacements[name] = f"[REDACTED_{name.upper()}]"

    def remove_pattern(self, name: str) -> None:
        """
        Remove PII pattern.

        Args:
            name: Pattern name to remove
        """
        if name in self.patterns:
            del self.patterns[name]
            del self._compiled_patterns[name]

        if name in self.replacements:
            del self.replacements[name]

