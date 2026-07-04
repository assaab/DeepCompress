import asyncio

import pytest

from deepcompress.core.optimizer import DTOONOptimizer
from deepcompress.exceptions import ConfigurationError
from deepcompress.models.document import ExtractedDocument, Page


class FakeLLMResponse:
    def __init__(self, text):
        self.text = text


class FakeLLMClient:
    def __init__(self, text):
        self.text = text
        self.calls = []

    async def query(self, context, question, system_prompt=None):
        self.calls.append(
            {
                "context": context,
                "question": question,
                "system_prompt": system_prompt,
            }
        )
        return FakeLLMResponse(self.text)


def make_document():
    return ExtractedDocument(
        document_id="doc-1",
        page_count=2,
        mode="small",
        pages=[
            Page(page_number=1, raw_text="Applicant: Ada Lovelace"),
            Page(page_number=2, raw_text="Income: $20,200 per month"),
        ],
    )


def test_raw_mode_keeps_page_text():
    optimizer = DTOONOptimizer()

    output = optimizer.optimize(make_document(), mode="raw")

    assert "Document ID: doc-1" in output
    assert "=== Page 1 ===" in output
    assert "Applicant: Ada Lovelace" in output
    assert "=== Page 2 ===" in output


def test_structured_mode_uses_llm_with_page_references():
    optimizer = DTOONOptimizer()
    llm = FakeLLMClient("title: Loan Application\namounts: [p2] $20,200")

    output = asyncio.run(
        optimizer.optimize_async(make_document(), mode="structured", llm_client=llm)
    )

    assert output == "title: Loan Application\namounts: [p2] $20,200"
    assert "[p1] Applicant: Ada Lovelace" in llm.calls[0]["context"]
    assert "key-value fields" in llm.calls[0]["question"]


def test_rag_mode_uses_llm_for_compact_chunks():
    optimizer = DTOONOptimizer()
    llm = FakeLLMClient("[p2] Income: total $20,200/month.")

    output = asyncio.run(
        optimizer.optimize_async(make_document(), mode="rag", llm_client=llm)
    )

    assert output == "[p2] Income: total $20,200/month."
    assert "RAG chunks" in llm.calls[0]["question"]


def test_llm_modes_require_llm_client():
    optimizer = DTOONOptimizer()

    with pytest.raises(ConfigurationError):
        asyncio.run(optimizer.optimize_async(make_document(), mode="structured"))
