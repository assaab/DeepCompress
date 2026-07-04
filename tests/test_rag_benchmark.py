import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.rag_benchmark import (
    Chunk,
    DEFAULT_DATASET,
    DEFAULT_QUESTIONS,
    compare_modes,
    deepcompress_rag_chunks,
    full_text_chunks,
    load_json,
    measure_question,
    retrieve,
)


def test_benchmark_compares_full_text_and_deepcompress_rag():
    documents = load_json(DEFAULT_DATASET)
    questions = load_json(DEFAULT_QUESTIONS)

    results = compare_modes(documents, questions)
    modes = {mode["mode"]: mode for mode in results["modes"]}

    assert set(modes) == {"full_text", "deepcompress_rag"}
    assert modes["full_text"]["tokens"] > modes["deepcompress_rag"]["tokens"]
    assert results["summary"]["token_reduction"] > 0
    assert results["summary"]["cost_reduction"] > 0
    assert modes["deepcompress_rag"]["answer_accuracy"] == 1.0
    assert modes["deepcompress_rag"]["retrieval_hit_rate"] == 1.0
    assert modes["deepcompress_rag"]["citation_correctness"] == 1.0
    assert modes["deepcompress_rag"]["exact_values_preserved"] > 0.9
    assert modes["deepcompress_rag"]["latency_ms"] >= 0


def test_retrieval_finds_expected_citation_page():
    document = load_json(DEFAULT_DATASET)[0]
    question = load_json(DEFAULT_QUESTIONS)[0]

    retrieved = retrieve(question, deepcompress_rag_chunks(document))

    assert retrieved
    assert retrieved[0].page == 2


def test_metric_detects_wrong_citation_page():
    question = {
        "id": "wrong-page",
        "document_id": "doc",
        "question": "What is the total monthly income?",
        "expected_answer_terms": ["$20,200"],
        "expected_values": ["$20,200"],
        "expected_pages": [9],
    }
    chunks = [Chunk(document_id="doc", page=2, text="[p2] Total monthly income $20,200.")]

    result = measure_question(
        question,
        chunks,
        context_text=chunks[0].text,
        model="gpt-4o",
    )

    assert result["answer_accuracy"] == 1.0
    assert result["retrieval_hit"] == 0.0
    assert result["citation_correct"] == 0.0


def test_exact_value_preservation_drops_when_value_missing():
    question = {
        "id": "missing-value",
        "document_id": "doc",
        "question": "What is the invoice ID?",
        "expected_answer_terms": ["INV-123"],
        "expected_values": ["INV-123", "$9,000"],
        "expected_pages": [1],
    }
    chunks = [Chunk(document_id="doc", page=1, text="[p1] Invoice ID INV-123.")]

    result = measure_question(
        question,
        chunks,
        context_text=chunks[0].text,
        model="gpt-4o",
    )

    assert result["exact_value_preservation"] == 0.5


def test_full_text_and_deepcompress_chunk_builders_keep_pages():
    document = load_json(DEFAULT_DATASET)[0]

    full_chunks = full_text_chunks(document)
    rag_chunks = deepcompress_rag_chunks(document)

    assert [chunk.page for chunk in full_chunks] == [1, 2, 3, 4]
    assert [chunk.page for chunk in rag_chunks] == [1, 2, 3, 4]


def test_run_benchmark_cli_writes_json_and_markdown(tmp_path):
    script = REPO_ROOT / "benchmarks" / "run_benchmark.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--output-dir",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Full Text RAG" in completed.stdout
    json_path = tmp_path / "benchmark_results.json"
    markdown_path = tmp_path / "benchmark_results.md"
    assert json_path.exists()
    assert markdown_path.exists()

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert {mode["mode"] for mode in data["modes"]} == {
        "full_text",
        "deepcompress_rag",
    }
    assert "| Tokens |" in markdown_path.read_text(encoding="utf-8")


def test_enterprise_rag_demo_runs_offline():
    script = REPO_ROOT / "examples" / "enterprise_rag_demo" / "demo.py"

    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "$20,200" in completed.stdout
    assert "Evidence:" in completed.stdout
    assert "Original:" in completed.stdout
    assert "Compressed:" in completed.stdout
    assert "Reduction:" in completed.stdout
