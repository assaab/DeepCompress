import importlib.metadata as metadata

import pytest

from deepcompress.core.config import DeepCompressConfig
from deepcompress.core.extractor import OCRExtractor
from deepcompress.exceptions import OCRError


def make_extractor() -> OCRExtractor:
    return OCRExtractor(DeepCompressConfig(enable_flash_attention=False))


def test_ocr_dependency_preflight_accepts_tested_deepseek_stack(monkeypatch):
    versions = {
        "transformers": "4.46.3",
        "tokenizers": "0.20.3",
    }

    monkeypatch.setattr(metadata, "version", lambda package: versions[package])

    make_extractor()._validate_ocr_dependency_versions()


def test_ocr_dependency_preflight_rejects_incompatible_tokenizers(monkeypatch):
    versions = {
        "transformers": "4.46.3",
        "tokenizers": "0.19.1",
    }

    monkeypatch.setattr(metadata, "version", lambda package: versions[package])

    with pytest.raises(OCRError) as exc_info:
        make_extractor()._validate_ocr_dependency_versions()

    assert "DeepSeek-OCR requires a tested transformers/tokenizers pair" in str(exc_info.value)
    assert "tokenizers==0.19.1" in exc_info.value.details["errors"][0]


def test_modelwrapper_tokenizer_error_is_repairable():
    extractor = make_extractor()

    assert extractor._is_repairable_model_cache_error(
        "data did not match any variant of untagged enum ModelWrapper"
    )
    assert extractor._is_repairable_model_cache_error("failed while parsing tokenizer.json")


def test_cache_repair_clears_only_configured_deepseek_model(monkeypatch, tmp_path):
    hf_home = tmp_path / "hf"
    model_cache = hf_home / "hub" / "models--deepseek-ai--DeepSeek-OCR"
    model_modules = hf_home / "modules" / "transformers_modules" / "deepseek-ai" / "DeepSeek-OCR"
    other_modules = hf_home / "modules" / "transformers_modules" / "deepseek-ai" / "OtherModel"

    model_cache.mkdir(parents=True)
    model_modules.mkdir(parents=True)
    other_modules.mkdir(parents=True)

    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_CACHE", raising=False)
    monkeypatch.delenv("HF_MODULES_CACHE", raising=False)

    cleared = make_extractor()._clear_huggingface_model_cache()

    assert not model_cache.exists()
    assert not model_modules.exists()
    assert other_modules.exists()
    assert any("models--deepseek-ai--DeepSeek-OCR" in path for path in cleared)
