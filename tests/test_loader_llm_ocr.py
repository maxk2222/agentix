"""Tests for KnowledgeLoader LLM OCR and multimodal image analysis."""

import asyncio
import base64
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
from langchain.schema import Document

from agent.loader import KnowledgeLoader


class TestKnowledgeLoaderLLMOCR:
    """Test LLM-based OCR with Tesseract fallback in KnowledgeLoader."""

    def _make_loader(self, ocr_mode="auto", llm_factory=None):
        return KnowledgeLoader(
            folder_path=Path("/tmp/test"),
            ocr_mode=ocr_mode,
            ocr_llm_prompt="Extract text from image.",
            ocr_llm_concurrency=2,
            llm_factory=llm_factory,
        )

    def test_llm_factory_none_skips_llm(self):
        """When no llm_factory is set, _extract_text_from_image_llm returns empty."""
        loader = self._make_loader(llm_factory=None)
        result = asyncio.run(
            loader._extract_text_from_image_llm("/nonexistent.png")
        )
        assert result == ""

    def test_llm_ocr_success(self):
        """LLM returns text successfully."""
        fake_result = MagicMock()
        fake_result.content = "Extracted text from LLM"
        fake_llm = MagicMock()
        fake_llm.ainvoke = AsyncMock(return_value=fake_result)

        loader = self._make_loader(
            ocr_mode="llm",
            llm_factory=lambda: fake_llm,
        )

        # Create a tiny temp image file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="))
            tmp_path = f.name

        try:
            result = asyncio.run(
                loader._extract_text_from_image_llm(tmp_path)
            )
            assert result == "Extracted text from LLM"
        finally:
            os.unlink(tmp_path)

    def test_aextract_auto_mode_fallback_to_tesseract(self):
        """In auto mode, if LLM fails, Tesseract is tried."""
        loader = self._make_loader(
            ocr_mode="auto",
            llm_factory=lambda: MagicMock(ainvoke=AsyncMock(side_effect=Exception("LLM error"))),
        )

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="))
            tmp_path = f.name

        try:
            with patch.object(loader, "_extract_text_from_image_sync", return_value="Tesseract text"):
                result = asyncio.run(
                    loader.aextract_text_from_image(tmp_path, "test.png")
                )
                assert len(result) == 1
                assert result[0].page_content == "Tesseract text"
        finally:
            os.unlink(tmp_path)

    def test_aextract_tesseract_only_mode(self):
        """In tesseract mode, LLM is not called."""
        loader = self._make_loader(ocr_mode="tesseract", llm_factory=lambda: MagicMock())

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="))
            tmp_path = f.name

        try:
            with patch.object(loader, "_extract_text_from_image_sync", return_value="Tess result"):
                result = asyncio.run(
                    loader.aextract_text_from_image(tmp_path, "test.png")
                )
                assert result[0].page_content == "Tess result"
        finally:
            os.unlink(tmp_path)

    def test_aextract_empty_result(self):
        """Both LLM and Tesseract return empty — empty list returned."""
        loader = self._make_loader(ocr_mode="auto", llm_factory=None)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="))
            tmp_path = f.name

        try:
            with patch.object(loader, "_extract_text_from_image_sync", return_value=""):
                result = asyncio.run(
                    loader.aextract_text_from_image(tmp_path, "test.png")
                )
                assert result == []
        finally:
            os.unlink(tmp_path)


class TestKnowledgeLoaderAnalyzeImageBytes:
    """Test _analyze_image_bytes for PDF/PPTX image analysis."""

    def test_analyze_with_llm(self):
        fake_result = MagicMock()
        fake_result.content = "Image analysis result"
        fake_llm = MagicMock()
        fake_llm.ainvoke = AsyncMock(return_value=fake_result)

        loader = KnowledgeLoader(
            folder_path=Path("/tmp/test"),
            pdf_image_extraction_mode="llm",
            llm_factory=lambda: fake_llm,
        )

        image_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        result = asyncio.run(
            loader._analyze_image_bytes(image_bytes, "image/png", "test#page1#img1")
        )
        assert result == "Image analysis result"

    def test_analyze_no_llm_no_tesseract(self):
        loader = KnowledgeLoader(
            folder_path=Path("/tmp/test"),
            pdf_image_extraction_mode="llm",
            llm_factory=None,
        )

        image_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        result = asyncio.run(
            loader._analyze_image_bytes(image_bytes, "image/png", "test")
        )
        assert result == ""
