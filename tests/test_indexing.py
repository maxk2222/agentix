from pathlib import Path

import pytest

from agent.indexing import IndexManager
from agent.config import AppConfig


def test_compute_local_file_fingerprints(tmp_path, monkeypatch):
    """Fingerprints should map normalized source -> SHA256 of raw file content."""
    kb_root = tmp_path / "KnowledgeDatabase"
    kb_root.mkdir()
    collection_folder = kb_root / "test_collection"
    collection_folder.mkdir()
    (collection_folder / "hello.txt").write_text("hello world")
    (collection_folder / "empty.pdf").write_bytes(b"")
    (collection_folder / "ignore.xyz").write_text("skip me")

    config = AppConfig(
        data_dir=tmp_path,
        instructions_file=tmp_path / "instructions.yaml",
    )
    # Patch knowledge_base_root so normalization works regardless of .env DATA_DIR.
    monkeypatch.setattr(
        AppConfig,
        "knowledge_base_root",
        property(lambda self: kb_root),
    )
    manager = IndexManager(config=config)

    fingerprints = manager.compute_local_file_fingerprints(collection_folder)

    assert len(fingerprints) == 2
    assert "test_collection/hello.txt" in fingerprints
    assert "test_collection/empty.pdf" in fingerprints
    assert "test_collection/ignore.xyz" not in fingerprints

    import hashlib
    expected_hash = hashlib.sha256(b"hello world").hexdigest()
    assert fingerprints["test_collection/hello.txt"] == expected_hash
    assert fingerprints["test_collection/empty.pdf"] == hashlib.sha256(b"").hexdigest()


def test_compute_local_file_fingerprints_missing_folder():
    """Missing folders should return an empty mapping."""
    config = AppConfig(
        data_dir=Path("/tmp"),
        instructions_file=Path("/tmp/instructions.yaml"),
    )
    manager = IndexManager(config=config)
    fingerprints = manager.compute_local_file_fingerprints(Path("/nonexistent"))
    assert fingerprints == {}


@pytest.mark.asyncio
async def test_compute_local_file_fingerprints_async_matches_sync(tmp_path, monkeypatch):
    """Async fingerprinting should produce identical results to the sync version."""
    kb_root = tmp_path / "KnowledgeDatabase"
    kb_root.mkdir()
    collection_folder = kb_root / "async_collection"
    collection_folder.mkdir()
    (collection_folder / "hello.txt").write_text("hello async")
    (collection_folder / "data.pdf").write_bytes(b"async pdf data")
    (collection_folder / "skip.xyz").write_text("skip me")

    config = AppConfig(
        data_dir=tmp_path,
        instructions_file=tmp_path / "instructions.yaml",
    )
    monkeypatch.setattr(
        AppConfig,
        "knowledge_base_root",
        property(lambda self: kb_root),
    )
    manager = IndexManager(config=config)

    sync_fps = manager.compute_local_file_fingerprints(collection_folder)
    async_fps = await manager.compute_local_file_fingerprints_async(collection_folder)

    assert async_fps == sync_fps
    assert len(async_fps) == 2
    assert "async_collection/hello.txt" in async_fps
    assert "async_collection/data.pdf" in async_fps
    assert "async_collection/skip.xyz" not in async_fps
