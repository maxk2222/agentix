"""Tests for app.backup_crypto."""

from __future__ import annotations

import base64
import os
from pathlib import Path

import pytest

from app import backup_crypto
from app.backup_crypto import (
    BackupCryptoError,
    BackupEncryptionKeyMissing,
    decrypt_file,
    encrypt_file,
    load_encryption_key,
)


@pytest.fixture
def key() -> bytes:
    return os.urandom(32)


def test_encrypt_decrypt_roundtrip(tmp_path: Path, key: bytes):
    src = tmp_path / "dump.bin"
    data = os.urandom(1024 * 1024 + 123)  # ~1 MB, unaligned
    src.write_bytes(data)
    enc = tmp_path / "dump.bin.enc"
    dec = tmp_path / "dump.restored.bin"

    info = encrypt_file(src, enc, key)
    assert info["size_plain"] == len(data)
    assert info["chunks"] == 1
    assert enc.read_bytes().startswith(b"RBACK1")

    out = decrypt_file(enc, dec, key)
    assert dec.read_bytes() == data
    assert out["sha256_plain"] == info["sha256_plain"]


def test_multi_chunk_roundtrip(tmp_path: Path, key: bytes, monkeypatch):
    monkeypatch.setattr(backup_crypto, "CHUNK_SIZE", 64 * 1024)
    src = tmp_path / "big.bin"
    data = os.urandom(3 * 64 * 1024 + 7)
    src.write_bytes(data)
    enc = tmp_path / "big.bin.enc"
    dec = tmp_path / "big.out"

    info = encrypt_file(src, enc, key)
    assert info["chunks"] == 4
    decrypt_file(enc, dec, key)
    assert dec.read_bytes() == data


def test_wrong_key_fails_and_removes_output(tmp_path: Path, key: bytes):
    src = tmp_path / "d.bin"
    src.write_bytes(b"secret data")
    enc = tmp_path / "d.enc"
    encrypt_file(src, enc, key)

    dec = tmp_path / "out.bin"
    with pytest.raises(BackupCryptoError, match="Integrity check"):
        decrypt_file(enc, dec, os.urandom(32))
    assert not dec.exists()


def test_corrupted_chunk_fails(tmp_path: Path, key: bytes):
    src = tmp_path / "d.bin"
    src.write_bytes(b"x" * 1000)
    enc = tmp_path / "d.enc"
    encrypt_file(src, enc, key)

    raw = bytearray(enc.read_bytes())
    raw[-5] ^= 0xFF  # flip bit inside last record
    enc.write_bytes(bytes(raw))
    with pytest.raises(BackupCryptoError):
        decrypt_file(enc, tmp_path / "out.bin", key)


def test_bad_magic_fails(tmp_path: Path, key: bytes):
    enc = tmp_path / "bad.enc"
    enc.write_bytes(b"NOTMAGIC" + os.urandom(100))
    with pytest.raises(BackupCryptoError, match="bad magic"):
        decrypt_file(enc, tmp_path / "out.bin", key)


def test_truncated_archive_fails(tmp_path: Path, key: bytes):
    src = tmp_path / "d.bin"
    src.write_bytes(b"x" * 5000)
    enc = tmp_path / "d.enc"
    encrypt_file(src, enc, key)
    raw = enc.read_bytes()
    enc.write_bytes(raw[:-20])
    with pytest.raises(BackupCryptoError):
        decrypt_file(enc, tmp_path / "out.bin", key)


def test_load_key_base64_and_hex(monkeypatch):
    k = os.urandom(32)
    monkeypatch.setenv("BACKUP_ENCRYPTION_KEY", base64.b64encode(k).decode())
    assert load_encryption_key() == k
    monkeypatch.setenv("BACKUP_ENCRYPTION_KEY", k.hex())
    assert load_encryption_key() == k


def test_load_key_missing_and_malformed(monkeypatch):
    monkeypatch.delenv("BACKUP_ENCRYPTION_KEY", raising=False)
    with pytest.raises(BackupEncryptionKeyMissing):
        load_encryption_key()
    monkeypatch.setenv("BACKUP_ENCRYPTION_KEY", "aGk=")  # decodes to 2 bytes
    with pytest.raises(BackupEncryptionKeyMissing):
        load_encryption_key()
