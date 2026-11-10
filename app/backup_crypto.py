"""Chunked AES-256-GCM file encryption for database backup archives.

Backups can be large (hundreds of MB), so encryption works on fixed-size
chunks instead of loading the whole dump into memory. Each chunk is an
independently authenticated AES-GCM record; the per-chunk nonce is derived
from a random 8-byte file prefix plus the chunk index, and the chunk index
is bound as AAD so chunks cannot be reordered or dropped silently.

File format::

    [6B magic "RBACK1"][8B nonce prefix]
    repeated: [4B big-endian record length][ciphertext || 16B GCM tag]

The symmetric key comes from the ``BACKUP_ENCRYPTION_KEY`` environment
variable (32 bytes, hex or base64 encoded). The feature fails closed: no
key -> no backups.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import secrets
from pathlib import Path
from typing import BinaryIO, Dict

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

logger = logging.getLogger(__name__)

MAGIC = b"RBACK1"
NONCE_PREFIX_LEN = 8
TAG_LEN = 16
CHUNK_SIZE = 8 * 1024 * 1024  # 8 MiB plaintext per record
_RECORD_HEADER_LEN = 4


class BackupCryptoError(Exception):
    """Raised for any encryption/decryption problem of backup artifacts."""


class BackupEncryptionKeyMissing(BackupCryptoError):
    """Raised when BACKUP_ENCRYPTION_KEY is unset or malformed."""


def encryption_configured() -> bool:
    """Return ``True`` when a valid ``BACKUP_ENCRYPTION_KEY`` is available."""
    try:
        load_encryption_key()
    except BackupEncryptionKeyMissing:
        return False
    return True


def load_encryption_key() -> bytes:
    """Parse ``BACKUP_ENCRYPTION_KEY`` (hex or base64, 32 raw bytes).

    Raises :class:`BackupEncryptionKeyMissing` when unset or malformed.
    """
    raw = os.environ.get("BACKUP_ENCRYPTION_KEY", "").strip()
    if not raw:
        raise BackupEncryptionKeyMissing(
            "BACKUP_ENCRYPTION_KEY is not set; encrypted backups are disabled. "
            "Generate one with: openssl rand -base64 32"
        )
    key: bytes
    try:
        key = bytes.fromhex(raw)
    except ValueError:
        try:
            key = base64.b64decode(raw, validate=True)
        except Exception:
            try:
                key = base64.urlsafe_b64decode(raw)
            except Exception:
                raise BackupEncryptionKeyMissing(
                    "BACKUP_ENCRYPTION_KEY is not valid hex or base64."
                )
    if len(key) != 32:
        raise BackupEncryptionKeyMissing(
            f"BACKUP_ENCRYPTION_KEY must decode to 32 bytes, got {len(key)}."
        )
    return key


def _chunk_nonce(prefix: bytes, index: int) -> bytes:
    """96-bit nonce: 8-byte random file prefix + 4-byte chunk counter."""
    return prefix + index.to_bytes(4, "big")


def _chunk_aad(index: int) -> bytes:
    return MAGIC + index.to_bytes(4, "big")


def encrypt_file(src: Path, dst: Path, key: bytes) -> Dict[str, object]:
    """Encrypt ``src`` to ``dst`` in chunks. Returns hashes and sizes.

    Computes sha256 of plaintext and ciphertext in the same pass so the
    manifest can carry both.
    """
    aesgcm = AESGCM(key)
    prefix = secrets.token_bytes(NONCE_PREFIX_LEN)
    plain_hash = hashlib.sha256()
    enc_hash = hashlib.sha256()
    plain_size = 0
    index = 0

    with open(src, "rb") as fin, open(dst, "wb") as fout:
        header = MAGIC + prefix
        fout.write(header)
        enc_hash.update(header)
        while True:
            chunk = fin.read(CHUNK_SIZE)
            if not chunk:
                break
            plain_hash.update(chunk)
            plain_size += len(chunk)
            record = aesgcm.encrypt(_chunk_nonce(prefix, index), chunk, _chunk_aad(index))
            out = len(record).to_bytes(_RECORD_HEADER_LEN, "big") + record
            fout.write(out)
            enc_hash.update(out)
            index += 1

    enc_size = dst.stat().st_size
    return {
        "sha256_plain": plain_hash.hexdigest(),
        "sha256_enc": enc_hash.hexdigest(),
        "size_plain": plain_size,
        "size_enc": enc_size,
        "chunks": index,
    }


def decrypt_file(src: Path, dst: Path, key: bytes) -> Dict[str, object]:
    """Decrypt ``src`` to ``dst``. Returns plaintext hash and size.

    Raises :class:`BackupCryptoError` on wrong key, tampered chunks or a
    truncated/corrupt archive. The output file is removed on failure.
    """
    aesgcm = AESGCM(key)
    plain_hash = hashlib.sha256()
    plain_size = 0
    index = 0
    try:
        with open(src, "rb") as fin, open(dst, "wb") as fout:
            header = fin.read(len(MAGIC) + NONCE_PREFIX_LEN)
            if len(header) != len(MAGIC) + NONCE_PREFIX_LEN or not header.startswith(MAGIC):
                raise BackupCryptoError("Not an encrypted Agentix backup archive (bad magic).")
            prefix = header[len(MAGIC):]
            while True:
                raw_len = fin.read(_RECORD_HEADER_LEN)
                if not raw_len:
                    break
                if len(raw_len) != _RECORD_HEADER_LEN:
                    raise BackupCryptoError("Truncated record header in backup archive.")
                rec_len = int.from_bytes(raw_len, "big")
                if rec_len <= TAG_LEN or rec_len > CHUNK_SIZE + TAG_LEN:
                    raise BackupCryptoError("Invalid record length in backup archive.")
                record = fin.read(rec_len)
                if len(record) != rec_len:
                    raise BackupCryptoError("Truncated record in backup archive.")
                try:
                    plain = aesgcm.decrypt(
                        _chunk_nonce(prefix, index), record, _chunk_aad(index)
                    )
                except InvalidTag:
                    raise BackupCryptoError(
                        "Integrity check failed: wrong key or corrupted backup archive."
                    )
                fout.write(plain)
                plain_hash.update(plain)
                plain_size += len(plain)
                index += 1
    except BackupCryptoError:
        dst.unlink(missing_ok=True)
        raise
    return {
        "sha256_plain": plain_hash.hexdigest(),
        "size_plain": plain_size,
        "chunks": index,
    }


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    """Streaming sha256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()
