"""Shared credential-at-rest encryption helpers (Fernet).

Used by stores that persist third-party credentials (e.g. WebDAV sources,
backup targets). The encryption key comes from the ``WEBDAV_ENCRYPTION_KEY``
environment variable; a development-only fallback is used with a warning when
it is unset.
"""

from __future__ import annotations

import base64
import logging
import os

from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)


def _get_fernet() -> Fernet:
    key = os.environ.get("WEBDAV_ENCRYPTION_KEY", "")
    if not key:
        logger.warning("WEBDAV_ENCRYPTION_KEY not set; using development fallback key.")
        key = "d" * 32  # Fallback for development only
    # Fernet needs a 32-byte base64-encoded key
    key_bytes = key.encode("utf-8")
    if len(key_bytes) < 32:
        key_bytes = key_bytes.ljust(32, b"0")
    elif len(key_bytes) > 32:
        key_bytes = key_bytes[:32]
    fernet_key = base64.urlsafe_b64encode(key_bytes)
    return Fernet(fernet_key)


def _encrypt_password(plain: str) -> str:
    return _get_fernet().encrypt(plain.encode("utf-8")).decode("utf-8")


def _decrypt_password(encrypted: str) -> str:
    return _get_fernet().decrypt(encrypted.encode("utf-8")).decode("utf-8")
