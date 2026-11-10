"""SSH exec builtin tool."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class SSHTool:
    """Execute a shell command on a remote host via SSH.

    Connection credentials (host, port, username, auth_type, password/private_key)
    are read from the tool configuration stored in the dashboard.
    """

    name = "ssh_exec"
    description = (
        "Execute a shell command on a preconfigured remote host via SSH. "
        "Use ONLY 'command' (required shell command) and optional 'timeout' (seconds, default 30, max 120). "
        "SSH credentials are configured in the tool settings."
    )

    @property
    def tool_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The shell command to execute on the remote host.",
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Max execution time in seconds (default 30, max 120).",
                            "default": 30,
                        },
                    },
                    "required": ["command"],
                },
            },
        }

    async def execute(
        self,
        command: str,
        timeout: int = 30,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        tool_config = (context or {}).get("tool_config", {})

        host = tool_config.get("host", "").strip()
        port = int(tool_config.get("port", 22) or 22)
        username = tool_config.get("username", "").strip()
        auth_type = (tool_config.get("auth_type", "") or "").strip().lower()

        if not host:
            raise ValueError("SSH host is not configured in the tool settings.")
        if not username:
            raise ValueError("SSH username is not configured in the tool settings.")
        if auth_type not in ("password", "key"):
            raise ValueError(
                "SSH auth_type must be 'password' or 'key' in the tool settings."
            )

        if auth_type == "password":
            password = tool_config.get("password", "")
            if not password:
                raise ValueError(
                    "SSH password is not configured in the tool settings."
                )
        else:  # auth_type == "key"
            private_key = tool_config.get("private_key", "").strip()
            if not private_key:
                raise ValueError(
                    "SSH private_key is not configured in the tool settings."
                )

        try:
            import asyncssh
        except ImportError as exc:
            raise RuntimeError(
                "asyncssh is not installed. Install it with: pip install asyncssh"
            ) from exc

        connect_options: Dict[str, Any] = {
            "host": host,
            "port": port,
            "username": username,
            "known_hosts": None,
        }

        if auth_type == "password":
            connect_options["password"] = tool_config.get("password", "")
        else:  # auth_type == "key"
            key_passphrase = tool_config.get("key_passphrase") or None
            try:
                key = asyncssh.import_private_key(
                    private_key, passphrase=key_passphrase
                )
                connect_options["client_keys"] = [key]
            except Exception as exc:
                raise ValueError(f"Failed to import SSH private key: {exc}") from exc

        effective_timeout = min(max(timeout, 1), 120)

        try:
            async with asyncssh.connect(**connect_options) as conn:
                result = await conn.run(command, timeout=effective_timeout)
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"SSH command timed out after {effective_timeout}s on {host}:{port}"
            )
        except asyncssh.PermissionDenied:
            raise RuntimeError(
                f"SSH authentication failed for {username}@{host}:{port}. "
                "Check credentials in the tool settings."
            )
        except asyncssh.ConnectionLost as exc:
            raise RuntimeError(
                f"SSH connection lost to {host}:{port}: {exc}"
            )
        except asyncssh.HostKeyNotVerifiable as exc:
            raise RuntimeError(
                f"SSH host key verification failed for {host}:{port}: {exc}"
            )
        except asyncssh.Error as exc:
            raise RuntimeError(
                f"SSH error connecting to {host}:{port}: {exc}"
            )

        stdout = result.stdout or ""
        stderr = result.stderr or ""
        exit_code = result.exit_status

        output_parts: List[str] = [stdout]
        if stderr:
            output_parts.append(f"\n[stderr]\n{stderr}")
        if exit_code is not None and exit_code != 0:
            output_parts.append(f"\n[exit code: {exit_code}]")

        combined = "".join(output_parts)
        if len(combined) > 15_000:
            combined = combined[:15_000] + "\n[output truncated]"

        return combined
