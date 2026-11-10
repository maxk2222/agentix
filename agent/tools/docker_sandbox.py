"""Conversation-scoped Docker sandbox manager for code execution.

Each ``conversation_id`` gets its own long-running Docker container.  The
container hosts a tiny HTTP server (``sandbox_server.py``) that executes
Python code in a persistent global namespace so variables, imports and
installed packages survive across multiple tool calls.  A TTL timer
automatically destroys idle containers after a configurable timeout.
"""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import logging
import os
import re
import secrets
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import aiohttp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TTL_SECONDS = int(os.environ.get("SANDBOX_TTL_SECONDS", "300"))
MAX_TTL_SECONDS = 3600
SANDBOX_PORT = 8080
SANDBOX_NETWORK = os.environ.get("SANDBOX_NETWORK", "agentix-sandbox-net")
# "1"/"true" is a hard operator floor: every sandbox's egress policy is
# forced to internet=off (the bridge is also created ``internal`` for
# defense in depth). Per-tool config can never re-enable internet.
SANDBOX_NETWORK_INTERNAL = os.environ.get("SANDBOX_NETWORK_INTERNAL", "").lower() in (
    "1", "true", "yes",
)
# Operator-floor blocklist — comma-separated IPs/CIDRs dropped for EVERY
# sandbox regardless of tool config (unioned with the per-tool
# ``egress_blocked`` list, e.g. org-internal ranges that must never be
# reachable even when an admin enables LAN access).
SANDBOX_EGRESS_BLOCKED_CIDRS = os.environ.get("SANDBOX_EGRESS_BLOCKED_CIDRS", "")
# Container runtime override — "runsc" (gVisor) or a Kata runtime hardens
# the kernel boundary beyond seccomp/AppArmor for hosts that install it.
SANDBOX_RUNTIME = os.environ.get("SANDBOX_RUNTIME", "")
# pip mirror / binary-only policy forwarded into the container env —
# sandbox_server.py applies them to the pip subprocess env only.
SANDBOX_PIP_INDEX_URL = os.environ.get("SANDBOX_PIP_INDEX_URL", "")
SANDBOX_PIP_EXTRA_INDEX_URL = os.environ.get("SANDBOX_PIP_EXTRA_INDEX_URL", "")
SANDBOX_PIP_ONLY_BINARY = os.environ.get("SANDBOX_PIP_ONLY_BINARY", "").lower() in (
    "1", "true", "yes",
)
# RLIMIT_NOFILE / RLIMIT_FSIZE — generated code inherits unlimited values
# otherwise (fork-bombs, accidental multi-GB dumps).
NOFILE_LIMIT = int(os.environ.get("SANDBOX_NOFILE_LIMIT", "4096"))
FSIZE_LIMIT_MB = int(os.environ.get("SANDBOX_FSIZE_LIMIT_MB", "2048"))
BASE_IMAGE = os.environ.get("SANDBOX_BASE_IMAGE", "python:3.11-slim-bookworm")
SANDBOX_IMAGE = os.environ.get("SANDBOX_IMAGE", "")
CONTAINER_PREFIX = "agentix-sandbox"
# Per-container shared secret (generated per start); injected via env and
# persisted as a container label so the manager can re-authenticate when it
# adopts a still-running container after a backend restart.
_SANDBOX_TOKEN_LABEL = "agentix.sandbox.token"
_SANDBOX_WS_LABEL = "agentix.sandbox.workspace"
_SANDBOX_AGENT_WS_MOUNT = "/agent_ws"


def _workspace_mount_label(agent_workspace_dir: Optional[Any]) -> str:
    """Label value identifying the mounted agent-workspace dir (or 'none').

    Compared against the container label on adoption so a workspace toggled
    on/off — or a scope dir that changed — forces recreation instead of
    silently reusing a container with the wrong mount.
    """
    if not agent_workspace_dir:
        return "none"
    return hashlib.sha256(str(agent_workspace_dir).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Egress policy
# ---------------------------------------------------------------------------

# Label identifying the egress policy baked into a running container —
# compared on cache-hit/adoption so a changed policy forces recreation
# instead of silently reusing a container with stale firewall rules.
_SANDBOX_EGRESS_LABEL = "agentix.sandbox.egress"
_NETCTL_CHAIN = "AGENTIX_EGRESS"

# "Local network" = anything not routable on the public internet — RFC1918,
# loopback, link-local, CGNAT (Tailscale), benchmarking + multicast.
# Deliberately broad so LAN-off also blocks sibling sandbox containers on
# the shared bridge subnet.
_LAN_CIDR_V4 = (
    "0.0.0.0/8",
    "10.0.0.0/8",
    "100.64.0.0/10",
    "127.0.0.0/8",
    "169.254.0.0/16",
    "172.16.0.0/12",
    "192.168.0.0/16",
    "198.18.0.0/15",
    "224.0.0.0/4",
    "240.0.0.0/4",
)
_LAN_CIDR_V6 = ("::/128", "::1/128", "fc00::/7", "fe80::/10", "ff00::/8")
# Cloud metadata endpoint — blocked unconditionally, even when LAN access
# is allowed: there is no legitimate workload against it.
_METADATA_IP = "169.254.169.254"
# Docker's embedded DNS resolver — dropping it kills the DNS-tunnel exfil
# channel an isolated bridge otherwise leaves open.
_EMBEDDED_DNS = "127.0.0.11"


def _parse_cidr_list(raw: Any) -> tuple:
    """Normalize a list or comma-string of IPs/CIDRs → (valid, invalid)."""
    if not raw:
        return [], []
    if isinstance(raw, str):
        items = [p.strip() for p in raw.split(",")]
    elif isinstance(raw, (list, tuple)):
        items = [str(p).strip() for p in raw]
    else:
        items = [str(raw).strip()]
    valid: list = []
    invalid: list = []
    for item in items:
        if not item:
            continue
        try:
            valid.append(str(ipaddress.ip_network(item, strict=False)))
        except ValueError:
            invalid.append(item)
    seen: set = set()
    deduped = []
    for v in valid:
        if v not in seen:
            seen.add(v)
            deduped.append(v)
    return deduped, invalid


def _resolve_egress_policy(sandbox_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Effective egress policy for one sandbox container.

    ``SANDBOX_NETWORK_INTERNAL`` is a hard floor — no tool or assistant
    config can re-enable internet when it is set. Config keys arrive via
    the tool's ``config`` merged with any per-assistant ``tools:`` override:

      * ``egress_internet`` (bool, default true) — allow public egress.
      * ``egress_local_network`` (bool, default false) — allow private/
        link-local/CGNAT ranges (LAN + sibling sandboxes on the bridge).
      * ``egress_blocked`` (list of IPs/CIDRs) — always dropped, unioned
        with the ``SANDBOX_EGRESS_BLOCKED_CIDRS`` env floor.
    """
    cfg = sandbox_config or {}
    internet = bool(cfg.get("egress_internet", True)) and not SANDBOX_NETWORK_INTERNAL
    lan = bool(cfg.get("egress_local_network", False))
    env_blocked, env_invalid = _parse_cidr_list(SANDBOX_EGRESS_BLOCKED_CIDRS)
    cfg_blocked, cfg_invalid = _parse_cidr_list(cfg.get("egress_blocked"))
    env_set = set(env_blocked)
    blocked = env_blocked + [b for b in cfg_blocked if b not in env_set]
    return {
        "internet": internet,
        "local_network": lan,
        "blocked": blocked,
        "invalid": env_invalid + cfg_invalid,
    }


def _egress_label(policy: Dict[str, Any]) -> str:
    canon = json.dumps(
        {
            "internet": policy["internet"],
            "local_network": policy["local_network"],
            "blocked": policy["blocked"],
        },
        sort_keys=True,
    )
    return hashlib.sha256(canon.encode()).hexdigest()[:16]


def _egress_rules(policy: Dict[str, Any], ipv6: bool = False) -> list:
    """iptables arg-vectors appended to the AGENTIX_EGRESS chain.

    Ordering matters: the embedded-DNS drop must precede the ``-o lo``
    accept (127.0.0.11 routes via loopback), and the metadata drop must
    precede the LAN allows so ``egress_local_network=on`` cannot reopen it.
    """
    rules: list = []
    if not ipv6:
        if not policy["internet"]:
            rules.append(["-d", _EMBEDDED_DNS, "-j", "DROP"])
        rules.append(["-d", _METADATA_IP, "-j", "DROP"])
    rules.append(["-o", "lo", "-j", "ACCEPT"])
    rules.append(["-m", "conntrack", "--ctstate", "ESTABLISHED,RELATED", "-j", "ACCEPT"])
    # Blocked CIDRs carry their address family — a v4 range must never be
    # handed to ip6tables (it would abort the whole sidecar script).
    want_family = 6 if ipv6 else 4
    for cidr in policy["blocked"]:
        try:
            if ipaddress.ip_network(cidr, strict=False).version != want_family:
                continue
        except ValueError:
            continue
        rules.append(["-d", cidr, "-j", "DROP"])
    if not policy["local_network"]:
        for cidr in (_LAN_CIDR_V6 if ipv6 else _LAN_CIDR_V4):
            rules.append(["-d", cidr, "-j", "DROP"])
    if not policy["internet"]:
        rules.append(["-j", "DROP"])
    return rules


def _egress_sidecar_script(policy: Dict[str, Any]) -> str:
    """Build the fail-closed shell script installed by the netns sidecar."""
    v4 = [
        f"iptables -N {_NETCTL_CHAIN} 2>/dev/null || iptables -F {_NETCTL_CHAIN}",
        f"iptables -C OUTPUT -j {_NETCTL_CHAIN} 2>/dev/null "
        f"|| iptables -I OUTPUT 1 -j {_NETCTL_CHAIN}",
    ]
    v4 += [
        f"iptables -A {_NETCTL_CHAIN} {' '.join(rule)}"
        for rule in _egress_rules(policy)
    ]

    v6 = [
        f"ip6tables -N {_NETCTL_CHAIN} 2>/dev/null || ip6tables -F {_NETCTL_CHAIN}",
        f"ip6tables -C OUTPUT -j {_NETCTL_CHAIN} 2>/dev/null "
        f"|| ip6tables -I OUTPUT 1 -j {_NETCTL_CHAIN}",
    ]
    v6 += [
        f"ip6tables -A {_NETCTL_CHAIN} {' '.join(rule)}"
        for rule in _egress_rules(policy, ipv6=True)
    ]
    # Join v4 and v6 with &&. A separator like `;` would mask a failed v4
    # chain whenever IPv6 is absent, returning success with no egress policy.
    return (
        " && ".join(v4)
        + " && if [ -f /proc/net/if_inet6 ]; then "
        + " && ".join(v6)
        + "; fi"
    )


_SANDBOX_TOKEN_ENV = "SANDBOX_TOKEN"
_SANDBOX_TOKEN_HEADER = "X-Sandbox-Token"

# Path to the Dockerfile used to build the custom sandbox image
_SANDBOX_DOCKERFILE = Path(__file__).with_name("sandbox.Dockerfile").resolve()
_SANDBOX_SERVER_SCRIPT = Path(__file__).with_name("sandbox_server.py").resolve()

# Resource limits applied to every sandbox container. 512m is tight for pip
# installs of larger trees (pandas & co.) — tune via SANDBOX_MEM_LIMIT.
MEM_LIMIT = os.environ.get("SANDBOX_MEM_LIMIT", "512m")
CPU_LIMIT = "1.0"

# In-container pip install cap (forwarded into the container env so
# sandbox_server.py honours it). Client-side HTTP waits get +30s slack.
INSTALL_TIMEOUT = int(os.environ.get("SANDBOX_INSTALL_TIMEOUT", "240"))

# Hard cap for a single /exec call — forwarded to the in-container server
# (best-effort thread interrupt) and used to clamp the tool's timeout arg.
EXEC_MAX_TIMEOUT = int(os.environ.get("SANDBOX_EXEC_MAX_TIMEOUT", "600"))

# Maximum characters we keep from pip-install output
_MAX_PIP_OUTPUT = 5_000

# Maximum characters we keep from code execution output
_MAX_EXEC_OUTPUT = 10_000


def _sanitize_conversation_id(cid: str) -> str:
    """Make a conversation id safe for use in a Docker container name."""
    safe = re.sub(r"[^a-zA-Z0-9_-]", "", str(cid))[:40]
    if not safe:
        safe = "unknown"
    return safe


def _container_name(conversation_id: str) -> str:
    """Deterministic container name for a conversation."""
    sanitized = _sanitize_conversation_id(conversation_id)
    # Append a short hash to avoid collisions if the sanitized id is truncated
    h = hashlib.sha256(conversation_id.encode()).hexdigest()[:8]
    return f"{CONTAINER_PREFIX}-{sanitized}-{h}"


def _sandbox_image_name() -> str:
    """Image tag derived from the sandbox build inputs.

    ``SANDBOX_IMAGE`` env explicitly overrides; otherwise the tag embeds a
    content hash of the Dockerfile and server script so changes to
    ``sandbox_server.py`` automatically produce a new tag and trigger a
    rebuild — a plain ``:latest`` would silently keep stale code forever.
    """
    if SANDBOX_IMAGE:
        return SANDBOX_IMAGE
    digest = hashlib.sha256()
    for path in (_SANDBOX_DOCKERFILE, _SANDBOX_SERVER_SCRIPT):
        try:
            digest.update(path.read_bytes())
        except OSError:
            digest.update(str(path).encode())
    return f"agentix/sandbox:{digest.hexdigest()[:12]}"


def _workspace_root() -> Path:
    """Workspace root **as the app sees it** (mkdir/chmod/delete, sweeping).

    When the backend itself runs in a container and talks to docker.sock, the
    daemon interprets bind sources on the *host* — so the app-side path can
    differ from the daemon-side path (see ``_workspace_mount_root``).
    ``SANDBOX_WORKSPACE_APP_ROOT`` overrides; falls back to the daemon path,
    then to ``<DATA_DIR>/sandbox_workspaces``.
    """
    explicit = os.environ.get("SANDBOX_WORKSPACE_APP_ROOT", "").strip()
    if explicit:
        return Path(explicit)
    daemon = os.environ.get("SANDBOX_WORKSPACE_ROOT", "").strip()
    if daemon:
        return Path(daemon)
    return Path(os.environ.get("DATA_DIR", "data")) / "sandbox_workspaces"


_warned_no_sandbox_daemon_root = False


def _workspace_mount_root() -> Path:
    """Workspace root **as the Docker daemon sees it** (volume bind source).

    ``SANDBOX_WORKSPACE_ROOT`` wins. Unset, the host path is derived from
    ``/proc/self/mountinfo`` (containerized app whose data dir is a host bind
    mount). Last resort is the app path itself — correct only when app and
    daemon share the filesystem; otherwise the sandbox silently mounts an
    empty dir.
    """
    global _warned_no_sandbox_daemon_root
    daemon = os.environ.get("SANDBOX_WORKSPACE_ROOT", "").strip()
    if daemon:
        return Path(daemon)
    from app.agent_workspace import daemon_bind_source

    derived = daemon_bind_source(_workspace_root())
    if derived is not None:
        return derived
    # Warn only inside a container — on native dev machines (Docker Desktop
    # shares the host fs) the app-path fallback below is correct.
    if not _warned_no_sandbox_daemon_root and _is_running_in_docker():
        _warned_no_sandbox_daemon_root = True
        logger.warning(
            "SANDBOX_WORKSPACE_ROOT is unset and no host bind mount covers %s; "
            "sandbox /workspace mounts will be empty. Set SANDBOX_WORKSPACE_ROOT "
            "to the host-visible workspace path.",
            _workspace_root(),
        )
    return _workspace_root().resolve()


def _conversation_workspace_dir(conversation_id: str) -> Path:
    """Per-conversation workspace dir on the app-visible filesystem."""
    from app.artifacts import sanitize_filename

    return _workspace_root() / sanitize_filename(conversation_id)


def _conversation_workspace_mount_source(conversation_id: str) -> Path:
    """Daemon-side bind source for the conversation's workspace."""
    from app.artifacts import sanitize_filename

    return _workspace_mount_root() / sanitize_filename(conversation_id)


def remove_conversation_workspace(conversation_id: str) -> None:
    """Delete a conversation's sandbox workspace directory (best-effort)."""
    try:
        shutil.rmtree(_conversation_workspace_dir(conversation_id), ignore_errors=True)
    except Exception as exc:
        logger.debug("Workspace removal failed for %s: %s", conversation_id, exc)


def _is_running_in_docker() -> bool:
    """Detect whether this process itself is running inside a Docker container."""
    if os.path.exists("/.dockerenv"):
        return True
    if os.environ.get("RUNNING_IN_DOCKER", "false").lower() == "true":
        return True
    try:
        with open("/proc/1/cgroup", "r") as fh:
            content = fh.read()
            if "docker" in content or "containerd" in content:
                return True
    except Exception:
        pass
    return False


def _get_host_port(container: Any, internal_port: int = SANDBOX_PORT) -> Optional[int]:
    """Extract the dynamically mapped host port for a running container."""
    try:
        ports = container.ports or {}
        key = f"{internal_port}/tcp"
        bindings = ports.get(key, [])
        for binding in bindings:
            host_port = binding.get("HostPort")
            if host_port:
                return int(host_port)
    except Exception:
        pass
    return None


def _get_network_gateway_ip(client: Any, network_name: str) -> Optional[str]:
    """Return the gateway IP for a Docker bridge network, if any."""
    try:
        net = client.networks.get(network_name)
        config = net.attrs.get("IPAM", {}).get("Config", [])
        if config:
            gateway = config[0].get("Gateway")
            if gateway:
                return gateway
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Sandbox handle
# ---------------------------------------------------------------------------

@dataclass
class SandboxHandle:
    """Lightweight reference to a running sandbox container."""

    conversation_id: str
    container_name: str
    url: str = ""
    token: str = ""
    workspace_label: str = "none"
    egress_label: str = "none"
    mount_warning: str = ""
    last_used_at: float = field(default_factory=time.time)
    timer_handle: Optional[asyncio.TimerHandle] = None


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class ConversationSandboxManager:
    """Manage persistent Docker sandbox containers per conversation.

    The manager is designed to be used as a singleton (one instance per
    process).  It keeps an in-memory cache of active handles and schedules
    TTL-based cleanup.
    """

    _instance: Optional["ConversationSandboxManager"] = None
    _lock: asyncio.Lock = asyncio.Lock()

    def __new__(cls) -> "ConversationSandboxManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self._cache: Dict[str, SandboxHandle] = {}
        self._ttl_seconds = max(60, min(DEFAULT_TTL_SECONDS, MAX_TTL_SECONDS))
        self._docker_available: Optional[bool] = None
        self._client: Optional[Any] = None
        self._init_lock = asyncio.Lock()

    # -- Docker client ----------------------------------------------------

    async def _ensure_client(self) -> Any:
        """Lazy-import and return the Docker SDK client.

        Tries the default Docker socket first, then common fallback paths
        (e.g. Docker Desktop's socket on macOS) so that Podman symlinks
        on /var/run/docker.sock don't block us.
        """
        if self._client is not None:
            return self._client
        if self._docker_available is False:
            raise RuntimeError("Docker sandbox unavailable: Docker client could not be initialized")
        async with self._init_lock:
            if self._client is not None:
                return self._client
            if self._docker_available is False:
                raise RuntimeError("Docker sandbox unavailable: Docker client could not be initialized")

            import docker
            last_exc: Optional[Exception] = None

            # Prefer explicit env var, then try common sockets
            candidates = []
            if os.environ.get("DOCKER_HOST"):
                candidates.append(os.environ["DOCKER_HOST"])
            # Docker Desktop (macOS modern)
            home = os.path.expanduser("~")
            candidates.append(f"unix://{home}/.docker/run/docker.sock")
            # Docker Desktop legacy / Linux default
            candidates.append("unix:///var/run/docker.sock")

            for base_url in candidates:
                try:
                    self._client = docker.DockerClient(base_url=base_url)
                    self._docker_available = True
                    logger.info("Docker client initialized via %s", base_url)
                    break
                except Exception as exc:
                    last_exc = exc
                    logger.debug("Docker socket %s failed: %s", base_url, exc)

            if self._client is None:
                self._docker_available = False
                logger.warning("Docker client initialization failed: %s", last_exc)
                raise RuntimeError(f"Docker sandbox unavailable: {last_exc}") from last_exc
        return self._client

    def _check_docker(self) -> None:
        if self._docker_available is False:
            raise RuntimeError("Docker sandbox unavailable: Docker client could not be initialized")

    # -- Network ----------------------------------------------------------

    async def _ensure_network(self) -> None:
        """Create the dedicated bridge network if it does not exist."""
        client = await self._ensure_client()
        try:
            net = client.networks.get(SANDBOX_NETWORK)
            try:
                internal = bool((net.attrs or {}).get("Internal"))
            except Exception:
                internal = False
            if not internal or SANDBOX_NETWORK_INTERNAL:
                return
            # Stale internal bridge from when SANDBOX_NETWORK_INTERNAL was
            # set — it would silently deaden egress the current policy
            # expects. Recreate as a normal bridge; only possible once no
            # container is attached (warn loudly otherwise).
            logger.warning(
                "Docker network %s is internal but SANDBOX_NETWORK_INTERNAL "
                "is unset — recreating it as a normal bridge",
                SANDBOX_NETWORK,
            )
            try:
                net.remove()
            except Exception as exc:
                logger.error(
                    "Cannot recreate internal network %s (%s) — sandbox "
                    "egress policy may not take effect until it is removed",
                    SANDBOX_NETWORK, exc,
                )
                return
        except Exception:
            pass  # missing or unreadable — (re)create below
        logger.info("Creating Docker network: %s", SANDBOX_NETWORK)
        try:
            client.networks.create(
                SANDBOX_NETWORK, driver="bridge", internal=SANDBOX_NETWORK_INTERNAL
            )
        except Exception as exc:
            logger.warning("Failed to create network %s: %s", SANDBOX_NETWORK, exc)
            # The network may already exist from a race; that's fine.

    # -- Container lifecycle ----------------------------------------------

    async def get_or_create(
        self,
        conversation_id: str,
        sandbox_config: Optional[Dict[str, Any]] = None,
        agent_workspace_dir: Optional[Any] = None,
    ) -> SandboxHandle:
        """Return a running sandbox handle for *conversation_id*, creating one if needed."""
        self._check_docker()
        await self._ensure_network()

        egress_policy = _resolve_egress_policy(sandbox_config)
        egress_label = _egress_label(egress_policy)

        # 1. In-memory cache hit -> reset TTL and return
        if conversation_id in self._cache:
            handle = self._cache[conversation_id]
            handle.last_used_at = time.time()
            self._schedule_cleanup(handle)
            # Verify the container is actually running AND still carries the
            # mount + egress config this call expects — a workspace toggled
            # on/off (or a scope dir that changed) and an egress policy that
            # changed must not silently reuse a stale container. The
            # daemon-side checks only cover containers unknown to this
            # process.
            ws_expected = _workspace_mount_label(agent_workspace_dir)
            if await self._container_is_running(handle.container_name):
                if handle.workspace_label == ws_expected and handle.egress_label == egress_label:
                    logger.debug("Sandbox cache hit for %s", conversation_id)
                    return handle
                logger.info(
                    "Sandbox %s cached but config changed "
                    "(ws %s!=%s or egress %s!=%s); recreating",
                    conversation_id,
                    handle.workspace_label, ws_expected,
                    handle.egress_label, egress_label,
                )
            else:
                logger.debug("Sandbox cache stale for %s; container not running", conversation_id)
            self._cancel_cleanup(handle)
            del self._cache[conversation_id]

        # 2. Check for an existing (restarted) container on Docker side
        container_name = _container_name(conversation_id)
        if await self._container_is_running(container_name):
            client = await self._ensure_client()
            container = client.containers.get(container_name)
            token = (container.labels or {}).get(_SANDBOX_TOKEN_LABEL) or ""
            image_ok = _sandbox_image_name() in (container.image.tags or [])
            ws_label = (container.labels or {}).get(_SANDBOX_WS_LABEL) or ""
            ws_ok = ws_label == _workspace_mount_label(agent_workspace_dir)
            egress_ok = (container.labels or {}).get(_SANDBOX_EGRESS_LABEL) == egress_label
            if not token or not image_ok or not ws_ok or not egress_ok:
                # Never adopt: pre-auth container (no token label), stale image
                # (old sandbox_server — e.g. missing /files/read), a mount
                # config that no longer matches the current agent workspace
                # (toggled on/off or scope dir changed mid-conversation), or
                # a different egress policy than the one now requested.
                logger.info(
                    "Sandbox %s stale (token=%s, image_ok=%s, ws_ok=%s, "
                    "egress_ok=%s); recreating",
                    container_name, bool(token), image_ok, ws_ok, egress_ok,
                )
            else:
                logger.info("Re-using existing sandbox container for %s", conversation_id)
                host_port = _get_host_port(container)
                candidates: list[str] = [f"http://{container_name}:{SANDBOX_PORT}"]
                if host_port:
                    candidates.append(f"http://host.docker.internal:{host_port}")
                    candidates.append(f"http://localhost:{host_port}")
                ready_url = await self._wait_for_ready(candidates, timeout=10, token=token)
                if not ready_url:
                    logger.warning("Existing sandbox container %s is not responding", container_name)
                    # Fall through to recreate
                else:
                    handle = SandboxHandle(
                        conversation_id=conversation_id,
                        container_name=container_name,
                        url=ready_url,
                        token=token,
                        workspace_label=ws_label,
                        egress_label=egress_label,
                    )
                    self._cache[conversation_id] = handle
                    self._schedule_cleanup(handle)
                    return handle

        # 3. Start a fresh container
        logger.info("Creating new sandbox container for %s", conversation_id)
        handle = await self._start_container(
            conversation_id, container_name, sandbox_config=sandbox_config,
            agent_workspace_dir=agent_workspace_dir,
            egress_policy=egress_policy,
        )
        self._cache[conversation_id] = handle
        self._schedule_cleanup(handle)
        return handle

    async def _container_is_running(self, container_name: str) -> bool:
        client = await self._ensure_client()
        try:
            container = client.containers.get(container_name)
            return container.status == "running"
        except Exception:
            return False

    async def _start_container(
        self,
        conversation_id: str,
        container_name: str,
        sandbox_config: Optional[Dict[str, Any]] = None,
        agent_workspace_dir: Optional[Any] = None,
        egress_policy: Optional[Dict[str, Any]] = None,
    ) -> SandboxHandle:
        client = await self._ensure_client()
        if egress_policy is None:
            egress_policy = _resolve_egress_policy(sandbox_config)
        egress_label = _egress_label(egress_policy)

        # Remove any leftover container with the same name
        try:
            old = client.containers.get(container_name)
            old.stop(timeout=5)
            old.remove(force=True)
            logger.debug("Removed leftover sandbox container %s", container_name)
        except Exception:
            pass

        # Ensure the custom sandbox image is built (contains sandbox_server.py)
        image = await self._ensure_sandbox_image(client)

        sandbox_mode = (sandbox_config or {}).get("sandbox_mode", "read_only")

        # Per-container shared secret: the in-container HTTP executor requires
        # it on every request so sibling sandboxes on the shared bridge network
        # and host/LAN peers cannot exec into this container.
        token = secrets.token_hex(24)

        from docker.types import LogConfig, Ulimit

        # Base kwargs
        run_kwargs: Dict[str, Any] = {
            "image": image,
            "name": container_name,
            "hostname": container_name,
            "detach": True,
            "environment": {
                _SANDBOX_TOKEN_ENV: token,
                # Lets the in-container executor fail /install fast with a
                # clear reason when the egress policy disables internet.
                "SANDBOX_NETWORK_ISOLATED": "" if egress_policy["internet"] else "1",
                # pip subprocess cap inside the container — keep the client's
                # HTTP wait above this so pip isn't killed mid-flight.
                "SANDBOX_INSTALL_TIMEOUT": str(INSTALL_TIMEOUT),
                # Per-exec hard cap inside the container.
                "SANDBOX_EXEC_MAX_TIMEOUT": str(EXEC_MAX_TIMEOUT),
            },
            # Attach ONLY to the dedicated sandbox bridge: leaving `network`
            # unset would also hook the container to the default bridge, which
            # quietly restores full internet egress and makes
            # SANDBOX_NETWORK_INTERNAL a no-op. On user-defined bridges the
            # container name resolves in embedded DNS automatically.
            "network": SANDBOX_NETWORK,
            "labels": {
                "agentix.sandbox.conversation_id": conversation_id,
                "agentix.sandbox.managed": "true",
                _SANDBOX_TOKEN_LABEL: token,
                _SANDBOX_WS_LABEL: _workspace_mount_label(agent_workspace_dir),
                _SANDBOX_EGRESS_LABEL: egress_label,
            },
            "mem_limit": MEM_LIMIT,
            "memswap_limit": MEM_LIMIT,
            "cpu_quota": int(float(CPU_LIMIT) * 100_000),
            "cpu_period": 100_000,
            # Fork-bomb guard — a plain python workload never needs hundreds
            # of processes; 512 leaves generous headroom for threads/procs
            # spawned by numpy & co.
            "pids_limit": 512,
            # Generated code runs unprivileged (uid 1000 in most modes) and
            # needs no Linux capabilities; drop the whole set and forbid
            # privilege escalation via setuid/file-cap binaries. cap_drop ALL
            # also means even a root-mode sandbox cannot edit the egress
            # rules the netctl sidecar installs.
            "cap_drop": ["ALL"],
            "security_opt": ["no-new-privileges:true"],
            # Search domains inherited from the host resolv.conf would leak
            # internal infrastructure names into the sandbox.
            "dns_search": ["."],
            # Zombie-reaping init: exec'd code spawning orphans must not pile
            # up zombies for the container's lifetime.
            "init": True,
            "sysctls": {"net.ipv4.ip_forward": "0"},
            # Generated code inherits the daemon's unlimited rlimits
            # otherwise — cap FDs and dump size explicitly.
            "ulimits": [
                Ulimit(name="nofile", soft=NOFILE_LIMIT, hard=NOFILE_LIMIT),
                Ulimit(
                    name="fsize",
                    soft=FSIZE_LIMIT_MB * 1024 * 1024,
                    hard=FSIZE_LIMIT_MB * 1024 * 1024,
                ),
            ],
            # Bound container log growth — chatty execs must not fill the
            # daemon's json-file store.
            "log_config": LogConfig(
                type="json-file", config={"max-size": "10m", "max-file": "3"}
            ),
            "command": ["python", "/sandbox_server.py"],
        }
        # Optional stronger isolation runtime (gVisor ``runsc``, Kata) —
        # passthrough so operators can opt in without image changes.
        if SANDBOX_RUNTIME:
            run_kwargs["runtime"] = SANDBOX_RUNTIME
        # pip mirror / binary-only policy — read and consumed by
        # sandbox_server.py's /install handler only (scrubbed from the
        # exec-visible process env there).
        for _env_key, _env_val in (
            ("SANDBOX_PIP_INDEX_URL", SANDBOX_PIP_INDEX_URL),
            ("SANDBOX_PIP_EXTRA_INDEX_URL", SANDBOX_PIP_EXTRA_INDEX_URL),
            ("SANDBOX_PIP_ONLY_BINARY", "1" if SANDBOX_PIP_ONLY_BINARY else ""),
        ):
            if _env_val:
                run_kwargs["environment"][_env_key] = _env_val
        mount_warnings: list[str] = []
        agent_ws_probe_dir: Optional[Path] = None

        # Apply sandbox mode specific settings
        volumes: Dict[str, Any] = {}
        # Dedicated exec-capable tmpfs for pip --target installs — writable in
        # every mode (read_only's /tmp is noexec) and big enough for numpy etc.
        # mode=1777 so uid-1000 runs can write; 'exec' is required because
        # Docker tmpfs mounts default to noexec — without it wheels that ship
        # C extensions (numpy, pandas, …) install fine but fail at import.
        run_kwargs["tmpfs"] = {"/var/sandbox-packages": "size=512m,mode=1777,exec"}

        if sandbox_mode == "read_only":
            run_kwargs["read_only"] = True
            run_kwargs["tmpfs"]["/tmp"] = "noexec,nosuid,size=100m"
            run_kwargs["user"] = "sandboxuser"
        elif sandbox_mode == "writable_user":
            run_kwargs["read_only"] = False
            run_kwargs["user"] = "sandboxuser"
        elif sandbox_mode == "writable_root":
            run_kwargs["read_only"] = False
            # run as root by omitting user
        elif sandbox_mode == "writable_workspace":
            run_kwargs["read_only"] = False
            run_kwargs["user"] = "sandboxuser"
            # Per-conversation host dir — mounting a shared /workspace would
            # leak files across conversations AND users. World-writable so
            # the unprivileged sandbox uid can write regardless of mapping.
            # The bind source is daemon-side; mkdir/chmod happen on the
            # app-side path.
            ws_dir = _conversation_workspace_dir(conversation_id)
            ws_dir.mkdir(parents=True, exist_ok=True)
            os.chmod(ws_dir, 0o777)
            ws_src = _conversation_workspace_mount_source(conversation_id)
            from app.agent_workspace import is_suspicious_mount_root

            if is_suspicious_mount_root(ws_src):
                # Fail closed: a bind source resolving to a system/shallow
                # root would expose an unrelated host subtree to generated
                # code. Better no /workspace than the wrong one.
                logger.error(
                    "Refusing to bind-mount suspicious /workspace source %s "
                    "for %s — check SANDBOX_WORKSPACE_ROOT",
                    ws_src, conversation_id,
                )
                mount_warnings.append(
                    f"/workspace mount skipped (suspicious bind source {ws_src})"
                )
            else:
                volumes[str(ws_src)] = {"bind": "/workspace", "mode": "rw"}
        else:
            # fallback to read_only for unknown modes
            run_kwargs["read_only"] = True
            run_kwargs["tmpfs"]["/tmp"] = "noexec,nosuid,size=100m"
            run_kwargs["user"] = "sandboxuser"

        # Persistent per-assistant agent workspace — mounted in EVERY mode so
        # the agent can share files with the sandbox even when the rest of the
        # container fs is read-only. App-side mkdir/chmod (uid 1000 writes);
        # the bind source is the daemon-visible path (AGENT_WORKSPACE_ROOT).
        if agent_workspace_dir:
            from app.agent_workspace import AgentWorkspaceStore, is_suspicious_mount_root

            ws_dir = Path(agent_workspace_dir)
            agent_src = AgentWorkspaceStore().mount_source(ws_dir)
            if is_suspicious_mount_root(agent_src):
                # Decide BEFORE touching the filesystem — mkdir/chmod on an
                # arbitrary caller-supplied path (e.g. "/") must not happen
                # for a mount we are about to refuse.
                logger.error(
                    "Refusing to bind-mount suspicious %s source %s for %s — "
                    "check AGENT_WORKSPACE_ROOT",
                    _SANDBOX_AGENT_WS_MOUNT, agent_src, conversation_id,
                )
                mount_warnings.append(
                    f"{_SANDBOX_AGENT_WS_MOUNT} mount skipped "
                    f"(suspicious bind source {agent_src})"
                )
            else:
                ws_dir.mkdir(parents=True, exist_ok=True)
                try:
                    os.chmod(ws_dir, 0o777)
                except OSError:
                    pass
                volumes[str(agent_src)] = {
                    "bind": _SANDBOX_AGENT_WS_MOUNT,
                    "mode": "rw",
                }
                agent_ws_probe_dir = ws_dir

        if volumes:
            run_kwargs["volumes"] = volumes

        # Always publish a host port so the sandbox is reachable both from the host
        # OS (app running locally) and from inside Docker (app in a container).
        run_kwargs["ports"] = {f"{SANDBOX_PORT}/tcp": None}

        try:
            container = client.containers.run(**run_kwargs)
            logger.info("Started sandbox container %s (image=%s)", container_name, image)
        except Exception as exc:
            logger.error("Failed to start sandbox container %s: %s", container_name, exc)
            raise RuntimeError(f"Failed to start sandbox container: {exc}") from exc

        # Refresh the container object so that port mappings are present.
        container.reload()

        # Egress policy inside the container netns — applied by a one-shot
        # NET_ADMIN sidecar BEFORE the sandbox is marked ready, so generated
        # code can never observe an unfiltered moment. Fail closed: a
        # container whose policy cannot be applied must not run unfiltered.
        try:
            self._apply_egress_policy(client, container_name, egress_policy, image)
        except Exception as exc:
            logger.error(
                "Egress policy failed for %s: %s", container_name, exc
            )
            try:
                container.stop(timeout=5)
            except Exception:
                pass
            try:
                container.remove(force=True)
            except Exception:
                pass
            raise RuntimeError(
                f"Failed to apply sandbox egress policy: {exc}"
            ) from exc

        if egress_policy["invalid"]:
            mount_warnings.append(
                "ignored invalid egress_blocked entries: "
                + ", ".join(egress_policy["invalid"])
            )

        # The container was created directly on SANDBOX_NETWORK — its name
        # already resolves in the bridge's embedded DNS, so no explicit
        # connect/alias step is needed here.

        # Build a list of candidate URLs.  The sandbox may be reachable via the
        # Docker-internal DNS name (when the app is attached to the same network),
        # via host.docker.internal (from another Docker container on the host),
        # or via localhost (when the app runs directly on the host OS).
        host_port = _get_host_port(container)
        gateway_ip = _get_network_gateway_ip(client, SANDBOX_NETWORK)
        candidates: list[str] = [f"http://{container_name}:{SANDBOX_PORT}"]
        if host_port:
            candidates.append(f"http://host.docker.internal:{host_port}")
            candidates.append(f"http://localhost:{host_port}")
            if gateway_ip:
                candidates.append(f"http://{gateway_ip}:{host_port}")

        ready_url = await self._wait_for_ready(candidates, timeout=60, token=token)
        if not ready_url:
            logger.error("Sandbox container %s did not become ready in time (tried: %s)", container_name, candidates)
            try:
                logs = container.logs(tail=50).decode("utf-8", errors="replace")
                logger.error("Sandbox container %s logs:\n%s", container_name, logs)
            except Exception as log_exc:
                logger.error("Could not retrieve logs for sandbox %s: %s", container_name, log_exc)
            try:
                container.stop(timeout=5)
            except Exception as stop_exc:
                logger.debug("Failed to stop unready sandbox %s: %s", container_name, stop_exc)
            try:
                container.remove(force=True)
            except Exception as remove_exc:
                logger.debug("Failed to remove unready sandbox %s: %s", container_name, remove_exc)
            raise RuntimeError(f"Sandbox container {container_name} failed to start")

        if agent_ws_probe_dir is not None:
            await self._probe_agent_ws_mount(
                ready_url, token, agent_ws_probe_dir, mount_warnings, conversation_id
            )

        return SandboxHandle(
            conversation_id=conversation_id,
            container_name=container_name,
            url=ready_url,
            token=token,
            workspace_label=_workspace_mount_label(agent_workspace_dir),
            egress_label=egress_label,
            mount_warning="; ".join(mount_warnings),
        )

    def _apply_egress_policy(
        self,
        client: Any,
        container_name: str,
        policy: Dict[str, Any],
        image: str,
    ) -> None:
        """Install the egress rules inside the sandbox's netns.

        Runs the sandbox image itself as a one-shot helper sharing the
        container's network namespace (``network_mode=container:``) with
        NET_ADMIN — the only capability the helper gets (``cap_drop ALL``
        + re-add), so the sandbox itself can never modify the result even
        when running as root. The helper exits after one pass; iptables
        rules live in the netns, not the container, so they survive it.
        """
        script = _egress_sidecar_script(policy)
        helper = client.containers.run(
            image=image,
            command=["sh", "-c", script],
            detach=True,
            # Same netns as the sandbox — the rules land in the sandbox's
            # OUTPUT chain, not the helper's.
            network_mode=f"container:{container_name}",
            # root, explicitly: the image's USER sandboxuser must not apply —
            # cap_add grants NET_ADMIN only to a uid-0 exec (non-root gets it
            # in bounding/permitted but never effective → "must be root").
            user="0",
            cap_drop=["ALL"],
            cap_add=["NET_ADMIN"],
            read_only=True,
            security_opt=["no-new-privileges:true"],
            pids_limit=16,
            mem_limit="32m",
            labels={"agentix.sandbox.netctl": "true"},
        )
        try:
            try:
                rc = helper.wait(timeout=60).get("StatusCode", -1)
            except TypeError:
                rc = helper.wait().get("StatusCode", -1)  # old docker-py
            try:
                logs = helper.logs().decode("utf-8", errors="replace")
            except Exception:
                logs = ""
        finally:
            try:
                helper.remove(force=True)
            except Exception:
                pass
        if rc != 0:
            raise RuntimeError(
                f"netctl sidecar exited with rc={rc}: {logs[-500:]}"
            )

    async def _probe_agent_ws_mount(
        self,
        base_url: str,
        token: str,
        ws_dir: Path,
        mount_warnings: list,
        conversation_id: str,
    ) -> None:
        """Verify the ``/agent_ws`` bind inside the sandbox is the app-side dir.

        A daemon path mismatch silently mounts an auto-created empty host
        directory — the sentinel file exists in exactly one place, so a miss
        proves the bind source is wrong regardless of cause.
        """
        sentinel = ws_dir / ".agentix_ws_probe"
        try:
            sentinel.write_text("ok")
            # Visibility AND writability: a mismatched bind source lands on a
            # root-owned auto-created host dir (755) which uid-1000 can see but
            # not write to — a read-only probe would call that healthy.
            code = (
                "import os\n"
                f"p = {_SANDBOX_AGENT_WS_MOUNT!r}\n"
                "ok = os.path.isfile(p + '/.agentix_ws_probe')\n"
                "try:\n"
                "    with open(p + '/.agentix_ws_wprobe', 'w') as f: f.write('x')\n"
                "    os.remove(p + '/.agentix_ws_wprobe')\n"
                "    w = True\n"
                "except OSError:\n"
                "    w = False\n"
                "print('VISIBLE', ok, 'WRITABLE', w)\n"
            )
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{base_url}/exec",
                    json={"code": code},
                    headers={_SANDBOX_TOKEN_HEADER: token},
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    data = await resp.json()
            stdout = str(data.get("stdout", ""))
            if "VISIBLE True" not in stdout:
                logger.error(
                    "Agent workspace mount probe failed for %s: sentinel not "
                    "visible in sandbox — the bind source does not match the "
                    "app-side dir (set AGENT_WORKSPACE_ROOT to the host-visible "
                    "workspace path)",
                    conversation_id,
                )
                mount_warnings.append(
                    f"{_SANDBOX_AGENT_WS_MOUNT} mounted but empty/wrong — the "
                    "agent workspace is not reachable from the sandbox"
                )
            elif "WRITABLE True" not in stdout:
                logger.error(
                    "Agent workspace mount probe failed for %s: /agent_ws is "
                    "not writable inside the sandbox (bind source ownership?)",
                    conversation_id,
                )
                mount_warnings.append(
                    f"{_SANDBOX_AGENT_WS_MOUNT} is mounted read-only for the "
                    "sandbox user — check bind source permissions"
                )
        except Exception as exc:
            logger.warning("agent_ws mount probe failed for %s: %s", conversation_id, exc)
        finally:
            for leftover in (sentinel, ws_dir / ".agentix_ws_wprobe"):
                try:
                    leftover.unlink(missing_ok=True)
                except OSError:
                    pass

    async def _wait_for_ready(
        self, urls: list[str], timeout: float = 30.0, token: str = ""
    ) -> str | None:
        """Poll the sandbox server via multiple candidate URLs.

        Returns the first URL that responds, or ``None`` if none answer before
        *timeout* elapses.
        """
        headers = {_SANDBOX_TOKEN_HEADER: token} if token else {}
        deadline = time.time() + timeout
        async with aiohttp.ClientSession() as session:
            while time.time() < deadline:
                for url in urls:
                    exec_url = f"{url}/exec"
                    try:
                        async with session.post(
                            exec_url,
                            json={"code": "pass"},
                            headers=headers,
                            timeout=aiohttp.ClientTimeout(total=2),
                        ) as resp:
                            if resp.status in (200, 400, 500):
                                logger.debug("Sandbox reachable at %s (status %s)", url, resp.status)
                                return url
                            logger.debug("Sandbox at %s returned unexpected status %s", url, resp.status)
                    except Exception as exc:
                        logger.debug("Sandbox at %s not reachable: %s", url, exc)
                await asyncio.sleep(0.5)
        return None

    # -- TTL cleanup ------------------------------------------------------

    def _schedule_cleanup(self, handle: SandboxHandle) -> None:
        self._cancel_cleanup(handle)
        loop = asyncio.get_event_loop()
        handle.timer_handle = loop.call_later(
            self._ttl_seconds, self._destroy_sync, handle.conversation_id
        )
        logger.debug("Scheduled sandbox cleanup for %s in %ds", handle.conversation_id, self._ttl_seconds)

    def _cancel_cleanup(self, handle: SandboxHandle) -> None:
        if handle.timer_handle is not None:
            handle.timer_handle.cancel()
            handle.timer_handle = None

    def _destroy_sync(self, conversation_id: str) -> None:
        """Synchronous entry point for the asyncio timer."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self.destroy(conversation_id))
            else:
                loop.run_until_complete(self.destroy(conversation_id))
        except Exception as exc:
            logger.warning("Failed to schedule sandbox destroy for %s: %s", conversation_id, exc)

    async def destroy(self, conversation_id: str) -> None:
        """Stop and remove the sandbox container for *conversation_id*."""
        handle = self._cache.pop(conversation_id, None)
        if handle is not None:
            self._cancel_cleanup(handle)

        container_name = _container_name(conversation_id)
        try:
            client = await self._ensure_client()
            container = client.containers.get(container_name)
        except Exception:
            # Container may already be gone; that's fine.
            return

        # Attempt graceful stop, then force kill, then remove – each guarded
        try:
            container.stop(timeout=5)
        except Exception as exc:
            logger.debug("Graceful stop failed for %s: %s", container_name, exc)
            try:
                container.kill()
            except Exception as kill_exc:
                logger.debug("Force kill failed for %s: %s", container_name, kill_exc)

        try:
            container.remove(force=True)
            logger.info("Destroyed sandbox container %s", container_name)
        except Exception as exc:
            logger.debug("Remove failed for %s: %s", container_name, exc)

    async def destroy_all(self) -> None:
        """Stop and remove **all** tracked sandbox containers."""
        # Copy keys so we can mutate the dict while iterating
        conversation_ids = list(self._cache.keys())
        if not conversation_ids:
            return
        logger.info("Destroying %d sandbox container(s)...", len(conversation_ids))
        results = await asyncio.gather(
            *[self.destroy(cid) for cid in conversation_ids],
            return_exceptions=True,
        )
        failures = [r for r in results if isinstance(r, Exception)]
        if failures:
            logger.warning("destroy_all encountered %d failure(s): %s", len(failures), failures)

    async def _ensure_sandbox_image(self, client: Any) -> str:
        """Build the custom sandbox image if it does not yet exist."""
        image_name = _sandbox_image_name()
        try:
            client.images.get(image_name)
            logger.debug("Sandbox image %s already exists", image_name)
            return image_name
        except Exception:
            pass

        if not _SANDBOX_SERVER_SCRIPT.exists():
            raise RuntimeError(f"Sandbox server script not found: {_SANDBOX_SERVER_SCRIPT}")
        if not _SANDBOX_DOCKERFILE.exists():
            raise RuntimeError(f"Sandbox Dockerfile not found: {_SANDBOX_DOCKERFILE}")

        logger.info("Building sandbox image %s ...", image_name)
        try:
            image_obj, _ = client.images.build(
                path=str(_SANDBOX_SERVER_SCRIPT.parent),
                dockerfile=str(_SANDBOX_DOCKERFILE.name),
                tag=image_name,
                rm=True,
            )
            logger.info("Built sandbox image %s (id=%s)", image_name, image_obj.id)
            return image_name
        except Exception as exc:
            logger.error("Failed to build sandbox image %s: %s", SANDBOX_IMAGE, exc)
            raise RuntimeError(f"Failed to build sandbox image: {exc}") from exc

    # -- Execution --------------------------------------------------------

    async def execute_in(
        self,
        handle: SandboxHandle,
        code: str,
        timeout: int = 30,
        queue_wait: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run *code* inside the sandbox container referenced by *handle*.

        Execs serialize inside a sandbox (shared namespace/stdout). A busy
        sandbox holds the request up to ``queue_wait`` seconds before 429ing;
        ``None`` keeps the interactive default of waiting one exec timeout —
        background tasks pass more so they queue instead of failing.
        """
        url = f"{handle.url}/exec"
        # The in-container watchdog fires at `timeout`; the HTTP wait gets
        # slack so the server can report the timeout itself (clean error),
        # and must additionally cover the server's lock queue wait.
        q_wait = float(timeout if queue_wait is None else max(0.0, queue_wait))
        payload = {"code": code, "timeout": timeout, "queue_wait": q_wait}
        headers = {_SANDBOX_TOKEN_HEADER: handle.token} if handle.token else {}
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=q_wait + timeout + 15),
                ) as resp:
                    data = await resp.json()
            except asyncio.TimeoutError:
                return {
                    "ok": False,
                    "stdout": "",
                    "stderr": "",
                    "error": f"Code execution timed out after {timeout} seconds",
                }
            except Exception as exc:
                return {
                    "ok": False,
                    "stdout": "",
                    "stderr": "",
                    "error": f"Failed to communicate with sandbox: {exc}",
                }

        stdout = str(data.get("stdout", ""))
        stderr = str(data.get("stderr", ""))
        error = data.get("error")
        ok = bool(data.get("ok", False)) and error is None

        # Truncate
        if len(stdout) > _MAX_EXEC_OUTPUT:
            stdout = stdout[:_MAX_EXEC_OUTPUT] + "\n...[stdout truncated]"
        if len(stderr) > _MAX_EXEC_OUTPUT:
            stderr = stderr[:_MAX_EXEC_OUTPUT] + "\n...[stderr truncated]"

        # Reset TTL on every successful interaction
        handle.last_used_at = time.time()
        self._schedule_cleanup(handle)

        return {
            "ok": ok,
            "stdout": stdout,
            "stderr": stderr,
            "error": error,
        }

    async def read_file_in(
        self,
        handle: SandboxHandle,
        path: str,
        *,
        max_bytes: int = 32 * 1024 * 1024,
        timeout: int = 30,
    ) -> Dict[str, Any]:
        """Read a file from inside the sandbox container; returns decoded bytes."""
        url = f"{handle.url}/files/read"
        payload = {"path": path, "max_bytes": max_bytes}
        headers = {_SANDBOX_TOKEN_HEADER: handle.token} if handle.token else {}
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as resp:
                    data = await resp.json()
            except asyncio.TimeoutError:
                return {"ok": False, "error": f"File read timed out after {timeout} seconds"}
            except Exception as exc:
                return {"ok": False, "error": f"Failed to communicate with sandbox: {exc}"}

        if not data.get("ok"):
            return {"ok": False, "error": str(data.get("error") or "file read failed")}

        import base64
        try:
            content = base64.b64decode(str(data.get("data_b64") or ""))
        except Exception:
            return {"ok": False, "error": "sandbox returned undecodable file data"}

        handle.last_used_at = time.time()
        self._schedule_cleanup(handle)

        return {
            "ok": True,
            "name": str(data.get("name") or Path(path).name),
            "size": int(data.get("size") or len(content)),
            "data": content,
        }

    async def install_in(
        self,
        handle: SandboxHandle,
        packages: list[str],
        timeout: int = INSTALL_TIMEOUT + 30,
        queue_wait: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Install pip packages inside the sandbox container.

        Installs serialize with execs (shared interpreter state). A busy
        sandbox holds the request up to ``queue_wait`` seconds (default 10,
        matching the server's interactive default); background callers pass
        more so they queue instead of failing.
        """
        url = f"{handle.url}/install"
        q_wait = 10.0 if queue_wait is None else max(0.0, float(queue_wait))
        payload = {"packages": packages, "queue_wait": q_wait}
        headers = {_SANDBOX_TOKEN_HEADER: handle.token} if handle.token else {}
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=q_wait + timeout),
                ) as resp:
                    data = await resp.json()
            except asyncio.TimeoutError:
                return {
                    "ok": False,
                    "stdout": "",
                    "stderr": "",
                    "error": f"Package installation timed out after {timeout} seconds",
                }
            except Exception as exc:
                return {
                    "ok": False,
                    "stdout": "",
                    "stderr": "",
                    "error": f"Failed to communicate with sandbox: {exc}",
                }

        stdout = str(data.get("stdout", ""))
        stderr = str(data.get("stderr", ""))
        error = data.get("error")
        ok = bool(data.get("ok", False)) and error is None

        if len(stdout) > _MAX_PIP_OUTPUT:
            stdout = stdout[:_MAX_PIP_OUTPUT] + "\n...[stdout truncated]"
        if len(stderr) > _MAX_PIP_OUTPUT:
            stderr = stderr[:_MAX_PIP_OUTPUT] + "\n...[stderr truncated]"

        handle.last_used_at = time.time()
        self._schedule_cleanup(handle)

        return {
            "ok": ok,
            "stdout": stdout,
            "stderr": stderr,
            "error": error,
        }

    async def jobs_in(
        self,
        handle: SandboxHandle,
        payload: Dict[str, Any],
        *,
        timeout: int = 60,
    ) -> Dict[str, Any]:
        """Forward a job-management action to the sandbox /jobs endpoint.

        The payload already carries action/job_id/etc.; the response is
        passed through untouched. 'wait' may block server-side, so the
        caller passes a client timeout covering payload['timeout'] + slack.
        """
        url = f"{handle.url}/jobs"
        headers = {_SANDBOX_TOKEN_HEADER: handle.token} if handle.token else {}
        status = 0
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as resp:
                    status = resp.status
                    data = await resp.json()
            except asyncio.TimeoutError:
                return {"ok": False, "error": f"Sandbox job request timed out after {timeout} seconds"}
            except Exception as exc:
                return {"ok": False, "error": f"Failed to communicate with sandbox: {exc}"}

        handle.last_used_at = time.time()
        self._schedule_cleanup(handle)

        if not isinstance(data, dict):
            return {"ok": False, "error": f"Sandbox returned invalid job response (HTTP {status})"}
        data.setdefault("ok", status < 400)
        return data
