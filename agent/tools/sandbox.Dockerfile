FROM python:3.11-slim-bookworm

# iptables is used by the egress sidecar: a one-shot helper container that
# shares the sandbox's netns (network_mode=container:) and installs the
# AGENTIX_EGRESS chain — the sandbox itself runs cap_drop ALL and cannot
# modify the rules afterwards.
RUN apt-get update \
    && apt-get install -y --no-install-recommends iptables \
    && rm -rf /var/lib/apt/lists/*

# Package installs inside the sandbox go through `uv pip install` (pip
# fallback in sandbox_server.py when the binary is absent).
RUN pip install --no-cache-dir "uv~=0.13.0"

RUN groupadd -r sandboxuser && useradd -r -g sandboxuser sandboxuser

COPY --chown=sandboxuser:sandboxuser sandbox_server.py /sandbox_server.py

# HOME is unset for the `useradd -r` system user — pip cache, matplotlib
# (MPLCONFIGDIR) and many libraries fail without writable config dirs.
ENV HOME=/tmp \
    MPLCONFIGDIR=/tmp/mplconfig \
    XDG_CACHE_HOME=/tmp/.cache \
    SANDBOX_PACKAGES_DIR=/var/sandbox-packages

WORKDIR /tmp

USER sandboxuser

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import socket; s=socket.socket(socket.AF_INET,socket.SOCK_STREAM); s.settimeout(2); s.connect(('127.0.0.1',8080)); s.close()" || exit 1

CMD ["python", "/sandbox_server.py"]
