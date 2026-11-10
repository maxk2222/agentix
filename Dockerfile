# syntax=docker/dockerfile:1

# uv export runs on the build platform: the uv binary is native code that
# crashes under QEMU user-mode emulation, so it must run on the native arch.
# The exported requirements file is universal (env markers keep it valid for
# every target platform).
FROM --platform=$BUILDPLATFORM python:3.11-slim-bookworm AS pydeps
WORKDIR /deps
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir "uv~=0.13.0" \
    && uv export --locked --no-dev --format requirements-txt -o /deps.txt

FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UVICORN_PORT=8000 \
    NLTK_DATA=/usr/share/nltk_data

# Toggle to pre-download NLTK resources at build time.
# Set to "false" if your build environment has no outbound HTTPS.
ARG NLTK_DOWNLOAD=true

# Git commit the image was built from; surfaced via GET /health.
ARG GIT_COMMIT_SHA=unknown
ENV GIT_COMMIT_SHA=$GIT_COMMIT_SHA

# Build-time hardening against transient apt failures: Debian mirrors roll
# bookworm-security periodically, so a Release file can be dated a few minutes
# ahead of a drifting runner clock ("is not valid yet") or briefly mismatch
# package hashes mid-roll. Disable the date/expiry checks (build-time only —
# the image contents are unaffected) and retry a few times before failing.
RUN ok=""; \
    for i in 1 2 3 4 5; do \
        apt-get update \
            -o Acquire::Check-Date=false \
            -o Acquire::Check-Valid-Until=false \
            -o Acquire::Max-FutureTime=600 \
            && ok=1 && break; \
        echo "apt-get update failed (attempt $i), retrying in 20s"; \
        sleep 20; \
    done; \
    [ -n "$ok" ] || exit 1; \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        build-essential \
        libmagic1 \
        poppler-utils \
        tesseract-ocr \
        tesseract-ocr-deu \
        tesseract-ocr-eng \
        postgresql-client \
        curl \
        ffmpeg \
        espeak-ng \
        espeak-ng-data \
        libgl1 \
        libglib2.0-0 \
        libreoffice-writer \
        libreoffice-impress \
        fonts-liberation \
        fonts-dejavu \
        graphviz \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# pip (not uv) installs in the target stage: the foreign-arch uv binary would
# segfault under QEMU emulation in cross-builds, while pip is pure Python.
COPY --from=pydeps /deps.txt /tmp/deps.txt
RUN pip install --no-cache-dir -r /tmp/deps.txt

# Pre-download common NLTK resources used by unstructured so startup doesn't need network access.
RUN set -e && \
    if [ "${NLTK_DOWNLOAD}" = "true" ]; then \
        mkdir -p "${NLTK_DATA}" && \
        { \
            echo 'import importlib'; \
            echo 'import os'; \
            echo ''; \
            echo 'download_dir = os.environ.get("NLTK_DATA", "/usr/share/nltk_data")'; \
            echo ''; \
            echo 'try:'; \
            echo '    nltk = importlib.import_module("nltk")'; \
            echo 'except Exception as exc:'; \
            echo '    print(f"NLTK not installed; skipping NLTK data download: {exc}")'; \
            echo 'else:'; \
            echo '    resources = ['; \
            echo '        "punkt",'; \
            echo '        "punkt_tab",'; \
            echo '        "averaged_perceptron_tagger",'; \
            echo '        "averaged_perceptron_tagger_eng",'; \
            echo '    ]'; \
            echo '    for resource in resources:'; \
            echo '        try:'; \
            echo '            ok = nltk.download(resource, download_dir=download_dir, quiet=True)'; \
            echo '            print(f"nltk.download({resource!r}) -> {ok}")'; \
            echo '        except Exception as exc:'; \
            echo '            print(f"nltk.download({resource!r}) failed: {exc}")'; \
        } > /tmp/download_nltk.py && \
        python /tmp/download_nltk.py && \
        rm /tmp/download_nltk.py; \
    else \
        echo "Skipping NLTK data download (NLTK_DOWNLOAD=false)"; \
    fi

RUN groupadd -r appuser && useradd -r -g appuser -d /home/appuser appuser \
    && mkdir -p /home/appuser \
    && chown -R appuser:appuser /home/appuser
ENV HOME=/home/appuser

COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser agent ./agent
COPY --chown=appuser:appuser defaults ./defaults
COPY --chown=appuser:appuser widget/dist ./widget/dist
COPY --chown=appuser:appuser dashboard/dist ./dashboard/dist

USER appuser

HEALTHCHECK --interval=60s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000
CMD ["python", "-m", "app.server"]