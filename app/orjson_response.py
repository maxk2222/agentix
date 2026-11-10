"""orjson-backed JSON response.

FastAPI's bundled ``ORJSONResponse`` is deprecated since 0.162 (Pydantic now
serializes response-model payloads on the Rust side). Endpoints here mostly
return bare dicts without declared response models, so a fast default
serializer is still wanted — this subclass keeps orjson without the
deprecation warning that fires on every instantiation of the bundled class.
"""
from __future__ import annotations

from typing import Any

import orjson
from fastapi.responses import JSONResponse


class ORJSONResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        return orjson.dumps(content)
