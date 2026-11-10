"""Tests for the server-selection dispatch in ``app.server``."""
import sys
import types
from types import SimpleNamespace

from app import server as server_mod


def _config(api_server="granian", api_reload=False):
    return SimpleNamespace(
        api_server=api_server,
        api_reload=api_reload,
        api_host="127.0.0.1",
        api_port=8000,
        log_level="INFO",
        instructions_file=server_mod.ROOT_DIR / "instructions.yaml",
        knowledge_base_root=server_mod.ROOT_DIR / "kb",
        bootstrap_default_runtime_data=lambda: None,
    )


def _patch_main(monkeypatch, config, ran):
    monkeypatch.setattr(server_mod, "_ensure_supported_python", lambda: None)
    monkeypatch.setattr(server_mod, "AppConfig", lambda: config)
    monkeypatch.setattr(server_mod, "_run_granian", lambda c: ran.setdefault("granian", True))
    monkeypatch.setattr(server_mod, "_run_uvicorn", lambda c: ran.setdefault("uvicorn", True))


def test_main_runs_granian_by_default(monkeypatch):
    ran = {}
    _patch_main(monkeypatch, _config("granian"), ran)
    server_mod.main()
    assert ran == {"granian": True}


def test_main_falls_back_to_uvicorn_on_unknown_server(monkeypatch, caplog):
    ran = {}
    _patch_main(monkeypatch, _config("weird"), ran)
    with caplog.at_level("WARNING"):
        server_mod.main()
    assert ran == {"uvicorn": True}
    assert "Unknown API_SERVER" in caplog.text


def test_main_runs_uvicorn_when_configured(monkeypatch):
    ran = {}
    _patch_main(monkeypatch, _config("uvicorn"), ran)
    server_mod.main()
    assert ran == {"uvicorn": True}


def test_run_uvicorn_passes_reload_dirs(monkeypatch):
    captured = {}
    monkeypatch.setattr(server_mod.uvicorn, "run", lambda *a, **kw: captured.update(kw))
    server_mod._run_uvicorn(_config(api_server="uvicorn", api_reload=True))
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8000
    assert captured["reload"] is True
    assert captured["reload_dirs"]


def test_run_granian_uses_asgi_interface(monkeypatch):
    captured = {}

    class FakeInterfaces:
        ASGI = "asgi"

    class FakeGranian:
        def __init__(self, target, **kwargs):
            captured["target"] = target
            captured.update(kwargs)

        def serve(self):
            captured["served"] = True

    fake_granian = types.ModuleType("granian")
    fake_granian.Granian = FakeGranian
    fake_constants = types.ModuleType("granian.constants")
    fake_constants.Interfaces = FakeInterfaces
    monkeypatch.setitem(sys.modules, "granian", fake_granian)
    monkeypatch.setitem(sys.modules, "granian.constants", fake_constants)

    server_mod._run_granian(_config(api_reload=True))

    assert captured["served"] is True
    assert captured["target"] == "app.main:app"
    assert captured["interface"] == "asgi"
    assert captured["workers"] == 1
    assert captured["reload"] is True
    assert captured["reload_paths"]
