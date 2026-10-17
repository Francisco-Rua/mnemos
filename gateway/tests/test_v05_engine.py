"""0.5 engine: model size by RAM, graceful 0.4 → 0.5 model migration, EmbeddingGemma task prompts."""
from __future__ import annotations

import hashlib
import os

import httpx

from hub_gateway import local_engine as le

BLOB = os.urandom(200_000)


def _m(role, name):
    return {"role": role, "name": name, "file": f"{name}.gguf", "size": len(BLOB),
            "sha256": hashlib.sha256(BLOB).hexdigest(), "url": f"https://models.test/{name}.gguf"}


def _setup(tmp_path, monkeypatch):
    monkeypatch.delenv("MNEMOS_MODELS_MANIFEST", raising=False)
    monkeypatch.setenv("MNEMOS_OLLAMA", "0")
    monkeypatch.setenv("MNEMOS_NO_KEYCHAIN", "1")
    monkeypatch.delenv("MNEMOS_BUILTIN", raising=False)
    monkeypatch.setenv("MNEMOS_LLM_SIZE", "4b")
    monkeypatch.setattr(le, "LLMS", {"4b": _m("llm", "new4b"), "1.7b": _m("llm", "new17")})
    monkeypatch.setattr(le, "EMBED", _m("embed", "emb"))
    monkeypatch.setattr(le, "LEGACY", [_m("llm", "old15")])
    monkeypatch.setattr(le, "runtime_ok", lambda: True)
    (tmp_path / "models").mkdir()
    for n in ("emb", "old15"):
        (tmp_path / "models" / f"{n}.gguf").write_bytes(BLOB)


def test_llm_size_by_ram(monkeypatch):
    monkeypatch.delenv("MNEMOS_LLM_SIZE", raising=False)
    monkeypatch.setattr(le, "ram_bytes", lambda: 8 * 1024 ** 3)
    assert le.llm_size() == "1.7b"
    monkeypatch.setattr(le, "ram_bytes", lambda: 16 * 1024 ** 3)
    assert le.llm_size() == "4b"
    monkeypatch.setenv("MNEMOS_LLM_SIZE", "1.7b")
    assert le.llm_size() == "1.7b"


async def test_old_model_keeps_working_then_is_replaced(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    e = le.Engine(tmp_path)
    assert e.builtin_ready() and e.llm("import").model == "old15.gguf"  # 0.4 users: no gap
    st = e.status()["builtin"]
    assert st["upgrade"] and not st["ready"]
    e.dl.transport = httpx.MockTransport(lambda req: httpx.Response(200, content=BLOB))
    await e.dl.run()
    assert e.dl.ready() and not (tmp_path / "models" / "old15.gguf").exists()
    assert e.llm("import").model == "new4b.gguf" and not e.status()["builtin"]["upgrade"]


async def test_autostart_only_when_user_said_yes(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    e = le.Engine(tmp_path)
    e.autostart()
    assert e.dl.task is None  # never asked → no download behind the user's back
    c = e.cfg()
    c["builtin"] = "on"
    e.save_cfg(c)
    e.dl.transport = httpx.MockTransport(lambda req: httpx.Response(200, content=BLOB))
    e.autostart()
    assert e.dl.task is not None
    await e.dl.task
    assert e.dl.ready()


async def test_embed_as_uses_task_prompts_and_falls_back():
    class Plain:
        async def embed(self, t):
            return [[1.0]] * len(t)

    class Gemma(Plain):
        async def embed_sim(self, t):
            return [[2.0]] * len(t)

    assert await le.embed_as(Plain(), ["a"], "sim") == [[1.0]]
    assert await le.embed_as(Gemma(), ["a"], "sim") == [[2.0]]
    assert await le.embed_as(Gemma(), ["a"], "doc") == [[1.0]]
    assert le.PROMPTS["sim"].startswith("task: sentence similarity") and le.SEM_DUP == 0.80


def test_no_classify_mode_left():
    from hub_gateway import local_discover as ld
    assert not hasattr(ld, "llm_filter") and not hasattr(ld, "CLASSIFY")
    assert le.BuiltinLLM.mode == "extract"
