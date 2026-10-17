"""Apple Intelligence engine through a fake helper speaking the fm-helper JSON-lines protocol."""
from __future__ import annotations

import json
import sys

import pytest

from hub_gateway import local_engine as le

FAKE = r'''import json, sys
print(json.dumps({"available": AVAILABLE, "reason": "" if AVAILABLE else "appleIntelligenceNotEnabled"}), flush=True)
if not AVAILABLE:
    sys.exit(0)
for line in sys.stdin:
    req = json.loads(line)
    print(json.dumps({"text": "```json\n" + json.dumps({"facts": [{"text": "Uses Tauri", "n": len(req["prompt"])}]}) + "\n```"}), flush=True)
'''


def helper(tmp_path, available: bool):
    (tmp_path / "fake_fm.py").write_text(f"AVAILABLE = {available}\n" + FAKE)
    p = tmp_path / "fm-helper"
    p.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{tmp_path / "fake_fm.py"}"\n')
    p.chmod(0o755)
    return p


@pytest.fixture
def env(tmp_path, monkeypatch):
    for k, v in {"MNEMOS_OLLAMA": "0", "MNEMOS_NO_KEYCHAIN": "1", "MNEMOS_BUILTIN": "0"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("MNEMOS_DISCOVER_LLM", raising=False)
    monkeypatch.setattr(le, "_is_mac", lambda: True)
    return tmp_path


async def test_apple_used_for_import_only(env, monkeypatch):
    monkeypatch.setenv("MNEMOS_FM_HELPER", str(helper(env, True)))
    e = le.Engine(env)
    llm = e.llm("import")
    assert llm.kind == "apple" and e.llm("hygiene") is None  # cleanup judge stays on Qwen3 / Ollama
    d = await llm.json("find facts")
    assert d["facts"][0]["text"] == "Uses Tauri"
    assert (await llm.json("again"))["facts"]  # same process, second request
    st = e.status()
    assert st["apple"] == {"choice": "auto", "available": True, "reason": ""}
    assert st["active"]["import"]["kind"] == "apple"
    await e.h_apple({"choice": "off"})
    assert e.llm("import") is None and json.loads((env / "engine.json").read_text())["apple"] == "off"


def test_apple_unavailable_falls_back_to_rules(env, monkeypatch):
    monkeypatch.setenv("MNEMOS_FM_HELPER", str(helper(env, False)))
    e = le.Engine(env)
    assert e.llm("import") is None
    assert e.status()["apple"]["reason"] == "appleIntelligenceNotEnabled"


def test_no_helper_off_mac(env, monkeypatch):
    monkeypatch.setattr(le, "_is_mac", lambda: False)
    assert le.fm_helper() is None
    monkeypatch.setattr(le, "_is_mac", lambda: True)
    monkeypatch.setenv("MNEMOS_FM_HELPER", "0")
    assert le.fm_helper() is None
