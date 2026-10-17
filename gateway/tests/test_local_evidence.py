"""Evidence index: mining stores the user's own messages (secret-filtered) and memory_search returns them as
from_sources; forget deletes them."""
from __future__ import annotations

import json

import pytest
from fastmcp import Client

from hub_gateway import local, local_discover as ld, local_evidence as le_, local_mine as lm
from hub_gateway.local_ui import UI

from test_local_mine import CountingLLM, FakeEngine


def test_chunks_keep_message_boundaries():
    b = "".join(f"message {i} " + "z" * 300 + "\n---\n" for i in range(5))
    cs = le_.chunks(b)
    assert len(cs) == 3 and all(c.startswith("message") for c in cs)


@pytest.fixture
def hub(tmp_path, monkeypatch):
    monkeypatch.setenv("MNEMOS_OLLAMA", "0")
    monkeypatch.setenv("MNEMOS_BUILTIN", "0")
    monkeypatch.delenv("HUB_OLLAMA_URL", raising=False)
    monkeypatch.delenv("HUB_EMBED_URL", raising=False)
    home = tmp_path / "mnemos"
    local.init_home(home, ["personal", "work"], "Ana")
    servers = local.build_all(home, 8765)
    return UI(servers, home, ""), servers


async def test_mining_feeds_search(tmp_path, hub):
    ui, servers = hub
    h = tmp_path / "home"
    proj = h / ".claude" / "projects" / "-Users-ana-code-atlas"
    proj.mkdir(parents=True)
    lines = [{"type": "user", "message": {"role": "user", "content": t}} for t in
             ["We moved the staging database to Hetzner Falkenstein last spring.", "password=hunter2hunter please",
              "Lunch ideas for Friday?"]]
    (proj / "s.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n")

    async def ev(src, b):
        await le_.add_batch(ui.home, src, b)
    d = ld.Discover(ui, h, engine=FakeEngine(CountingLLM()))
    m = lm.Miner(ui, d, on_batch=ev)
    src = next(s for s in (await d.scan({}))["sources"] if s["kind"] == "claude_code")
    await m.h_start({"sources": [{"id": src["id"], "context": "work"}]})
    await m.task
    assert (await m.h_status({}))["evidence"]["work"] == 1
    async with Client(servers["work"]) as c:
        r = (await c.call_tool("memory_search", {"query": "where is the staging database"})).data
        assert "Falkenstein" in r["from_sources"][0]["text"] and "hunter2" not in json.dumps(r)
        r = (await c.call_tool("memory_search", {"query": "staging database", "include_sources": False})).data
        assert "from_sources" not in r
    async with Client(servers["personal"]) as c:  # other contexts never see it
        assert "from_sources" not in (await c.call_tool("memory_search", {"query": "staging database"})).data
    await m.h_forget({})
    assert le_.count(ui.home, "work") == 0
