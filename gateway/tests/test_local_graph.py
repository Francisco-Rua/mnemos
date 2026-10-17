"""Own graph: model triples per note, built in the background, incremental, deleted notes drop out; no model →
the names/tags graph."""
from __future__ import annotations

from hub_gateway import local_graph as lg
from hub_gateway.local_ui import build_graph

NOTES = [{"id": "n1", "text": "Ana works at Acme on the Atlas project."},
         {"id": "n2", "text": "Atlas runs on Postgres 16."},
         {"id": "n3", "text": "Ana's brother Tomás lives in Córdoba."}]


class LLM:
    name = "Built-in · fake"

    def __init__(self):
        self.calls = 0

    async def json(self, prompt):
        self.calls += 1
        out = []
        if "Acme" in prompt:
            out += [["Ana", "works at", "Acme", 0], ["Ana", "works on", "Atlas", 0]]
        if "Postgres" in prompt:
            out += [["Atlas", "uses", "Postgres 16"]]  # no note number: matched by name
        if "Tomás" in prompt:
            out += [["Ana", "sibling of", "Tomás", 2], ["bad"], ["x", "y", "", 1]]
        return {"triples": out}


async def test_triples_incremental_and_view(tmp_path):
    g = lg.Graph(tmp_path / "graph.sqlite")
    llm = LLM()
    assert g.start(NOTES, llm)
    await g.task
    v = lg.view(NOTES, g.triples())
    names = {n["label"] for n in v["nodes"]}
    assert {"Ana", "Acme", "Atlas", "Postgres 16", "Tomás"} <= names and v["source"] == "model"
    atlas = next(e for e in v["edges"] if {e["source"], e["target"]} == {"atlas", "postgres 16"})
    assert "uses" in atlas["label"] and "n2" in next(n for n in v["nodes"] if n["id"] == "atlas")["notes"]
    calls = llm.calls
    assert not g.start(NOTES, llm) and llm.calls == calls  # nothing new: no model call
    edited = [NOTES[0], {"id": "n2", "text": "Atlas moved to SQLite."}]  # n3 deleted, n2 changed
    assert g.start(edited, llm)
    await g.task
    assert llm.calls == calls + 1
    v = lg.view(edited, g.triples())
    assert "Tomás" not in {n["label"] for n in v["nodes"]}


def test_no_model_keeps_names_graph(tmp_path):
    g = lg.Graph(tmp_path / "graph.sqlite")
    assert not g.start(NOTES, None) and g.triples() == []
    assert "nodes" in build_graph(NOTES)
