"""Own lightweight graph for the Graph view (no Cognee in the app): the local model reads the notes, 12 at a time, and
writes (subject, relation, object) triples, kept in <home>/data/<ctx>/graph.sqlite per note. Built in the background
when the Graph view is opened, only for notes that are new or changed; triples of deleted notes go away. Without a
local model the view keeps the name/tag graph (local_ui.build_graph). Only Ollama or the built-in model are used:
notes never go to an API or another app for this.
"""
from __future__ import annotations

import asyncio
import collections
import hashlib
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any

log = logging.getLogger("hub_gateway.local_graph")
TRIPLES = ("Extract the entities and relations stated in these notes about a person (people, companies, projects, "
           "products, tools, places). Use short canonical names. "
           'Reply JSON only: {"triples": [["subject", "relation", "object", <note number>], ...]}\n\n')
PER_CALL = 12
MAX_NOTES = 600


def _h(t: str) -> str:
    return hashlib.sha1(t.encode("utf-8", "ignore")).hexdigest()[:16]


class Graph:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self.task: asyncio.Task | None = None
        self.progress: dict[str, int] = {}
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._c() as c:
            c.executescript("""CREATE TABLE IF NOT EXISTS notes (id TEXT PRIMARY KEY, h TEXT);
                CREATE TABLE IF NOT EXISTS triples (note TEXT, s TEXT, r TEXT, o TEXT);
                CREATE INDEX IF NOT EXISTS t_note ON triples(note);""")

    def _c(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=10)

    def todo(self, items: list[dict]) -> list[dict]:
        live = {it["id"]: it for it in items if not it.get("obsolete")}
        with self._lock, self._c() as c:
            have = dict(c.execute("SELECT id, h FROM notes").fetchall())
            for gone in set(have) - set(live):
                c.execute("DELETE FROM notes WHERE id=?", (gone,))
                c.execute("DELETE FROM triples WHERE note=?", (gone,))
        return [it for i, it in live.items() if have.get(i) != _h(str(it.get("text", "")))][:MAX_NOTES]

    def store(self, notes: list[dict], triples: list[tuple[str, str, str, str]]) -> None:
        with self._lock, self._c() as c:
            for n in notes:
                c.execute("DELETE FROM triples WHERE note=?", (n["id"],))
                c.execute("INSERT OR REPLACE INTO notes VALUES (?,?)", (n["id"], _h(str(n.get("text", "")))))
            c.executemany("INSERT INTO triples VALUES (?,?,?,?)", triples)

    def triples(self) -> list[tuple[str, str, str, str]]:
        with self._lock, self._c() as c:
            return c.execute("SELECT note, s, r, o FROM triples").fetchall()

    async def build(self, items: list[dict], llm: Any) -> None:
        todo = self.todo(items)
        self.progress = {"done": 0, "total": len(todo)}
        for i in range(0, len(todo), PER_CALL):
            part = todo[i:i + PER_CALL]
            chunk = "\n".join(f"{j}. {str(n.get('text', ''))[:400]}" for j, n in enumerate(part))
            try:
                got = (await llm.json(TRIPLES + chunk)).get("triples") or []
            except Exception as e:  # noqa: BLE001 — try these notes again next time
                log.warning("graph batch skipped: %s", type(e).__name__)
                continue
            rows = []
            for x in got:
                if not (isinstance(x, list) and len(x) >= 3 and all(isinstance(y, str) and y.strip() for y in x[:3])):
                    continue
                k = x[3] if len(x) > 3 and isinstance(x[3], int) and 0 <= x[3] < len(part) else None
                if k is None:  # no note number: the note naming both ends, else the object, else the subject
                    low = [str(n.get("text", "")).lower() for n in part]
                    score = [2 * (x[2].lower() in t) + (x[0].lower() in t) for t in low]
                    k = max(range(len(part)), key=lambda j: score[j]) if part and max(score) else None
                if k is not None:
                    rows.append((part[k]["id"], " ".join(x[0].split())[:60], " ".join(x[1].split())[:40],
                                 " ".join(x[2].split())[:60]))
            self.store(part, rows)
            self.progress["done"] = min(len(todo), i + PER_CALL)

    def start(self, items: list[dict], llm: Any) -> bool:
        """Build in the background (one run at a time). True when it is (still) building."""
        if self.task and not self.task.done():
            return True
        if not llm or not self.todo(items):
            return False
        self.task = asyncio.create_task(self.build(items, llm))
        return True


_G: dict[str, Graph] = {}


def get(home: Path, ctx: str) -> Graph:
    p = Path(home) / "data" / ctx / "graph.sqlite"
    if str(p) not in _G:
        _G[str(p)] = Graph(p)
    return _G[str(p)]


def view(items: list[dict], triples: list[tuple[str, str, str, str]], max_nodes: int = 60, max_edges: int = 160) -> dict:
    """Graph view data from model triples, same shape as local_ui.build_graph (+ relations)."""
    live = {it["id"]: it for it in items if not it.get("obsolete")}
    canon: dict[str, str] = {}

    def key(name: str) -> str:
        k = name.casefold()
        canon.setdefault(k, name)
        return k
    notes_of: dict[str, set[str]] = collections.defaultdict(set)
    pairs: collections.Counter = collections.Counter()
    rels: dict[tuple[str, str], list[str]] = collections.defaultdict(list)
    for note, s, r, o in triples:
        if note not in live:
            continue
        a, b = key(s), key(o)
        notes_of[a].add(note)
        notes_of[b].add(note)
        if a != b:
            e = tuple(sorted((a, b)))
            pairs[e] += 1
            line = f"{canon[a]} — {r} — {canon[b]}"
            if line not in rels[e]:
                rels[e].append(line)
    keep = sorted(notes_of, key=lambda k: -len(notes_of[k]))[:max_nodes]
    ks = set(keep)
    edges = [(e, w) for e, w in pairs.most_common() if e[0] in ks and e[1] in ks][:max_edges]
    used = {n for k in keep for n in sorted(notes_of[k])[:12]}
    return {"notes": len(live), "source": "model",
            "nodes": [{"id": k, "label": canon[k], "kind": "name", "count": len(notes_of[k]),
                       "notes": sorted(notes_of[k])[:12]} for k in keep],
            "edges": [{"source": a, "target": b, "weight": w, "label": "; ".join(rels[(a, b)][:3])} for (a, b), w in edges],
            "texts": {i: str(live[i].get("text", ""))[:600] for i in used}}
