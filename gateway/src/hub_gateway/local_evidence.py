"""Evidence index: the user's own messages to AI tools, as read by full-history mining, kept searchable (keyword FTS5 +
embeddings, fused like notes) so memory_search can show "what you said" next to saved notes. 0.5 benchmark: raw
message chunks answered more questions than any set of extracted facts (6/16 vs 5/16, 16/16 coverage).

One SQLite file per context: <home>/data/<ctx>/evidence.sqlite (same schema as the notes index; dataset = source
kind). Only text that already passed the secret filter is stored. Settings → "Forget source messages" deletes it.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .local_search import LocalIndex

CHUNK = 800
KINDS = ["claude_code", "cursor_chats", "chatgpt_zip", "claude_zip"]
_IDX: dict[str, LocalIndex] = {}


def get(home: Path, ctx: str, embedder: Any = None) -> LocalIndex:
    p = str(Path(home) / "data" / ctx / "evidence.sqlite")
    if p not in _IDX:
        if embedder is None:
            from . import local_engine
            embedder = local_engine.EmbedProxy(local_engine.get(home))
        _IDX[p] = LocalIndex(p, embedder)
    return _IDX[p]


def chunks(batch: str, size: int = CHUNK) -> list[str]:
    """Messages of a mining batch ("...\\n---\\n" separated) grouped into ~size-char chunks, message boundaries kept."""
    out, cur = [], ""
    for m in (x.strip() for x in batch.split("\n---\n")):
        if not m:
            continue
        if cur and len(cur) + len(m) > size:
            out.append(cur.strip())
            cur = ""
        cur += m + "\n"
    if cur.strip():
        out.append(cur.strip())
    return out


async def add_batch(home: Path, src: dict, batch: str) -> int:
    idx = get(home, src["context"])
    n = 0
    for c in chunks(batch):
        idx.put("ev-" + hashlib.sha1(c.encode("utf-8", "ignore")).hexdigest()[:20], src["kind"], c, None)
        n += 1
    await idx._embed_missing()
    return n


async def search(home: Path, ctx: str, query: str, k: int = 3) -> list[dict[str, Any]]:
    p = Path(home) / "data" / ctx / "evidence.sqlite"
    if not p.exists():
        return []
    hits = await get(home, ctx).search(query, KINDS, k, "hybrid")
    return [{"source": h["dataset"], "text": h["text"][:600], "match": h["match"]} for h in hits]


def count(home: Path, ctx: str) -> int:
    p = Path(home) / "data" / ctx / "evidence.sqlite"
    if not p.exists():
        return 0
    idx = get(home, ctx)
    with idx._lock, idx._conn() as c:
        return int(c.execute("SELECT count(*) FROM docs").fetchone()[0])


def forget(home: Path, ctxs: list[str]) -> int:
    n = 0
    for ctx in ctxs:
        p = Path(home) / "data" / ctx / "evidence.sqlite"
        _IDX.pop(str(p), None)
        for f in (p, p.with_name(p.name + "-wal"), p.with_name(p.name + "-shm")):
            if f.exists():
                f.unlink()
                n += 1
    return n
