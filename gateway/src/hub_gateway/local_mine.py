"""Full-history mining: the import model reads EVERY message of the chosen sources in the background, batch by
batch, instead of the newest few batches a preview has time for (0.5 benchmark: 60 batches → ~5x the useful facts).

Incremental and resumable: messages are batched per file in a fixed order, each batch is identified by the hash of
its text, and the hashes already read are kept in <home>/mining.json (written after every batch). A restart resumes
where it stopped; mining again later only reads batches that are new (new chats, new messages at the end of a chat).
Found facts are candidates: nothing is saved until the user approves them (Import → Full history → Save selected),
and saving goes through memory_save, so its duplicate check (words and meaning) still applies.

Only paths from the user's last "Scan this Mac" can be mined (the same rule as the preview), and secrets never reach
the model (local_discover.has_secret). Needs a model (Built-in, Ollama, Apple or an API the user enabled for import).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Callable

from . import local_discover as ld
from . import local_evidence
from . import local_import as li

log = logging.getLogger("hub_gateway.local_mine")
BATCH = 5000
MAX_CANDIDATES = 5000
MAX_SEEN = 200_000


def batches(msgs: list[str], size: int = BATCH) -> list[str]:
    """Same batching as local_discover.llm_facts (messages trimmed, secrets dropped), deterministic."""
    out, cur = [], ""
    for m in msgs:
        m = m.strip()[:1200]
        if not m or ld.has_secret(m):
            continue
        if cur and len(cur) + len(m) > size:
            out.append(cur)
            cur = ""
        cur += m + "\n---\n"
    if cur:
        out.append(cur)
    return out


def _h(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()[:16]


class Miner:
    def __init__(self, ui: Any, discover: Any, on_batch: Callable[[dict, str], Any] | None = None) -> None:
        self.ui, self.discover, self.on_batch = ui, discover, on_batch
        self.file = Path(ui.home) / "mining.json"
        self.task: asyncio.Task | None = None
        self.st = self._load()
        self.live: dict[str, Any] = {}

    # state
    def _load(self) -> dict:
        try:
            d = json.loads(self.file.read_text())
        except (OSError, json.JSONDecodeError):
            d = {}
        d.setdefault("state", "idle")  # idle | running | paused | done | error
        d.setdefault("sources", [])  # [{id, kind, label, paths, context}]
        d.setdefault("seen", [])  # batch hashes already read
        d.setdefault("candidates", [])  # [{id, text, context, source, source_label, status}]
        return d

    def _save(self) -> None:
        tmp = self.file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.st))
        os.replace(tmp, self.file)

    # work
    def _plan(self) -> list[tuple[dict, str]]:
        """Every unread batch of every source, oldest file first (reading only; nothing goes to a model)."""
        seen, todo, total = set(self.st["seen"]), [], 0
        far = time.monotonic() + 10 ** 9
        for src in self.st["sources"]:
            for p in src["paths"]:
                _notes, msgs = ld.read_source({**src, "paths": [p]}, far)
                for b in batches(msgs):
                    total += 1
                    if _h(b) not in seen:
                        todo.append((src, b))
        self.live.update(total=total, done=total - len(todo))
        return todo

    async def _run(self) -> None:
        try:
            self.live = {"phase": "Reading your history", "done": 0, "total": 0}
            todo = await asyncio.to_thread(self._plan)
            llm = await asyncio.to_thread(self.discover.engine.llm, "import")
            if llm is None:
                raise ValueError("full-history mining needs a model: Settings → Improve my memory (or Ollama)")
            self.st["engine"] = {"kind": llm.kind, "name": llm.name}
            known: dict[str, list[str]] = {}
            for src, b in todo:
                if self.st["state"] != "running":
                    break
                ctx = src["context"]
                if ctx not in known:
                    known[ctx] = await self.ui._existing(ctx)
                self.live["phase"] = f"Reading {src['label']} ({llm.name})"
                facts = await ld.batch_facts(llm, b)
                pool = known[ctx] + [c["text"] for c in self.st["candidates"] if c["context"] == ctx]
                pv = li.preview([f for f in facts if not ld.has_secret(f)], pool)
                for it in pv["items"]:
                    if not it["duplicate"] and len(self.st["candidates"]) < MAX_CANDIDATES:
                        self.st["candidates"].append({"id": _h(ctx + it["text"]), "text": it["text"], "context": ctx,
                                                      "source": src["kind"], "source_label": src["label"],
                                                      "status": "new"})
                if self.on_batch:
                    try:
                        await self.on_batch(src, b)
                    except Exception as e:  # noqa: BLE001 — the evidence index is optional
                        log.warning("evidence index: %s", type(e).__name__)
                self.st["seen"] = (self.st["seen"] + [_h(b)])[-MAX_SEEN:]
                self.live["done"] = self.live.get("done", 0) + 1
                self._save()
                await asyncio.sleep(0)  # let the hub answer requests between batches
            if self.st["state"] == "running":
                self.st["state"] = "done"
                self.st["finished"] = time.time()
        except Exception as e:  # noqa: BLE001
            self.st["state"], self.st["error"] = "error", str(e)[:300]
        self._save()

    def _go(self) -> None:
        self.st["state"] = "running"
        self.st.pop("error", None)
        self._save()
        self.task = asyncio.create_task(self._run())

    def resume_on_start(self) -> None:
        """Called from the hub's lifespan: a mining run that was going when the app quit continues."""
        if self.st["state"] == "running" and self.st["sources"]:
            self._go()

    # hooks
    async def h_status(self, _b: dict) -> dict:
        c = self.st["candidates"]
        return {"state": self.st["state"], "error": self.st.get("error"), "engine": self.st.get("engine"),
                "sources": [{"label": s["label"], "context": s["context"]} for s in self.st["sources"]],
                "done": self.live.get("done"), "total": self.live.get("total"), "phase": self.live.get("phase"),
                "read_batches": len(self.st["seen"]), "found": len(c),
                "new": [x for x in c if x["status"] == "new"][:500],
                "saved": sum(1 for x in c if x["status"] == "saved"),
                "dismissed": sum(1 for x in c if x["status"] == "dismissed"),
                "evidence": {ctx: local_evidence.count(self.ui.home, ctx) for ctx in self.ui.servers}}

    async def h_start(self, b: dict) -> dict:
        if self.task and not self.task.done():
            raise ValueError("already mining")
        chosen = []
        for c in b.get("sources") or []:
            s = self.discover.sources.get(str((c or {}).get("id")))
            ctx = str((c or {}).get("context") or "")
            if s and ctx in self.ui.servers and s["kind"] not in ("claude_md", "agents_md", "cursor_rules", "obsidian"):
                chosen.append({"id": s["id"], "kind": s["kind"], "label": s["label"], "paths": s["paths"],
                               "context": ctx})
        if not chosen:
            raise ValueError("pick at least one chat source (scan again if the list is old)")
        self.st["sources"] = chosen
        self._go()
        return await self.h_status({})

    async def h_pause(self, _b: dict) -> dict:
        if self.st["state"] == "running":
            self.st["state"] = "paused"
            self._save()
        return await self.h_status({})

    async def h_resume(self, _b: dict) -> dict:
        if self.task and not self.task.done():
            raise ValueError("already mining")
        if not self.st["sources"]:
            raise ValueError("nothing to resume: start from Scan this Mac")
        self._go()
        return await self.h_status({})

    async def h_save(self, b: dict) -> dict:
        ids = {str(i) for i in b.get("ids") or []}
        pick = [c for c in self.st["candidates"] if c["id"] in ids and c["status"] == "new"]
        res = await self.discover.commit({"notes": pick})
        for c in pick:
            c["status"] = "saved"
        self._save()
        return res

    async def h_dismiss(self, b: dict) -> dict:
        ids = {str(i) for i in b.get("ids") or []}
        for c in self.st["candidates"]:
            if c["id"] in ids and c["status"] == "new":
                c["status"] = "dismissed"
        self._save()
        return await self.h_status({})

    async def h_forget(self, _b: dict) -> dict:
        """Delete the searchable copy of source messages (notes stay). Batches stay marked as read."""
        if self.task and not self.task.done():
            raise ValueError("pause the full-history reading first")
        local_evidence.forget(self.ui.home, list(self.ui.servers))
        return await self.h_status({})

    def install(self) -> None:
        h = self.ui.hooks
        h[("POST", "mine/forget-sources")] = self.h_forget
        h[("GET", "mine")] = self.h_status
        h[("POST", "mine/start")] = self.h_start
        h[("POST", "mine/pause")] = self.h_pause
        h[("POST", "mine/resume")] = self.h_resume
        h[("POST", "mine/save")] = self.h_save
        h[("POST", "mine/dismiss")] = self.h_dismiss

