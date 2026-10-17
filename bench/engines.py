"""Benchmark engines for Mnemos local mode. Same interface as hub_gateway.local_engine LLMs:
async json(prompt) -> dict, .name, .kind, .mode ("classify" = judge rule sentences, "extract" = write facts)."""
from __future__ import annotations
import asyncio, json, os, subprocess, sys, threading, time
from pathlib import Path
import httpx
from hub_gateway.local_engine import _json, OllamaLLM

MODELS = Path(os.environ.get("BENCH_MODELS", "/tmp/mnemos-bench/models"))
GGUF = {  # name: (file, prompt suffix)
    "qwen2.5-1.5b": ("qwen25-1.5b.gguf", ""),
    "qwen3-1.7b": ("qwen3-1.7b.gguf", " /no_think"),
    "qwen3-4b": ("qwen3-4b.gguf", ""),
    "gemma3-4b": ("gemma3-4b.gguf", ""),
}


class GGUFLLM:
    kind = "builtin"

    def __init__(self, key: str, mode: str = "classify", ctx: int = 4096) -> None:
        from llama_cpp import Llama
        f, self.suffix = GGUF[key]
        self.name, self.mode, self._lock = f"llama.cpp · {key}", mode, threading.Lock()
        self.m = Llama(model_path=str(MODELS / f), n_ctx=ctx, n_gpu_layers=-1, verbose=False,
                       n_threads=max(2, (os.cpu_count() or 4) - 2))
        self.calls = 0

    def _run(self, prompt: str) -> str:
        with self._lock:
            self.calls += 1
            out = self.m.create_chat_completion(messages=[{"role": "user", "content": prompt + self.suffix}],
                                                temperature=0, max_tokens=1024, response_format={"type": "json_object"})
        t = out["choices"][0]["message"]["content"] or ""
        return t.split("</think>")[-1]

    async def json(self, prompt: str) -> dict:
        return _json(await asyncio.to_thread(self._run, prompt))


class OpenAICompat:
    """mlx_lm.server or any OpenAI-compatible local server."""
    kind = "builtin"

    def __init__(self, url: str, model: str, name: str, mode: str = "extract") -> None:
        self.url, self.model, self.name, self.mode, self.calls = url, model, name, mode, 0

    async def json(self, prompt: str) -> dict:
        self.calls += 1
        async with httpx.AsyncClient(timeout=300, trust_env=False) as c:
            r = await c.post(self.url + "/v1/chat/completions", json={
                "model": self.model, "temperature": 0, "max_tokens": 1024,
                "messages": [{"role": "user", "content": prompt + "\nReply with the JSON object only."}]})
            r.raise_for_status()
        t = r.json()["choices"][0]["message"]["content"] or ""
        return _json(t.split("</think>")[-1])


class AppleFM:
    """Apple on-device Foundation Model through bench/fm-helper (one JSON line per request)."""
    kind = "builtin"

    def __init__(self, helper: str, mode: str = "extract") -> None:
        self.p = subprocess.Popen([helper], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        first = json.loads(self.p.stdout.readline())
        if not first.get("available"):
            raise RuntimeError("Apple FM unavailable: " + first.get("reason", ""))
        self.name, self.mode, self._lock, self.calls, self.errors = "Apple Foundation Model", mode, threading.Lock(), 0, 0

    def _run(self, prompt: str) -> str:
        with self._lock:
            self.calls += 1
            self.p.stdin.write(json.dumps({"prompt": prompt + "\nReply with the JSON object only, no markdown."}) + "\n")
            self.p.stdin.flush()
            r = json.loads(self.p.stdout.readline())
        if "error" in r:
            self.errors += 1
            return "{}"
        t = r["text"].strip()
        if t.startswith("```"):
            t = t.strip("`").removeprefix("json")
        return t

    async def json(self, prompt: str) -> dict:
        return _json(await asyncio.to_thread(self._run, prompt))


def ollama(model: str = "llama3.1:8b", mode: str = "extract") -> OllamaLLM:
    o = OllamaLLM("http://127.0.0.1:11434", model)
    o.mode = mode
    return o


def make(spec: str):
    """spec: rules | <gguf key>:<mode> | mlx-qwen3-4b:<mode> | apple-fm:<mode> | ollama:<mode>"""
    if spec == "rules":
        return None
    key, mode = spec.split(":")
    if key in GGUF:
        return GGUFLLM(key, mode)
    if key == "mlx-qwen3-4b":
        return OpenAICompat("http://127.0.0.1:8091", os.environ.get("MLX_MODEL", "/tmp/mnemos-bench/mlx-q3-4b"),
                            "MLX · qwen3-4b", mode)
    if key == "apple-fm":
        return AppleFM(os.environ.get("FM_HELPER", "/tmp/mnemos-bench/fm-helper"), mode)
    if key == "ollama":
        return ollama(mode=mode)
    raise SystemExit("unknown engine " + spec)


def rss_of(pattern: str) -> int:
    """Total RSS (bytes) of processes matching pattern (for out-of-process engines)."""
    out = subprocess.run(["ps", "-axo", "rss=,command="], capture_output=True, text=True).stdout
    return sum(int(l.split(None, 1)[0]) * 1024 for l in out.splitlines() if pattern in l and "ps -axo" not in l)
