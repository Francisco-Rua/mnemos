"""Import benchmark: Scan this Mac (read-only) + preview with one engine. Writes nothing to any Mnemos home.
usage: bench_import.py <engine spec> <out.json>"""
import asyncio, json, os, resource, sys, threading, time
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))
from hub_gateway import local_discover as ld
import engines

spec, out = sys.argv[1], sys.argv[2]
if os.environ.get("EXTRACT_CALLS"):  # coverage experiment: mine more of the history than the product's 6 batches
    import functools
    _orig = ld.llm_facts
    ld.llm_facts = functools.partial(_orig, calls=int(os.environ["EXTRACT_CALLS"]))
peak = {"ext": 0}
def sampler(pat):
    while True:
        peak["ext"] = max(peak["ext"], engines.rss_of(pat)); time.sleep(1)
pat = {"ollama": "ollama", "mlx-qwen3-4b": "mlx_lm", "apple-fm": "fm-helper"}.get(spec.split(":")[0])
if pat:
    threading.Thread(target=sampler, args=(pat,), daemon=True).start()

async def main():
    t0 = time.time()
    llm = engines.make(spec)
    load = time.time() - t0
    found = ld.scan(Path.home(), ["personal", "work"])
    for s in found:
        s["context"] = "personal"
    t = time.time()
    r = await ld.preview(found, {}, lambda p: None, llm=llm, budget=float(os.environ.get("BUDGET", "3600")))
    dt = time.time() - t
    items = [{"text": i["text"], "source": i["source"], "dup": bool(i["duplicate"])} for i in r["items"]]
    res = {"engine": spec, "load_s": round(load, 1), "preview_s": round(dt, 1), "timed_out": r["timed_out"],
           "new": r["new"], "dups": r["duplicates"], "calls": getattr(llm, "calls", 0),
           "errors": getattr(llm, "errors", 0),
           "rss_self_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // (1024 * 1024),
           "rss_ext_mb": peak["ext"] // (1024 * 1024),
           "by_source": {}, "items": items}
    for i in items:
        if not i["dup"]:
            res["by_source"][i["source"]] = res["by_source"].get(i["source"], 0) + 1
    json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "items"}))
asyncio.run(main())
sys.stdout.flush(); os._exit(0)
