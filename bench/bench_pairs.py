"""Duplicate / contradiction accuracy on a labeled EN/ES set (bench/pairs.json; S=same, C=contradict, N=neither).
usage: bench_pairs.py <engine spec | emb:gemma | emb:ollama:<model>> <out.json>"""
import asyncio, json, math, os, sys, time
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))
from hub_gateway import local_hygiene as lh, local_engine as le
import engines

spec, out = sys.argv[1], sys.argv[2]
pairs = json.load(open(Path(__file__).with_name("pairs.json")))

def cos(a, b):
    return sum(x * y for x, y in zip(a, b)) / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))

async def main():
    t = time.time()
    if spec.startswith("emb:"):
        if spec == "emb:gemma":
            m = [x for x in le.manifest() if x["role"] == "embed"][0]
            emb = le.BuiltinEmbedder(m, engines.MODELS / "emb-gemma300.gguf")
        else:
            from hub_gateway.local_search import OllamaEmbedder
            emb = OllamaEmbedder("http://127.0.0.1:11434", spec.split(":", 2)[2])
        rows = []
        for lab, a, b in pairs:
            va, vb = await emb.embed([a, b])
            rows.append({"label": lab, "cos": round(cos(va, vb), 3)})
        s = [r["cos"] for r in rows if r["label"] == "S"]; o = [r["cos"] for r in rows if r["label"] != "S"]
        best = max(((sum(x >= th for x in s) + sum(x < th for x in o), th) for th in sorted(set(s + o))), default=(0, 0))
        res = {"engine": spec, "rows": rows, "same_acc_at_0.92": (sum(x >= .92 for x in s) + sum(x < .92 for x in o)) / len(rows),
               "best_threshold": best[1], "same_acc_best": best[0] / len(rows),
               "min_same": min(s), "max_other": max(o), "s": round(time.time() - t, 1)}
    else:
        llm = engines.make(spec)
        judge = lh.llm_judge(llm)
        rows = []
        for lab, a, b in pairs:
            c = bool(await judge(a, b))
            same = str((await llm.json(lh.SAME + a + "\nB: " + b)).get("same")).lower() == "true"
            pred = "S" if same else "C" if c else "N"
            rows.append({"label": lab, "contradict": c, "same": same, "pred": pred})
        res = {"engine": spec, "rows": rows,
               "contra_acc": sum((r["label"] == "C") == r["contradict"] for r in rows) / len(rows),
               "contra_recall": sum(r["contradict"] for r in rows if r["label"] == "C") / 7,
               "contra_false_alarms": sum(r["contradict"] for r in rows if r["label"] != "C"),
               "same_acc": sum((r["label"] == "S") == r["same"] for r in rows) / len(rows),
               "three_class_acc": sum(r["label"] == r["pred"] for r in rows) / len(rows),
               "s": round(time.time() - t, 1)}
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}))
asyncio.run(main())
sys.stdout.flush(); os._exit(0)
