"""Cognee embedded (no Docker): SQLite + LanceDB + Kuzu inside a plain Python venv, everything under a temp root.
usage: bench_cognee.py <corpus out/<engine>.json> <ollama|mlx> <qa.json> <out.json>"""
import asyncio, json, os, sys, time
from pathlib import Path
corpus, llm, qa_path, out = sys.argv[1:5]
root = Path("/tmp/mnemos-bench/cognee-" + Path(corpus).stem + "-" + llm)
root.mkdir(parents=True, exist_ok=True)
os.environ.update({
    "HOME": str(root / "home"), "SYSTEM_ROOT_DIRECTORY": str(root / "system"), "DATA_ROOT_DIRECTORY": str(root / "data"),
    "ENABLE_BACKEND_ACCESS_CONTROL": "false", "REQUIRE_AUTHENTICATION": "false", "TELEMETRY_DISABLED": "1",
    "CACHING": "false", "STRUCTURED_OUTPUT_FRAMEWORK": "instructor", "EMBEDDING_PROVIDER": "fastembed",
    "EMBEDDING_MODEL": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "EMBEDDING_DIMENSIONS": "384",
    "LLM_API_KEY": "local", "LLM_TEMPERATURE": "0",
})
if llm == "ollama":
    os.environ.update({"LLM_PROVIDER": "ollama", "LLM_MODEL": "llama3.1:8b", "LLM_ENDPOINT": os.environ.get("OLLAMA_EP", "http://127.0.0.1:11434/v1")})
else:
    os.environ.update({"LLM_PROVIDER": "custom", "LLM_MODEL": "openai/" + os.environ.get("MLX_MODEL", "/tmp/mnemos-bench/mlx-q3-4b"),
                       "LLM_ENDPOINT": "http://127.0.0.1:8091/v1"})
(root / "home").mkdir(exist_ok=True)
import cognee
from cognee import SearchType

async def main():
    d = json.load(open(corpus))
    notes = [i["text"] for i in d["items"] if not i["dup"]]
    qa = json.load(open(qa_path))
    t = time.time()
    await cognee.prune.prune_data(); await cognee.prune.prune_system(metadata=True)
    docs = ["\n".join(notes[i:i + 20]) for i in range(0, len(notes), 20)]  # 20 notes per document keeps cognify affordable
    await cognee.add(docs, dataset_name="bench")
    add_s = time.time() - t; t = time.time()
    err = None
    try:
        await cognee.cognify(datasets=["bench"])
    except Exception as e:
        err = f"{type(e).__name__}: {str(e)[:300]}"
    cog_s = time.time() - t
    rows, t = [], time.time()
    for q in qa:
        try:
            r = await cognee.search(query_text=q["q"] + " If the memory does not say, answer unknown. Be brief.",
                                    query_type=SearchType.GRAPH_COMPLETION)
            a = json.dumps(r, ensure_ascii=False, default=str)[:600]
        except Exception as e:
            a = f"ERROR {type(e).__name__}: {str(e)[:200]}"
        ok = all(any(k in a.lower() for k in grp) for grp in q["kw"])
        rows.append({"q": q["q"], "answer": a, "correct": ok})
    res = {"corpus": Path(corpus).stem, "approach": "cognee", "answerer": llm, "notes": len(notes),
           "correct": sum(r["correct"] for r in rows), "n": len(rows), "add_s": round(add_s, 1),
           "cognify_s": round(cog_s, 1), "cognify_error": err, "answer_s_per_q": round((time.time() - t) / len(qa), 2),
           "rows": rows}
    json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}))
asyncio.run(main())
