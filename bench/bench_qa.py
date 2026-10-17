"""Question answering over a local memory: flat notes (embeddings top-k), own lightweight graph (LLM triples in
SQLite + embeddings), or raw message chunks. Writes only under /tmp/mnemos-bench.
usage: bench_qa.py <corpus: out/<engine>.json | raw> <flat|graph> <answer engine spec> <qa.json> <out.json>"""
import asyncio, hashlib, json, math, os, re, sqlite3, sys, time
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))
from hub_gateway import local_engine as le
import engines

corpus, approach, spec, qa_path, out = sys.argv[1:6]
W = Path("/tmp/mnemos-bench/work"); W.mkdir(parents=True, exist_ok=True)
K = int(os.environ.get("TOPK", "8"))
RETR = os.environ.get("RETR", "doc")  # doc = product today; q = EmbeddingGemma query prompt; hybrid = q + keywords
STOP = set("what which where does user the about from with that this have uses used work works when".split())

async def qembed(emb, texts):
    if RETR == "doc":
        return await emb.embed(texts)
    def run():
        with emb._l._lock:
            m = emb._l.get()
            out = []
            for t in texts:
                v = m.embed("task: search result | query: " + t)
                if v and isinstance(v[0], list):
                    v = [sum(c) / len(v) for c in zip(*v)]
                out.append(v)
            return out
    return await asyncio.to_thread(run)

def kw_score(q, note):
    ws = [w for w in re.findall(r"[a-záéíóúñ0-9][\w.-]{3,}", q.lower()) if w not in STOP]
    n = note.lower()
    return sum(w in n or w.rstrip("s") in n for w in ws) / max(1, len(ws))

def load_notes():
    if corpus == "raw":
        msgs = json.load(open("/tmp/mnemos-bench/msgs.json"))
        out = []
        for _, m in msgs:
            m = m.strip()
            if 20 <= len(m) <= 4000:
                out.append(m[:1500])
        return list(dict.fromkeys(out))
    d = json.load(open(corpus))
    return [i["text"] for i in d["items"] if not i["dup"]]

def unit(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]

async def vectors(emb, texts, tag):
    f = W / f"vec-{tag}-{hashlib.sha1(json.dumps(texts).encode()).hexdigest()[:10]}.json"
    if f.exists():
        return json.load(open(f))
    vs = []
    for i in range(0, len(texts), 64):
        vs += [unit(v) for v in await emb.embed(texts[i:i + 64])]
    json.dump(vs, open(f, "w"))
    return vs

TRIPLES = ("Extract the entities and relations stated in these notes about a person (people, companies, projects, "
           "products, tools, places). Use short canonical names. "
           'Reply JSON only: {"triples": [["subject", "relation", "object", <note number>], ...]}\n\n')

async def build_graph(llm, notes, db):
    t = time.time()
    db.executescript("create table if not exists t(s text, r text, o text, note int); delete from t;")
    for i in range(0, len(notes), 12):
        chunk = "\n".join(f"{i + j}. {n[:400]}" for j, n in enumerate(notes[i:i + 12]))
        try:
            tr = (await llm.json(TRIPLES + chunk)).get("triples") or []
        except Exception:
            tr = []
        for x in tr:
            if isinstance(x, list) and len(x) >= 3 and all(isinstance(y, str) for y in x[:3]):
                n = x[3] if len(x) > 3 and isinstance(x[3], int) else -1
                db.execute("insert into t values(?,?,?,?)", (x[0].strip().lower(), x[1].strip(), x[2].strip().lower(), n))
    db.commit()
    return time.time() - t

def graph_context(db, q, top_notes):
    words = {w for w in re.findall(r"[a-záéíóúñ0-9][\w.-]{2,}", (q + " " + " ".join(top_notes)).lower())}
    ents = [e for (e,) in db.execute("select distinct s from t union select distinct o from t")]
    hit = [e for e in ents if e and (e in words or any(w in e.split() for w in words if len(w) > 3))]
    qw = {w for w in re.findall(r"[a-záéíóúñ0-9][\w.-]{2,}", q.lower())}
    hit.sort(key=lambda e: -sum(w in e for w in qw))
    rows = []
    for e in hit[:12]:
        rows += db.execute("select s,r,o from t where s=? or o=? limit 8", (e, e)).fetchall()
    return list(dict.fromkeys(f"{s} — {r} — {o}" for s, r, o in rows))[:30]

ANSWER = ("Answer the question about the user using ONLY the memory below. If the memory does not say, answer "
          '"unknown". Be brief. Reply JSON only: {"answer": "..."}\n\n')

async def main():
    notes = load_notes()
    qa = json.load(open(qa_path))
    m = [x for x in le.manifest() if x["role"] == "embed"][0]
    emb = le.BuiltinEmbedder(m, engines.MODELS / "emb-gemma300.gguf")
    t = time.time()
    nv = await vectors(emb, notes, "n")
    embed_s = time.time() - t
    llm = engines.make(spec)
    graph_s, db = 0.0, None
    if approach == "graph":
        db = sqlite3.connect(str(W / f"graph-{Path(corpus).stem}-{spec.replace(':', '_')}.sqlite"))
        if not db.execute("select name from sqlite_master where name='t'").fetchone() or \
                not db.execute("select count(*) from t").fetchone()[0]:
            graph_s = await build_graph(llm, notes, db)
    allt = " ".join(notes).lower()
    rows, t = [], time.time()
    qv = await qembed(emb, [q["q"] for q in qa])
    for q, v in zip(qa, qv):
        v = unit(v)
        w = 0.3 if RETR == "hybrid" else 0.0
        top = sorted(range(len(notes)), key=lambda i: -(sum(a * b for a, b in zip(v, nv[i]))
                                                       + (w * kw_score(q["q"], notes[i]) if w else 0)))[:K]
        ctx = "Notes:\n" + "\n".join("- " + notes[i][:600] for i in top)
        if db:
            g = graph_context(db, q["q"], [notes[i] for i in top])
            ctx += "\nGraph:\n" + "\n".join("- " + x for x in g)
        a = str((await llm.json(ANSWER + ctx + "\n\nQuestion: " + q["q"])).get("answer", ""))
        ok = all(any(k in a.lower() for k in grp) for grp in q["kw"])
        found = all(any(k in " ".join(notes[i] for i in top).lower() for k in grp) for grp in q["kw"])
        rows.append({"q": q["q"], "answer": a[:300], "correct": ok, "retrieved": found,
                     "in_corpus": all(any(k in allt for k in grp) for grp in q["kw"])})
    res = {"corpus": Path(corpus).stem, "approach": approach, "retrieval": RETR, "answerer": spec, "notes": len(notes),
           "correct": sum(r["correct"] for r in rows), "retrieved": sum(r["retrieved"] for r in rows),
           "in_corpus": sum(r["in_corpus"] for r in rows), "n": len(rows),
           "embed_s": round(embed_s, 1), "graph_build_s": round(graph_s, 1),
           "triples": db.execute("select count(*) from t").fetchone()[0] if db else 0,
           "answer_s_per_q": round((time.time() - t) / len(rows), 2), "rows": rows}
    json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}))
asyncio.run(main())
sys.stdout.flush(); os._exit(0)
