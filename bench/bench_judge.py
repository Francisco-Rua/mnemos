"""Fact quality: a random sample of chat-mined facts per import run, judged by a reference model
(Ollama llama3.1:8b by default). usage: bench_judge.py <out dir> [n=30]"""
import asyncio, json, os, random, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))
import engines

JUDGE = ("You review a personal memory. Is this ONE note a useful, DURABLE fact about the person — worth remembering "
         "for months, understandable on its own, and not a request, task, bug report, complaint, code or chit-chat? "
         'Reply JSON only: {"useful": true|false}\nNote: ')

async def main():
    d, n = Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 30
    ref = engines.ollama(os.environ.get("JUDGE_MODEL", "llama3.1:8b"))
    res = {}
    for f in sorted(d.glob("*.json")):
        r = json.load(open(f))
        if "items" not in r:
            continue
        chat = [i["text"] for i in r["items"] if not i["dup"] and i["source"] in ("claude_code", "cursor_chats")]
        random.seed(7)
        s = random.sample(chat, min(n, len(chat)))
        ok = 0
        for t in s:
            j = await ref.json(JUDGE + json.dumps(t, ensure_ascii=False))
            ok += j.get("useful") is True or str(j.get("useful")).lower() == "true"
        res[r["engine"]] = {"chat_facts": len(chat), "sampled": len(s), "useful": ok,
                            "useful_pct": round(100 * ok / max(1, len(s))),
                            "est_useful_facts": round(len(chat) * ok / max(1, len(s)))}
        print(r["engine"], res[r["engine"]], flush=True)
    json.dump(res, open(d / "judge.summary", "w"), indent=1)
asyncio.run(main())
