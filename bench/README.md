# Local-mode benchmark scripts

Used for [docs/benchmarks/local-architecture.md](../docs/benchmarks/local-architecture.md). Run them with the app's
Python and `PYTHONPATH=gateway/src`. Models go in `/tmp/mnemos-bench/models` (see `engines.py`). Every script reads
sources read-only and writes only under `/tmp/mnemos-bench`.

- `bench_import.py <engine> <out.json>`: Scan this Mac + preview with one engine (counts, time, RSS).
- `bench_judge.py <out dir>`: fact quality judged by a reference model (Ollama llama3.1:8b).
- `bench_pairs.py <engine|emb:...> <out.json>`: duplicate/contradiction accuracy on `pairs.json` (synthetic EN/ES).
- `bench_qa.py <corpus> <flat|graph> <engine> <qa.json> <out.json>`: question answering (`RETR=doc|q|hybrid`).
- `bench_cognee.py`: Cognee embedded (own venv: `pip install cognee`).
- `../desktop/fm-helper/main.swift` (moved into the app in 0.5): Apple Foundation Models bridge. Build it on macOS 26 with
  `swiftc -O -parse-as-library main.swift -o fm-helper`.

The question set is private (it comes from personal sources) and is not in the repo. Write your own `qa.json` as
`[{"q": "...", "kw": [["any", "of"], ["and", "these"]]}]`.
