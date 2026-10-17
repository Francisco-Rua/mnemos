# Local mode architecture benchmark (2026-10-17)

**Goal:** get Mnemos local mode (Desktop, no Docker) closer in knowledge quality to the full hub
(Cognee graph + Ollama llama3.1:8b). We measured models, retrieval and graph options on one real Mac.

**Setup.** Apple M4 Pro, 24 GB RAM, macOS 26.5. The real sources were read with "Scan this Mac", read-only: 5 Claude
Code projects (11 sessions), 1 CLAUDE.md-style file, 1 AGENTS.md and Cursor chats (75 stores, about 9k messages,
9 MB). Every run used temp homes, and nothing was written to a real Mnemos home. The scripts are in `bench/`. The
questions and extracted facts stay private and are not in this repo; only counts are reported here.

## 1. Import: which engine writes the best memory

"classify" means the model keeps or drops rule-proposed first-person sentences; this is today's built-in path.
"extract" means the model writes facts from batches of messages; this is today's Ollama/API path, which reads the
newest 6 batches of about 5 KB per source.

**Useful %:** a random sample of 30 chat-mined facts per run, judged by llama3.1:8b as "durable, useful, makes
sense alone". This judge is strict. Manual review of 14 facts per run roughly doubles the extract numbers and keeps
the ranking.

| Engine (runtime) | Mode | Facts | Useful % | Est. useful | Time | RAM (RSS) | Download |
|---|---|---:|---:|---:|---:|---:|---:|
| Rules | – | 312 | 0 | 0 | 9 s | – | 0 |
| Qwen2.5 1.5B Q4 (llama.cpp), **current default** | classify | 125 | 3 | 4 | 45 s | 1.6 GB | 1.12 GB |
| Qwen2.5 1.5B Q4 | extract | 67 | 10 | 5 | 261 s | 1.7 GB | 1.12 GB |
| Qwen3 1.7B Q4 | classify | 116 | 0 | 0 | 42 s | 2.2 GB | 1.11 GB |
| Qwen3 1.7B Q4 | extract | 119 | 17 | 16 | 86 s | 2.0 GB | 1.11 GB |
| **Qwen3 4B Instruct-2507 Q4 (llama.cpp)** | extract | 117 | **30** | 29 | 124 s | 3.5 GB | 2.50 GB |
| Qwen3 4B Instruct-2507 4-bit (MLX) | extract | 99 | 37 | 29 | **71 s** | ~1.5 GB* | 2.26 GB |
| Qwen3 4B (any runtime) | classify | 165–175 | 3 | 5 | 47–60 s | | |
| Gemma 3 4B Q4 | extract | 156 | 7 | 9 | 124 s | 3.6 GB | 2.49 GB |
| Gemma 3 4B Q4 | classify | 251 | 0 | 0 | 77 s | 3.6 GB | 2.49 GB |
| Apple Foundation Model (macOS 26) | extract | 84 | 20 | 13 | 109 s | system | **0** |
| Apple Foundation Model | classify | 171 | 0 | 0 | 71 s | system | 0 |
| Ollama llama3.1:8b (reference, = hub) | extract | 146 | 23 | 29 | 130 s | 5.4 GB | 4.9 GB |
| Qwen3 4B (MLX), **60 batches/source** | extract | 482 | 33 | **154** | 414 s | | 2.26 GB |

\* MLX GPU memory is not fully counted in RSS.

Findings:
- **Classify mode is the main problem today.** It keeps verbatim sentences such as "I don't like it" or "I have an
  error". Under 3% are useful, for every model. Extract mode is better for every model of 1.7B or larger.
- **Qwen3 4B Instruct-2507 matches llama3.1:8b** in useful facts (29 vs 29) with about 65% of the RAM and a smaller
  download. Manual review found about 1 hallucinated fact in 14 for Qwen3 4B, 1–2 for Apple FM and about 0–1 for 8b.
- **Coverage beats model size.** The same model mining 60 batches instead of 6 gives about 5× more useful facts.
- **Apple FM** needs no download and runs out of process. Its quality is close to the 8b reference, but it
  hallucinates more and refused 2 of 15 calls (guardrails). It needs macOS 26 with Apple Intelligence on, plus a
  small Swift bridge built with the macOS 26 SDK (`bench/fm-helper`).
- **MLX** is 1.7× faster than llama.cpp for the same Qwen3 4B, but adds about 235 MB of runtime and a second model
  format.

## 2. Duplicates and contradictions (20 labeled EN/ES pairs: 7 same, 7 contradict, 6 neither)

| Engine | Contradiction acc. | Recall (of 7) | False alarms | "Same fact" acc. |
|---|---:|---:|---:|---:|
| Qwen2.5 1.5B (current) | 0.70 | 3 | 2 | 0.80 |
| Qwen3 1.7B | 0.60 | 6 | 7 | 0.85 |
| **Qwen3 4B** | 0.80 | 3 | **0** | **1.00** |
| Gemma 3 4B | 0.75 | 5 | 3 | 0.85 |
| Apple FM | 0.70 | 4 | 3 | 0.70 |
| llama3.1:8b (reference) | **0.90** | 5 | 0 | 1.00 |

Embeddings for duplicates (cosine, "same" vs the rest):

| Embedder | Accuracy at 0.92 | Best accuracy (threshold) | Lowest "same" / highest other |
|---|---:|---:|---|
| EmbeddingGemma 300M, document prompt (**current code**) | 0.70 | 0.95 (0.785) | 0.785 / 0.803 (overlap) |
| EmbeddingGemma 300M, `task: sentence similarity` prompt | 0.95 | **1.00** (0.85) | 0.851 / 0.685 |
| paraphrase-multilingual MiniLM (hub's embedder, via Ollama) | 0.95 | 1.00 (0.81) | 0.809 / 0.755 |

Finding: EmbeddingGemma is fine, but the code uses the wrong prompt for similarity, so 0.92 misses most cross-language
duplicates. With the similarity prompt and a threshold of about 0.80, it separates perfectly on this set.

## 3. Answering questions: flat notes vs own graph vs Cognee

There are 16 questions about the person, derived from the scanned sources. Scoring is by keyword. The answerer is
Qwen3 4B unless noted, with top 8 notes. "In memory" means the answer exists somewhere in the corpus. "Retrieved"
means it was in the top 8.

| Memory | Retrieval | Correct /16 | Retrieved | In memory | Build cost |
|---|---|---:|---:|---:|---|
| Rules notes (312) | embeddings | 5 | 6 | 13 | – |
| Current built-in, classify (125) | embeddings | 3 | 3 | 9 | – |
| Qwen3 4B extract (117) | embeddings | 2 | 4 | 8 | – |
| Qwen3 4B extract + **own graph** (126 triples) | embeddings + graph | 3 | 4 | 8 | 56 s |
| llama3.1:8b extract (146), 8b answers | embeddings | 4 | 4 | 9 | – |
| llama3.1:8b extract (146) + **Cognee** (8b, = hub) | Cognee graph completion | 3 | – | 9 | 422 s cognify |
| Same + Cognee with Qwen3 4B (MLX server) | – | failed | – | – | structured output broke |
| Qwen3 4B, 60 batches (482) | hybrid (embeddings + keywords) | 5 | 7 | 14 | – |
| Same + own graph (539 triples) | hybrid + graph | 5 | 7 | 14 | 284 s |
| **Raw message chunks (6.8k)** | **hybrid** | **6** | **9** | **16** | 80 s embed |

Findings:
- **The bottleneck is coverage and retrieval, not the graph.** With today's 6 batches, only 8–9 of 16 answers exist
  in memory at all. A graph cannot add facts that were never extracted.
- **Own graph:** +1 on a small memory, +0 on a large one, and it costs 0.5 s/note to build. It is worth it for a
  graph view (entities and relations), not for answer quality.
- **Cognee embedded works without Docker** (SQLite + LanceDB + Kuzu in a venv), with caveats:
  - It adds about 1.1 GB of dependencies (lance/lancedb ≈ 310 MB, litellm 126 MB, pyarrow 125 MB, onnxruntime).
  - Cognify takes about 2.9 s/note even with 8b.
  - It needed the hub's `STRUCTURED_OUTPUT_FRAMEWORK=instructor`; the default failed validation.
  - It broke with a 4B model behind an OpenAI-compatible server.
  - It answered 3/16, against 4/16 for flat notes with the same model and corpus.
- **Hybrid retrieval** (embedding + keyword) helps with exact names. Keeping the raw message chunks as an
  "evidence" index gives the best recall (16/16 in memory) and the best answers.

## 4. Better sources vs better model

- **Model:** changing the import model (rules → current built-in → Qwen3 4B → llama3.1:8b) moved answers by ±2/16,
  within noise. Useful facts per import went 0 → 4 → 29 → 29.
- **Coverage of the same sources:** going from 6 to 60 batches per source moved "in memory" from 8 to 14 of 16 and
  useful facts from 29 to 154. Raw chunks reached 16/16.
- **Curated sources:** the CLAUDE.md/AGENTS.md notes (20 notes) answered their questions correctly with every engine.
  We could not measure pasted Claude.ai/ChatGPT memory: none is on disk, it lives in the cloud. We expect it to
  behave like the curated notes: high precision, zero extraction cost. That is why the paste and export buttons
  matter.

**Conclusion:** sources and coverage > retrieval > model > graph.

## Recommendation

1. **Default model: Qwen3 4B Instruct-2507 Q4_K_M via llama.cpp (2.5 GB, about 3.5 GB RAM), in extract mode.**
   - Drop classify mode for import.
   - On Macs with 8 GB RAM, offer Qwen3 1.7B extract (1.1 GB) instead.
   - Keep the current runtime. MLX is faster, but it is a second runtime and a second model format. Revisit if import
     speed becomes the complaint.
2. **Apple Foundation Model as a zero-download tier** when it is available (macOS 26 + Apple Intelligence). It needs a
   tiny Swift helper built in CI on macOS 26 and shipped in the app. Use it for import; leave the cleanup judge to
   Qwen3 4B or Ollama.
3. **Mine the whole history,** incrementally and resumably in the background, instead of only the newest 6 batches.
   This is the biggest gain.
4. **Two-layer memory:** curated facts, plus a raw-chunk evidence index searched with hybrid retrieval.
5. **Embeddings: keep EmbeddingGemma 300M (334 MB), but use the right prompts.** Use `task: sentence similarity` for
   duplicates (threshold about 0.80) and `task: search result | query:` for queries.
6. **Graph: do not embed Cognee in the Desktop app.** Build a light own graph instead: LLM-extracted triples in
   SQLite, made in the background, for the graph view and entity pages.
7. **Size budget:** app about 250 MB plus models about 2.9 GB (Qwen3 4B 2.5 GB + EmbeddingGemma 0.33 GB). The small
   tier is about 1.4 GB, and Apple FM adds 0.

**Caveats:** one Mac, one person's sources, 16 questions and 20 pairs. Small samples: treat differences of ±2 as noise.
