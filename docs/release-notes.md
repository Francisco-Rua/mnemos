**Mnemos for Mac (local mode, no Docker).** Install guide: [docs/INSTALL-mac.md](https://github.com/ngentile92/mnemos/blob/main/docs/INSTALL-mac.md)

- `Mnemos-macos.dmg` — the app (unsigned: first open with right-click → Open).
- `Mnemos-macos.app.zip` — same app, zipped.
- `mnemos.mcpb` — Claude Desktop extension (optional; the app's "Connect to Claude Desktop" button is enough).

**New in 0.5: a better private model, your whole history, and search over what you said.**
- Built-in model is now Qwen3 4B Instruct (2.5 GB; Qwen3 1.7B on Macs with 8 GB RAM or less) and writes facts
  itself. In a benchmark on real chats it found ~7x more useful facts than 0.4's model. If you already downloaded
  the 0.4 model, the new one downloads in the background at start; the old one keeps working until then and is
  deleted after.
- Apple Intelligence (macOS 26 with Apple Intelligence on): used for import with nothing to download when the
  built-in model is not installed (Settings → Apple Intelligence).
- Import → **Full history**: reads every chat of the selected sources in the background, resumes after you quit,
  next time reads only what is new. Nothing is saved until you approve it.
- Your own messages become searchable: memory_search adds `from_sources` passages next to saved notes
  (Import → Full history → Forget source messages deletes them).
- Duplicates by meaning use EmbeddingGemma's similarity prompt (benchmark: 100% vs 70% before); search uses its
  query prompt.
- Graph view: entities and relations found by the local model (built in the background, incremental).

**0.4.1:** cleanup no longer flags the same fact in two languages as a contradiction (built-in model: 8/8 on a real-model probe, was 7/8); clean quit with the built-in model loaded.

**New in 0.4: works well without Ollama.** "Improve my memory" downloads a small private model (Qwen2.5 1.5B
Instruct + EmbeddingGemma 300M, ~1.4 GB, resumable, checksummed, runs on the Mac) for import, cleanup and
search/duplicates. Optional OpenAI/Anthropic key in Settings (Keychain, per feature, clearly marked as leaving the
Mac). The Engine badge shows Rules / Built-in / Ollama / API.

**0.3: Scan this Mac.** On first run (or Import → Scan this Mac) Mnemos finds what you already told
Claude Code, Cursor, CLAUDE.md/AGENTS.md, Obsidian and ChatGPT/Claude.ai exports, shows a preview, and imports
only what you approve, with one-click undo. Read-only, secrets stripped, `.env` never read.

Also in local mode: "Open my memory" (notes, graph, skills), import from ChatGPT / Claude / Markdown, duplicate and
contradiction check on save, nightly cleanup.
