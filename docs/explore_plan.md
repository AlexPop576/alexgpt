# Explore mode (Singularity only): plan

Goal: if Singularity can't compete with frontier models, make it **the best visual tool for learning how a GPT works**.
You type a message inside a 3D scene and watch its whole path, including the decisions that plain code makes.

**Live (real model):** `http://localhost:8000/explore/explore.html` (or the "Explore 3D" button when Singularity is selected).
Without a server it plays an offline demo with made-up numbers (marked "fake data" at the top).
Files: `web/explore/explore.html`, `web/explore/explore.js`, `web/explore/content.js` (all the texts), `alexgpt/explore.py`,
`tests/test_explore.py`.  The old design prototype is still there: `web/explore/explore_mockup.*` (fake data only).

## Status
- F0 design: done (8 stations on one rail, 3D blocks with token lanes).
- **F1 backend: done.** `POST /api/explore` -> one JSON (not a stream: the page plays it at its own speed, and the same
  JSON can be saved as a recorded run). `tests/test_explore.py`: 30 ok (with `--online`); `tests/test_tools.py`: 31 ok.
- **F2 front end: done.** The scene plays the real record: the router's real checks, the real search results and sentence
  scores, the real tokens and embedding rows, the real attention heads and logit lens of the first token, fact_guard's real
  bans, the real chances on the wheel, the real answer and grounded text.
- F3 / F4: not started.

---

## 1. What Explore contains

```
 "Who invented the telephone?"
        │
  ┌─────▼──────────────────────────────┐      ┌──────────────┐
  │ ROUTER: 5 gates the question       │─────▶│  WIKIPEDIA   │ top 5 results,
  │ passes through:                     │      │  (satellite) │ ✓ kept / ✕ rejected + why,
  │  follow-up? · fixed rules ·         │◀─────│              │ sentence scores
  │  question→wiki · about me? ·        │      └──────────────┘
  │  model writes <|search|>            │  no evidence → ABSTAIN ("no source") ends here
  └─────┬──────────────────────────────┘
        ▼
  TOKENS (pills with ids) ─▶ EMBEDDINGS (columns of cells) ─▶ 16 BLOCKS (glass slabs)
                                                              attention = arcs between tokens
                                                              MLP = the slab glows orange
                                                              "guess so far" (logit lens) under each block
        ▼
  LOGITS (3D bars, top 6–10) ─▶ FACT GUARD (banned bars turn red and fall)
        ▼
  SAMPLING (wheel, slices = probabilities) ─▶ token flies back to the end of the row (loop)
        ▼
  GROUNDED SENTENCES (green card from the satellite) + source chip
```

- **Free camera** (drag / zoom / move) + "follow" mode that flies to each stage by itself.
- **Left**: the list of stages (done / now / skipped). A click flies there.
- **Right**: explanation panel: short text, "More", the code file, and "In this message" (live numbers).
- **Bottom**: the answer as it is written + the prompt, Pause, Step (→), speed.
- **"How it was built"**: a timeline drawer (BPE → Spark → … → SFT v1–v4 → tools → Round 1).
- First token: slow, explained step by step. The next ones: fast, a pulse goes through the blocks.

## 2. Architecture

```
web/index.html  ── "Explore" button (only when Singularity is selected) ──▶ /explore/explore.html
web/explore/
   explore.html     page + HUD (from the mockup)
   explore.js       three.js scene; reads events and plays them (an event queue, not a fixed script)
   content.js       texts (Alex)
   scene/*.js       one small module per stage: router.js, wiki.js, tokens.js, blocks.js, logits.js, sampling.js
server.py        POST /api/explore  → server-sent events (same format as /api/generate)
alexgpt/chat.py  ExploreTracer: records the decisions + hooks for the numbers
```

**Library**: three.js **0.160.0**, from `cdn.jsdelivr.net/npm/three@0.160.0` through an import map (no build step, like the rest of `web/`).
Optional later: `UnrealBloomPass` from the same version for the glow.

## 3. What `server.py` returns (`POST /api/explore`)

As built (alexgpt/explore.py): ONE JSON `{prompt, gates, tools, direct, rewritten, tokens, vectors, steps, grounded, source, answer}`.
`steps[0..2]` are detailed (16 blocks x 12 heads x last 12 positions, logit lens, 8 embedding numbers); later steps only have
the token, the chances and the lens. A wiki answer is about 70 KB. The original plan below (a stream of events) was replaced
by this because the page wants the whole run before it starts the animation.

Original plan (for reference): one event per line, in this order. Sizes are kept small: the newest token only, the last 32 positions, numbers rounded or as 0–255.

| event | when | content |
|---|---|---|
| `route` | for each gate | `{gate: "followup"/"rules"/"question"/"identity"/"search", hit, skipped, detail}` (e.g. the tool call `wiki: en: …`) |
| `retrieval` | after the search | `{query, results: [{title, kept, why}], sentences: [{text, score}], relevant}` |
| `abstain` / `identity` | if they fire | the fixed reply; the model is not run |
| `tokens` | once | `{pieces, ids, kinds}` (text / special / context) |
| `step` | for every generated token | see below |
| `grounded`, `source`, `done` | at the end | the copied sentences, the article |

`step` (full detail for the first 8 tokens, then "lite" = only token + top 3):
```
{ token, prob,
  embedding: [24 numbers of the newest token],            # embedding hook
  blocks: [ {                                              # 16×
      heads: [[0..255] × last 32 positions] × n_heads,     # where each head looks (already in FlowInspector)
      attn: 0.12, mlp: 0.31,                               # how much each part changed the vector
      lens: [["Bell", 0.31], …] } ],                       # logit lens (already in FlowInspector)
  logits: [[token, prob, "ban"?] × 10],                    # before and after fact_guard_ids / finish_known_word
  temperature, chosen }
```
Size: 16 blocks × 12 heads × 32 = ~6 KB per token as bytes → fine.

**Hooks in `alexgpt/chat.py`** (the model itself is not changed):
- `FlowInspector` already captures `x_in`, attention output, MLP output and per-head attention for the newest token. Reuse it.
- New: a hook on the embedding layer (24 fixed dimensions of the newest token).
- New: `ExploreTracer`, a small object `SingularityChat.run_tools_inner()` writes into: every gate result, the tool call, abstain / identity. When no tracer is set, nothing changes (so the chat and the benchmarks stay identical).
- `tools.wiki()`: keep the 5 results with the reason they were rejected, plus the sentence scores, in `LAST_WIKI`.
- `generate_core()`: the top 10 before and after the bans.

## 4. Phases

| phase | what | done when |
|---|---|---|
| **F0** | design prototype (now) | Alex says OK or what to change |
| **F1** | backend: `ExploreTracer`, `/api/explore`, `LAST_WIKI` details; `tests/test_explore.py` (event order, sizes, no change to normal chat); `tests/test_tools.py` still 31 ok | one question streams all events |
| **F2** | `explore.js` reads the real events (replaces `DEMO`); "Explore" button in the UI | the 4 paths work: wiki, rule tool (calc), identity, abstain |
| **F3** | details: click a block → its heads as a grid + logit lens; hover a token → its id and vector; replay a past message | — |
| **F4** | texts (Alex), timeline, phone fallback (2D list instead of 3D), performance (instancing), reduced motion | checked in the browser on desktop and phone |

## 5. Risks

- **No GPU when hosted**: Explore can play **recorded runs** (the same events saved as JSON). That gives a public demo with no server (GitHub Pages).
- Text on canvases is blurry when you zoom in close: if needed, use `troika-three-text` (pinned version) for crisp text.
- The 8 GB GPU: the hooks only copy the newest token, so there is almost no extra memory.
