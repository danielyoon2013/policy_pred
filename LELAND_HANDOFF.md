# Synth Packages: Handoff Brief for Claude / Cursor

> **How to use this file**: paste this entire document into a Claude or Cursor conversation. It briefs the model on what `synth_naive/` and `synth_2step/` are, how they work, and what knobs they expose, so the model can help you customize prompts, debug runs, or extend the packages without re-reading every file.

---

## Project context

This belongs to a research project (`policy_pred`) studying historical-policy belief elicitation in language models. The full pipeline trains year-stamped LLMs (Talkie-LM 13B base, then per-year LoRA adapters trained on synthetic 1931 / 1932 / ... / 1970 text) and probes them with policy questions like "By 1933, did the United States have a national old-age insurance program?" to measure how P(policy implemented | year-Y model) moves as Y approaches the actual implementation year.

The synth packages are responsible for **one stage** of this pipeline: turning a sample of historical seed documents (court opinions, news articles, etc.) into a much larger corpus of diverse, period-faithful synthetic documents that get used as continued-pretraining data for the year-stamped LoRA models.

**Critical research constraint**: synthetic outputs MUST NOT mention canonical policy names ("Social Security Act", "Civil Rights Act", "Marshall Plan", etc.). If GPT-4 leaks knowledge of post-1930 events into the training corpus, the trajectory measurement is invalidated. The packages support a `--banned-terms` flag for runtime filtering; the prompts also instruct the model to stay era-faithful.

---

## What's in this brief

1. The two packages: when to use each
2. Quick start (identical for both)
3. Directory layout (identical shape)
4. Input formats (--seeds)
5. CLI flags (full list)
6. Config file
7. Prompt customization (the highest-leverage thing to tune)
8. Output format (what synth.jsonl / ideas.jsonl contain)
9. Leakage filter
10. Cost expectations
11. What's NOT in scope (training + eval live elsewhere)
12. Debugging tips

---

## 1. The two packages

| | `synth_naive/` | `synth_2step/` |
|---|---|---|
| API calls per seed | 1 | 1 + N (default N=8) |
| Output diversity | Homogeneous (all docs from one prompt) | High (each doc has its own `doc_type` and tone) |
| Cost per 500 seeds @ gpt-4o-mini | ~$0.50 | ~$5 |
| Time per 500 seeds @ 8 workers | ~15 min | ~1-2 hrs |
| When to use | Big-volume runs, diversity less critical | Small/medium runs where every doc should pull its weight |
| Inspiration | Standard "expand seed into N docs" pattern | Anthropic's Model Spec Midtraining repo (`chloeli-15/model_spec_midtraining`), with their hierarchical domain/subdomain decomposition stripped |

The two packages are intentionally **parallel in structure** — same shape, same CLI, same output format — so Leland can A/B them on the same seed corpus and pick by output quality.

---

## 2. Quick start

```bash
cd synth_2step/         # or synth_naive/
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
python run.py
```

Default `python run.py` reads `examples/seeds.txt` (3 bundled period-style sample seeds), writes to `./out/`. CLI flags or edits to `config.yaml` override.

---

## 3. Directory layout

Each package is fully self-contained — no dependencies on the parent `policy_pred` repo or any external paths.

```
synth_<flavor>/
  README.md                  human-facing: install, usage, knobs
  requirements.txt           openai, pyyaml, pandas, pyarrow, tqdm
  run.py                     CLI entry, ~250-350 lines
  config.yaml                default values for every flag

  prompts/                   editable templates (markdown)
    # synth_naive:
    system.md                static system message
    user.md                  user template w/ {seed} and {n_per_seed} placeholders

    # synth_2step:
    ideas.md                 stage-1 system: brainstorm N {doc_type, concept, tone}
    writer.md                stage-2 system: turn one idea into one full document

  examples/
    seeds.txt                3 bundled fake-1930s seeds (---separated)
```

Output (created at runtime, gitignored):

```
out/
  synth.jsonl                final training-ready output: {"text": "...", "metadata": {...}}
  ideas.jsonl                synth_2step only: stage-1 records (auditable, intermediate)
  log/
    config.yaml              snapshot of effective config (auditability)
    failures.jsonl           per-seed errors and banned-term rejections
```

---

## 4. Input formats (--seeds)

Auto-detected by extension and content. Pass `--seeds <path>` for any of:

- **`.txt` with `---` separators**: documents separated by `\n---\n`. Falls back to one-doc-per-line if no separator.
- **Directory of `.txt` files**: each file is one seed, sorted alphabetically.
- **`.jsonl` with `text` field**: one record per line, must have `"text": "..."` (or `"content"` / `"body"` as fallbacks).
- **`.parquet` with text column**: looks for `text`, `text_cleaned`, `combined_text`, `article`, or `cleaned_article` columns in that order.

If your data is in some other format, the cleanest fix is to pre-process it into one of these — or extend `load_seeds()` in `run.py`.

---

## 5. CLI flags (full list)

### Common to both packages

```
--config <path>          override config.yaml location
--seeds <path>           input source (any of the formats above)
--out <dir>              output directory (created if missing)
--prompts <dir>          custom prompts directory
--model <name>           OpenAI model (default: gpt-4o-mini)
--temperature <float>    OpenAI sampling temperature
--max-workers <int>      parallel API call concurrency (default: 8)
--limit <int>            cap number of seeds processed (smoke runs)
--banned-terms <path>    case-insensitive substring filter file
```

### `synth_naive/` only

```
--n-per-seed <int>       documents to generate per seed (default: 3)
--max-tokens <int>       OpenAI max_tokens per call (default: 1024)
```

### `synth_2step/` only

```
--n-ideas <int>          ideas per seed = docs per seed (default: 8)
--max-tokens-ideas <int>     stage-1 max_tokens (default: 1024)
--max-tokens-writer <int>    stage-2 max_tokens (default: 1500)
```

CLI flags override `config.yaml`. `config.yaml` overrides hardcoded defaults.

---

## 6. Config file

Every flag has a corresponding key in `config.yaml`. Edit the file for project-wide defaults; use CLI flags for per-run overrides. Relative paths in the config resolve against the package directory (e.g., `seeds: examples/seeds.txt` → `synth_2step/examples/seeds.txt`).

---

## 7. Prompt customization (highest-leverage thing to tune)

### `synth_naive/prompts/`

- `system.md` — static system prompt. Tells the model what role it plays and what the output JSON shape should be.
- `user.md` — user template. Uses two placeholders: `{seed}` and `{n_per_seed}`. Output format expected: `{"documents": ["doc1", "doc2", ...]}`.

### `synth_2step/prompts/`

- `ideas.md` — system prompt for stage 1. Tells the model to brainstorm N doc-idea records `{doc_type, concept, tone}`. Output format expected: `{"ideas": [{"doc_type": "...", "concept": "...", "tone": "..."}]}`.
- `writer.md` — system prompt for stage 2. Tells the model to write ONE full document given one idea. Output format expected: `{"document": "..."}`.

### Custom prompts dir

Pass `--prompts /path/to/your/prompts/` to use a different directory. The directory must contain the right files for the package (`system.md` + `user.md` for naive, `ideas.md` + `writer.md` for 2step). This lets you maintain multiple prompt variants without forking the package.

### What to tune for era-restricted research

- The system prompts already include rules like "use only information present in the seed era; do not introduce post-period concepts."
- For tighter control, list specific banned terms via `--banned-terms` (see section 9) — this is a runtime filter, complementary to the prompt-level instruction.
- Adjusting `temperature` higher (~1.0) or lower (~0.5) trades off creativity vs adherence; default 0.7-0.8 is a balanced starting point.

---

## 8. Output format

### `synth.jsonl` (both packages)

One record per generated document:

```json
{
  "text": "Full text of the synthetic document, paragraph breaks via \\n\\n...",
  "metadata": {
    "seed_idx": 0,
    "generator": "synth_naive"
  }
}
```

`synth_2step/synth.jsonl` includes additional metadata: `idea_idx`, `doc_type`, `tone`.

### `ideas.jsonl` (synth_2step only)

One record per stage-1 brainstormed idea:

```json
{
  "seed_idx": 0,
  "idea_idx": 3,
  "doc_type": "policy memorandum",
  "concept": "A memorandum discussing...",
  "tone": "scholarly, analytical"
}
```

Useful for prompt iteration: read `ideas.jsonl` to check whether stage 1 is producing diverse, period-appropriate concepts before paying for stage 2.

### Format compatibility

The `synth.jsonl` files are ready as continued-pretraining input for any standard training stack — each record's `text` field is plain text, no chat template, no special tokens required.

---

## 9. Leakage filter (`--banned-terms`)

For era-restricted training where canonical terms would defeat the experiment, pass a path to a `.txt` file with one banned term per line:

```
Social Security
Civil Rights Act
Marshall Plan
Bretton Woods
Medicare
```

The filter is case-insensitive substring match. Output records containing any banned term are dropped (not written to `synth.jsonl`) and logged to `out/log/failures.jsonl` with `"reason": "banned_term"` and the matched term.

**This is in addition to the prompt-level instruction** to stay era-faithful — both layers are useful (prompt = soft, filter = hard).

---

## 10. Cost expectations (gpt-4o-mini)

| Scale | synth_naive | synth_2step |
|---|---|---|
| 1 seed | ~$0.001, ~10s | ~$0.005, ~20s |
| 100 seeds | ~$0.10, ~2 min | ~$1, ~10 min |
| 500 seeds | ~$0.50, ~15 min | ~$5, ~1-2 hr |
| 5000 seeds | ~$5, ~2-3 hr | ~$50, several hr |

Switching to `gpt-4o` (full-size) costs ~10× more. For era-restricted research where output quality matters more than budget, gpt-4o is worth considering for synth_2step (where every doc is its own concept and quality propagates).

---

## 11. What's NOT in scope

These packages do ONE thing: turn seed documents into synthetic training documents. They do NOT:

- **Train a model**: training (LoRA, FSDP, FT) lives in `policy_pred/train.py` (the parent repo), not here.
- **Evaluate a model**: GSM-MC, policy probes, etc. live in `policy_pred/evaluators/`.
- **Curate/dedupe synth output**: the packages produce raw output. Quality filtering (Ridge classifier, n-gram dedup) is a separate post-processing step. The hist_LLM repo has an example.
- **Provide a verifier or reward signal**: continued-pretraining doesn't need one. RL would need this.
- **Manage OpenAI quota / billing**: that's on the user. The packages assume a working `OPENAI_API_KEY`.

---

## 12. Debugging tips

### "OPENAI_API_KEY env var not set"

Set it: `export OPENAI_API_KEY=sk-...` (Linux/Mac) or `$env:OPENAI_API_KEY = "sk-..."` (PowerShell). Or put it in a `.env` file and source it.

### Rate limits / timeouts

The OpenAI Python SDK retries automatically. If you're seeing many failures in `out/log/failures.jsonl`, lower `--max-workers` (default 8 → try 4) to reduce concurrent calls.

### "no seeds at <path>"

Check that the path exists, has the right extension, and that the file isn't empty. For `.txt` files, ensure documents are separated by `\n---\n` (or one per line if you have very short seeds).

### "no recognized text column" for `.parquet`

The package looks for `text`, `text_cleaned`, `combined_text`, `article`, or `cleaned_article`. If your column has another name, either rename it or pre-process the parquet.

### Output JSONL has fewer records than expected

Look at `out/log/failures.jsonl`:
- `"error": ...` — API call failed or response wasn't valid JSON. Reduce `--max-workers` if persistent.
- `"reason": "banned_term"` — leakage filter dropped the record. Check whether the term is too aggressive or whether the prompt needs tightening.
- `"reason": "parse_failed"` — the model returned valid JSON but the wrong shape. Inspect the prompt; the model may need a clearer output-format instruction.

### Seed text too long

Defaults truncate seeds to 4000 characters before sending. Override via `config.yaml` (`max_seed_chars: <N>`). Cost scales linearly with seed length.

### Prompt isn't producing the right output

Edit `prompts/<file>.md` directly and rerun. The package re-reads prompts on every run; no caching. For systematic iteration, copy `prompts/` to multiple directories and pass `--prompts /path/to/v2/`.

---

## End of brief

If you're an LLM reading this: when the user asks for help customizing, debugging, or extending these packages, refer back to the relevant section above. The codebase is small (run.py is ~300 lines per package); read it directly for any details not covered here.
