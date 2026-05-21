# synth_2step

A standalone, two-stage synthetic-data generator. For each seed document:

1. **Ideate** — one OpenAI call → list of N `{doc_type, concept, tone}` records.
2. **Write** — one OpenAI call per idea → full synthetic document.

Output is JSONL with `{"text": "..."}` records, one per generated document. Total per seed: `1 + N` API calls.

This is the diversity-focused sibling of `../synth_naive/`. The two-stage decomposition is inspired by Anthropic's [Model Spec Midtraining repo](https://github.com/chloeli-15/model_spec_midtraining), with their hierarchical domain/subdomain decomposition stripped — we treat the seed text itself as the diversity source.

## Quick start

```bash
cd synth_2step/
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...

python run.py
```

Defaults read `examples/seeds.txt` (3 sample 1930s-style docs, 8 ideas each → 24 documents) and write to `out/`.

## Pointing at your own data

```bash
# Parquet with a "text" column
python run.py --seeds /path/to/your/corpus.parquet --out /path/to/output/

# JSONL with {"text": "..."} records
python run.py --seeds /path/to/your/corpus.jsonl --out /path/to/output/

# Directory of .txt files (one document per file)
python run.py --seeds /path/to/your/seed_dir/ --out /path/to/output/

# Plain .txt with documents separated by "\n---\n"
python run.py --seeds /path/to/your/seeds.txt --out /path/to/output/
```

## All knobs

```bash
python run.py \
    --seeds <path>             \  # input
    --out <dir>                \  # output dir, created if missing
    --model gpt-4o-mini        \
    --n-ideas 8                \  # docs per seed; total calls = (1+N)*seeds
    --temperature 0.8          \
    --max-tokens-ideas 1024    \  # stage 1 budget
    --max-tokens-writer 1500   \  # stage 2 budget per doc
    --max-workers 8            \
    --limit 5                  \  # cap seeds (smoke run)
    --banned-terms <file>      \  # leakage filter
    --prompts <dir>            \  # custom prompts dir (must have ideas.md + writer.md)
```

Or edit `config.yaml` for project-wide defaults. CLI flags override config.

## Output

```
out/
  ideas.jsonl              # one record per generated idea (auditable, intermediate)
  synth.jsonl              # one {"text": "..."} record per generated document
  log/
    config.yaml            # frozen copy of the effective config
    failures.jsonl         # per-call errors and filter rejections
```

`ideas.jsonl` is intermediate but kept around so you can inspect the brainstormed concepts before they get expanded — useful for prompt-iteration.

Use `synth.jsonl` directly as continued-pretraining data. Each record is plain text plus a `metadata` field (seed index, idea index, doc_type, tone, generator name).

## Customizing the prompts

Edit `prompts/ideas.md` (stage 1) and `prompts/writer.md` (stage 2), or pass `--prompts /path/to/your/prompts/`.

The two files use these conventions:

- `prompts/ideas.md` — system prompt for stage 1. The user message is built from `{seed}` and `{n_ideas}` (the script substitutes them at runtime). Output expected: `{"ideas": [{"doc_type": "...", "concept": "...", "tone": "..."}, ...]}`.
- `prompts/writer.md` — system prompt for stage 2. The user message includes the seed (for era context, do-not-reference) and the idea record. Output expected: `{"document": "..."}`.

## Leakage filter

For era-restricted training where certain canonical terms (e.g., "Social Security Act") would defeat the experiment if they appeared verbatim, pass `--banned-terms terms.txt` (one term per line, case-insensitive substring match). Output records containing any banned term are dropped and logged to `log/failures.jsonl`.

## Cost rough estimate (gpt-4o-mini, default `n_ideas=8`)

500 seeds × (1 ideate + 8 write) = 4500 calls. Average call is ~1500 tokens out → ~7M output tokens → **~$5-15 of API spend**.

For larger experiments scale linearly. gpt-4o costs roughly 10× more.

## When to use this vs synth_naive

- **synth_naive** (sibling package): one call per seed, faster, cheaper, more homogeneous outputs. Good for big-volume runs where diversity matters less.
- **synth_2step** (this package): ~3-9× the API spend per seed but produces much more diverse outputs. The intermediate `ideas.jsonl` is auditable and helps prompt iteration. Good for small/medium-volume runs where every doc should pull its weight.

A/B them on your data and decide by the diversity / quality of the outputs.
