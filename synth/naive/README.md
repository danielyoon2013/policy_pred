# synth_naive

A standalone, single-file synthetic-data generator. For each seed document, makes **one OpenAI call** asking for N synthetic training documents inspired by the seed. Output is JSONL with `{"text": "..."}` records.

This is the simpler of two synth packages — see also `../synth_2step/`, which uses an Anthropic-style two-stage (ideate → execute) pipeline that produces more diverse output at higher cost.

## Quick start

```bash
cd synth_naive/
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...

python run.py
```

That's it. Defaults read `examples/seeds.txt` (3 sample 1930s-style docs) and write `out/synth.jsonl`.

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

`--seeds` auto-detects format by extension and content.

## All knobs

```bash
python run.py \
    --seeds <path>         \  # input
    --out <dir>            \  # output dir, created if missing
    --model gpt-4o-mini    \  # OpenAI model
    --n-per-seed 3         \  # docs to generate per seed
    --temperature 0.7      \
    --max-tokens 1024      \
    --max-workers 8        \  # parallel API calls
    --limit 10             \  # cap seeds (smoke run)
    --banned-terms <file>  \  # leakage filter (1 term per line)
    --prompts <dir>        \  # custom prompts dir (must have system.md + user.md)
```

Or edit `config.yaml` for project-wide defaults. CLI flags override config.

## Output

```
out/
  synth.jsonl              # one {"text": "..."} record per generated document
  log/
    config.yaml            # frozen copy of the effective config
    failures.jsonl         # per-seed errors and filter rejections (for retry/audit)
```

Use `synth.jsonl` directly as continued-pretraining data. Each record is plain text; no chat template, no special tokens.

## Customizing the prompt

Edit `prompts/system.md` and `prompts/user.md`, OR drop a `prompts/` folder anywhere and pass `--prompts /path/to/your/prompts/`.

The two prompt files use these placeholders:
- `prompts/system.md` — no placeholders. Just the static system message.
- `prompts/user.md` — `{seed}` (the seed text), `{n_per_seed}` (an integer).

Output must be JSON of shape `{"documents": ["doc 1...", "doc 2...", ...]}`.

## Leakage filter

For era-restricted training where certain canonical terms (e.g., "Social Security Act") would defeat the experiment if they appeared verbatim, pass `--banned-terms terms.txt` (one term per line, case-insensitive substring match). Output records containing any banned term are dropped and logged to `log/failures.jsonl` with `"reason": "banned_term"`.

## Cost rough estimate (gpt-4o-mini)

500 seeds × 1 call × ~1500 tokens out per call → ~750K output tokens → **~$0.50-2 of API spend**.

For larger experiments scale linearly. gpt-4o costs roughly 10× more.

## When to use this vs synth_2step

- **synth_naive** (this package): faster, cheaper, more homogeneous outputs. Good for big-volume runs where diversity matters less.
- **synth_2step**: ~2× the API spend per seed but produces much more diverse outputs (different doc types, framings). Good for small-volume runs where every doc should pull its weight.

A/B them on your data and decide.
