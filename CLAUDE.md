# policy_pred — durable project guidance

Read `STATE.md` first for current per-stage status; this file is for rules that don't change between sessions.

## Reading order

1. `STATE.md` — current real-vs-stub status, open questions, latest recommendation.
2. This file — durable architecture, conventions, gotchas.
3. `policies/catalog.yaml` — the policies we elicit beliefs about (schema in flux; see Catalog section below).

## Project goal

Build year-stamped LLM models, one per year 1931-1970, each = previous year's model + that year's text, via cumulative LoRA midtraining on top of Talkie-LM 13B (a base model trained on pre-1931 English, vendored as a git submodule). The probe layer asks each year-model questions about catalog policies and measures how `P(impl | year-Y)` evolves as `Y` approaches the actual enactment year.

Hypothesis: a model trained only on text through year Y "believes in" a policy more strongly as Y approaches the enactment year, even though Y < enactment year. Methodology paper, not product.

## Architectural rules

- **Cumulative training only.** Year-Y model = year-(Y-1)'s LoRA adapter + this year's text shard. The chain must not be reordered or parallelised across years.
- **One base model: Talkie.** No backend abstraction layer (we tried, it was premature). If we ever need a second base model, we'll add it then, not pre-emptively.
- **LoRA-only.** Full fine-tuning of 13B is ruled out — see Compute model below.
- **V1 is real text only.** Generative synthesis (using OpenAI to expand pre-enactment policy discourse for SFT) is V2 territory; don't rebuild it inline. See "V2: synthesis (deferred)" below.
- **Length-normalized log-prob is the eval primitive.** `score_continuations(prompt, options)` returns per-token average log-prob. Same shape works for Yes/No and 4-way MC.

## Repository conventions

- `pathlib.Path` for filesystem paths in function signatures, never `str`.
- Type hints on public functions (anything callable across modules). Internal helpers may omit them.
- Plain prose docstrings — explain *why*, not *what*. No Args/Returns/Raises sections; the type hints are the contract.
- One responsibility per module. No orchestration shim — the script sequence below is the orchestration.

## Repository layout

```
policy_pred/
  config.py                    paths, year range, hyperparameters
  talkie.py                    TalkieBackend: streaming bf16 loader + score_continuations
  midtrain.py                  LoRA continued pretraining loop (CUDA only)
  STATE.md                     current state (changes per session)
  CLAUDE.md                    durable rules (this file)
  __init__.py                  empty, makes the dir a Python package
  policies/catalog.yaml        the policies (work-in-progress)
  docs/                        runbooks (REMOTE_SETUP.md, etc.)
  models/talkie_vendor/        git submodule, NEVER edit
  scripts/
    smoke_talkie.py            validate base loader + sanity probes
    probe_with_adapter.py      compare base vs base+adapter on the same probes
    inspect_corpus.py          corpus collection breakdowns by year
```

## Data layout (on D:)

The data root is on `D:` even though the code lives on Dropbox. The 53 GB Talkie checkpoint is too big to relocate.

```
D:/hist_LLM/policy_pred/
  models/talkie_base/{final.ckpt, vocab.txt}    # 53 GB + 4.6 MB
  years/{Y}/
    raw.parquet                                 # written by slice_corpus.py
    checkpoint/                                 # written by midtrain.py (LoRA adapter)
  eval/{policy_id}/{Y}.json                     # written by probe.py
```

Source corpora live at `D:/hist_LLM/corpus/raw/` (bulk per-year subsets) and `D:/hist_LLM/additional_data/raw/{newswire,news_archives}/` (Tier-1 wire/NYT/FT). Path helpers in `config.py` (`year_dir`, `year_corpus_path`, `year_checkpoint_dir`, `policy_eval_path`) are the only place these paths should be constructed — never join paths inline.

## Submodule rule

Never edit anything under `models/talkie_vendor/`. It is a git submodule of `talkie-lm/talkie` (Apache-2.0, inference-only). Customisations live in `talkie.py` at the repo root, which imports from the vendor and overrides where needed (e.g. `_forward_all_positions` re-implements `TalkieModel.forward` to expose `[B, T, V]` logits). Updating the vendor is `git submodule update`, not a manual copy.

## Compute model

LoRA-only on Talkie 13B. Full FT is ruled out: ~150-200 GB GPU memory for the optimizer state alone (needs FSDP across 4× A100 80GB) and ~50 GB checkpoint per year × 40 years = 2 TB on disk. LoRA stays at ~28 GB peak (frozen bf16 base ~26 GB + adapter ~120 MB + activations) and fits one A100 80GB with comfortable headroom. DDP if scaling beyond one GPU; no FSDP for 13B. Local CPU is sufficient for plumbing and dev (slice/probe/analyze); training itself goes to rented A100s.

## CKPT format gotcha

Talkie ships as a single ~53 GB fp32 `final.ckpt` (raw PyTorch state_dict, possibly nested under `model_state_dict` or `model`) plus a byte-level `vocab.txt` (tiktoken-loaded BPE). It is **not** a transformers checkpoint — `AutoModelForCausalLM.from_pretrained` will not work. Always go through `TalkieBackend` in `talkie.py`, which uses the streaming-bf16 loader (`_load_ckpt_streamed`): `torch.load(..., mmap=True)` so the fp32 state dict is paged in, copied tensor-by-tensor into a pre-allocated bf16 model, then released. Steady peak ~28 GB instead of the vendored loader's ~100 GB spike. Tokenizer: `talkie.tokenizer.build_tokenizer(weights_dir / "vocab.txt", style="base")`. Do not substitute a HF tokenizer.

## V2: synthesis (deferred)

Generative synthesis using OpenAI is planned for V2 to capture pre-enactment policy motivation (e.g. labor-movement discourse leading up to Social Security 1935). Highest-risk piece of the project: GPT-4 has 2024 training data and will leak post-1930 knowledge into supposedly era-restricted output. When V2 starts:

- No catalog policy `name` or `source` substring may appear in synth output. Audit by grep before scaling past one year.
- The catalog `description` field is deliberately a paraphrase, not the canonical act name — use the same discipline in synth prompts.
- Prefer extraction-only synthesis (rephrase/summarize existing era text) over fully generative synthesis.

## Catalog

Schema is **in flux** — see `policies/catalog.yaml` for current fields. Durable rule: **NO `canonical_act_name` field, by design.** Verbatim recall of "Social Security Act" by name in the training set defeats the trajectory measurement for Social Security; the catalog's `description` paraphrase is the leakage-prevention discipline. Other fields (`implementation_year`, `anticipation_start_year`, etc.) may evolve as the research design firms up.

## Pipeline (script sequence)

There is no CLI orchestrator. The scripts are run in this order:

```
python scripts/inspect_corpus.py              # one-time, exploratory (CPU)
python scripts/smoke_talkie.py                # validate base loads (CPU OK)
python midtrain.py --data <parquet>           # LoRA train one year (CUDA only)
python scripts/probe_with_adapter.py --adapter <dir>   # compare base vs +adapter
```

For the V1 validation run (year 1931 only) see `docs/REMOTE_SETUP.md`. `midtrain.py` will eventually become sequential per year (year Y starts from year (Y-1)'s adapter); the V2 expansion adds a `slice_corpus.py` for aggregated year shards and an `analyze.py` for trajectory plots once we have multiple years. Run scripts from the project root with `.venv/` activated.

## Pointers

- `talkie.py::TalkieBackend` — the loader + scoring class.
- `talkie.py::_load_ckpt_streamed` — memory-efficient ckpt loader.
- `talkie.py::_forward_all_positions` — `[B, T, V]` forward (replicates vendored `TalkieModel.forward` minus the last-token slice).
- `midtrain.py` — LoRA continued-pretraining loop; refuses to run on CPU.
- `scripts/smoke_talkie.py` — reference implementation of how to use TalkieBackend end-to-end.
- `scripts/probe_with_adapter.py` — base vs base+adapter probe comparison.
- `docs/REMOTE_SETUP.md` — runbook for renting an A100 and running V1 end-to-end.
- `STATE.md` — current state; update it when you finish a session.
