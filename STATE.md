# policy_pred — handoff state (2026-04-30)

## Goal & hypothesis
Build year-stamped LLM models (one per year), each trained cumulatively on text
through year Y, then probe each year-model for the probability of historical
policy enactment. Hypothesis: P(impl | year-Y model) rises as Y approaches the
actual implementation year and peaks at or shortly before enactment. Research
project — methodology paper, not a product.

## Pipeline stages (CLI in `pipeline.py`)
1. **slice** — pull per-year text shards from the historical corpus
2. **synth** — OpenAI batch API → synthetic instruction data per year
3. **train** — cumulative midtrain (raw text) + SFT (synth) per year, init from
   prior year's checkpoint
4. **eval** — for each policy in `policies/catalog.yaml`, probe each year-model,
   save normalized log-probs
5. **analyze** — build P(impl | year) trajectories, compute argmax-year and
   trajectory metrics
6. **setup-model** — populate a registered base model into D: (HF download for
   talkie, copy/symlink for nanochat)

## Architecture decisions
- **Cumulative, not independent**: year-Y model = base + 1931..Y. The model spec
  is carried forward year by year; only `path` is replaced.
- **Backend abstraction**: `models/base.py` `ModelBackend` with two impls
  (nanochat, talkie). All pipeline code goes through `factory.load_backend()` —
  swapping backends is a one-token change in `config.BASE_MODEL`.
- **Registry**: `models/registry.yaml` maps stable names (`talkie_base`,
  `nanochat_1925`) to weights paths and HF ids. `config.BASE_MODEL` is a name.
- **Eval primitive**: `score_continuations(prompt, options) -> [logprob]`,
  length-normalized. Same call shape works for yes/no and MC probes.
- **Synthesis must NOT mention catalog policies by name** — leakage invalidates
  the trajectory. `synthesize/prompts.py` is empty pending design + dry-run
  review. This is the highest-risk single piece of the project.

## What's real vs stub
**Real**:
- `config.py`, `pipeline.py` orchestrator
- `models/factory.py`, `models/registry.py`, `models/registry.yaml`
- `policies/catalog.yaml` (5 seed policies), `policies/__init__.py` with
  `load_catalog()`

**`NotImplementedError` stubs** awaiting design + implementation:
- `corpus/year_slice.py`
- `synthesize/prompts.py` (empty), `synthesize/generate.py`
- `train/midtrain.py`, `train/sft.py`
- `policies/probes.py`
- `eval/elicit.py`, `eval/analyze.py`
- `models/nanochat.py`, `models/talkie.py`

## Base model going forward: Talkie-LM
- HF id: `talkie-lm/talkie-1930-13b-base`. Apache-2.0. 13B params. Trained on
  260B tokens of pre-1931 English.
- **Weights already on D:** at `D:/hist_LLM/policy_pred/models/talkie_base/`
  (53.12 GB `final.ckpt` + 4.6 MB `vocab.txt`). The data folder was deliberately
  left in place when the code folder moved out of `hist_LLM` — 53 GB is too big
  to relocate casually. `models/registry.yaml` still references this path.
- **Format gotchas**: ships as a single PyTorch checkpoint (NOT a transformers
  checkpoint) plus a custom byte-level vocab (`vocab.txt`, 262144 base64-encoded
  byte tokens). `AutoModelForCausalLM.from_pretrained` will not work on this.
- **Their repo (https://github.com/talkie-lm/talkie) is inference-only.**
  Files: `src/talkie/{model.py, tokenizer.py, sampling.py, generate.py,
  chat.py, cli.py, config.py, download.py}`. No optimizer, no loss, no
  dataloader, no training loop. Pure PyTorch — no custom CUDA kernels.
- **Architecture**: GPT-style with RMSNorm + RoPE + multi-head attention (NO
  GQA), plus extra learnable gain modules (`HeadGain`, `WeightGain`, `ActGain`)
  and embedding skip connections with learnable gain. Standard `nn.Module` —
  composes with PEFT, FSDP, accelerate.

## Plan for using Talkie
1. **Vendor** `model.py` + `tokenizer.py` + `sampling.py` from talkie's repo
   into `models/talkie_vendor/`. Apache-2.0, ~few hundred lines total.
2. **Implement `TalkieBackend.score_continuations`** using their loader. This
   alone unblocks the eval-only baseline (probe Talkie cold without training).
3. **Smoke test**: load the 53 GB ckpt, score "Yes" vs "No" on a sample probe,
   verify log-probs are sensible.
4. **Then training**: their inference forward likely returns last-token logits
   only; expose `[B, T, V]` for training. Build a small accelerate-based loop
   or HF Trainer wrapper around their `nn.Module`. Add LoRA via PEFT, targeting
   attention QKV/proj + MLP linears.
5. **Scale**: 13B + LoRA fits a 24 GB GPU; per-year adapter ~100 MB. Full FT is
   a non-starter (50 GB ckpt × 40 years = 2 TB, plus needs FSDP + multi-GPU
   80GB). LoRA only.

## Data layout (on D:, NOT moved)
```
D:\hist_LLM\policy_pred\           <- data root (path stays, even though code moved)
├── models\
│   └── talkie_base\               <- 53 GB, populated
├── years\{Y}\                     <- written by slice/synth/train stages
│   ├── raw.parquet
│   ├── sft.jsonl
│   └── checkpoint\{midtrain,sft}\
└── eval\{policy_id}\{Y}.json
```
Source corpus stays at `D:\hist_LLM\corpus\raw\` and
`D:\hist_LLM\additional_data\raw\` — `corpus/year_slice.py` reads from there
and writes to `D:\hist_LLM\policy_pred\years\Y\`.

## Recommended corpus subset for 1900-1950
Inputs to `corpus/year_slice.py`:

**Tier 1 (high signal for policy discourse)**:
- `D:\hist_LLM\additional_data\raw\newswire\{year}_data_clean.json`
- `D:\hist_LLM\additional_data\raw\news_archives\NYT_filtered_500char\nyt_{year}.parquet`
- `D:\hist_LLM\additional_data\raw\news_archives\FT\{year}.parquet`

**Tier 2 (background fluency, after quality filter)**:
- `D:\hist_LLM\corpus\raw\{year}\subset_*.parquet` — pass through the existing
  Ridge quality classifier (3M-of-10.8M docs survive for 1900-1949 above
  threshold).

**Skip for v1**: Economist (per-week parquets — needs aggregation overhead),
foreign-language collections, `periods/1900_1949/base_data` (mixed across years,
can't slice).

Per-year output budget after dedup + quality filter: ~2-6 GB / 0.5-1.5 B tokens.

## Catalog (5 seed policies)
See `policies/catalog.yaml`. Each entry has `implementation_year`,
`anticipation_start_year`, `domain`, `region`, `source`, `notes`:
- `social_security_1935` (impl 1935, anticipated 1929+)
- `bretton_woods_1944` (impl 1944, anticipated 1941+)
- `marshall_plan_1948` (impl 1948, anticipated 1946+)
- `civil_rights_1964` (impl 1964, anticipated 1955+)
- `medicare_1965` (impl 1965, anticipated 1945+)

`config.py` year range: `START_YEAR=1931`, `END_YEAR=1970`.

## Open questions
1. **Probe form**: Yes/No with log-probs vs. 4-way MC with plausible distractors.
   Yes/No is simpler and more directly interpretable as P(impl); MC is better
   calibrated but distractor design becomes a confound.
2. **Synthesis prompts**: must avoid catalog policy leakage. Need template
   design + manual review of dry-run on a sample year (e.g. 1935). Highest-risk
   single piece — bad prompts invalidate every downstream stage.
3. **Training stack**: HF Trainer + PEFT vs. small custom accelerate loop.
   Talkie's `nn.Module` composes with both; pick by least friction for 13B +
   LoRA on the available hardware.

## Recommended next step
Vendor talkie's `model.py` + `tokenizer.py` + `sampling.py` into
`models/talkie_vendor/`, implement `TalkieBackend.score_continuations`, and run
a smoke test: load the 53 GB ckpt, score `"Yes"` vs `"No"` on a sample probe,
verify the log-probs are sensible. This unblocks the eval-only baseline path
before any training infrastructure work.
