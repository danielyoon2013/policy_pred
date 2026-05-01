# Remote setup — LoRA midtrain Talkie on year-1931 data

End-to-end runbook for the V1 validation run: provision an A100 80GB,
LoRA-midtrain Talkie on 1931 NYT data, and check whether the adapter
shifted the model's behaviour. Expected total cost: ~$5-15.

## 1. Provision the box

Pick any of: Lambda Labs (~$1.10/hr A100 80GB on-demand), vast.ai
(~$0.80-1.50/hr spot), RunPod (similar). I'd use Lambda for the first
run — no spot eviction risk during a 2-hour training job.

Spec: **1× A100 80GB**, Ubuntu 22.04, Python 3.11+, CUDA 12.x. Default
ephemeral disk is fine (we only need ~80 GB free).

SSH in. All commands below run on the remote box unless noted.

## 2. Clone repo + submodule

```bash
git clone https://github.com/danielyoon2013/policy_pred.git
cd policy_pred
git submodule update --init --recursive
```

## 3. Python env (CUDA build of torch)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# CUDA torch (NOT the CPU build the local box uses)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Training extras
pip install peft accelerate

# Same deps the local box has
pip install tiktoken huggingface-hub pyyaml pandas pyarrow numpy
```

## 4. Download Talkie weights (~5-10 min)

```bash
huggingface-cli download talkie-lm/talkie-1930-13b-base \
    --local-dir ~/talkie_base \
    --local-dir-use-symlinks False
```

Result: `~/talkie_base/{final.ckpt, vocab.txt, ...}`.

## 5. Point the code at the remote weights path

Two options. Pick one.

### Env var (preferred — no code edit)
```bash
export TALKIE_WEIGHTS_DIR=$HOME/talkie_base
```
Add it to `~/.bashrc` if you'll use this box across sessions. `config.py`
respects this env var on every script invocation.

### Inline edit
Edit `config.py`, change `TALKIE_WEIGHTS_DIR = ...` to your path. Don't
commit this — the env-var path is the durable approach.

## 6. Get 1931 training data onto the remote

From your **local** box (not the remote):
```bash
scp D:/hist_LLM/additional_data/raw/news_archives/NYT_filtered_500char/nyt_1931.parquet \
    user@REMOTE-IP:~/policy_pred/data/
```

(Create the `data/` dir first on the remote: `mkdir -p ~/policy_pred/data`.)

NYT 1931 is ~50 MB and ~24K articles. Enough text (~30M tokens) to give
the LoRA something to learn from in V1.

## 7. Smoke test the loader (~5-10 min)

Quick check that Talkie loads correctly on the remote before paying for
a full training run:

```bash
python scripts/smoke_talkie.py
```

Expect: Paris > London on the first probe (clear PASS), London > Paris
on the second. If the smoke test fails, do not proceed — fix loading
issues first.

## 8. Run midtrain (~1-2 hours)

```bash
python midtrain.py \
    --data data/nyt_1931.parquet \
    --rank 32 \
    --out checkpoints/year_1931
```

What to watch:
- **Loss should decrease** over the first 100-200 steps. Starts ~3-5,
  drops as the LoRA learns.
- **`tok/s`** stays roughly constant after warmup; A100 80GB target is
  ~3-8K tok/s for 13B + LoRA r=32 with grad checkpointing on.
- **No OOM**. If it OOMs, reduce `--batch-size` to 4 or even 2.

The output adapter at `checkpoints/year_1931/` is ~120 MB and contains
`adapter_config.json` + `adapter_model.safetensors`.

### Optional: tiny test run first
If you want to verify the loop runs end-to-end before committing to the
full ~1-2 hour run, do a 50-step smoke training:
```bash
python midtrain.py \
    --data data/nyt_1931.parquet \
    --rank 32 \
    --max-steps 50 \
    --out checkpoints/test_run
```
~5-10 min on A100. Loss should drop a bit; that's enough to know the
loop is working.

## 9. Compare base vs base+adapter

```bash
python scripts/probe_with_adapter.py --adapter checkpoints/year_1931
```

Reads:
- **Sanity probes** (Paris/London) should still PASS — the adapter didn't
  break the base.
- **SS-1930** scores should differ from base. The script prints `Δ` per
  option and a max-Δ summary. >0.05 nats means training had measurable
  effect.

## 10. Pull the adapter back to local

From your **local** box:
```bash
scp -r user@REMOTE-IP:~/policy_pred/checkpoints/year_1931 ./checkpoints/
```

120 MB; takes seconds. You now have a year-1931 LoRA adapter on local.

## 11. Stop the box

Don't forget. Lambda / RunPod don't stop billing automatically when you
close the SSH session.

---

## Troubleshooting

**`midtraining requires CUDA, got device=cpu`** — torch fell back to CPU
build. `pip uninstall torch && pip install torch --index-url
https://download.pytorch.org/whl/cu121`.

**OOM during training** — try `--batch-size 4` (or 2). If still OOM,
also drop `--seq-len 1024`. Grad checkpointing is on by default; if
you'd disabled it, re-enable it.

**`talkie.model` ImportError** — submodule not initialized. `git
submodule update --init --recursive`.

**HF download is slow** — set `HF_HUB_ENABLE_HF_TRANSFER=1` and `pip
install hf_transfer` before re-running `huggingface-cli download`.

**Adapter "doesn't move" the SS-1930 probe** — this is a real research
finding, not a bug. Could mean: the rank is too low (try `--rank 128`),
the training set is too small (need more 1931 text, not just NYT), or
the probe form is too noisy (try a year-completion probe instead).
