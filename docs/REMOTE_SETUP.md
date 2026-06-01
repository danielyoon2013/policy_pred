# Remote setup — runbook for A100 pod

End-to-end runbook for training Talkie+LoRA on a rented GPU pod. Two experiments documented here:
- **Experiment A**: math re-train with MC4-format synth, eval on GSM-MC (~$5-10, ~30 min on 4 GPUs)
- **Experiment B**: policy POC for SS-1935, comparing synth_naive vs synth_2step (~$15-30, ~1.5 hrs)

The setup section (steps 1-5) is the same for both; jump to the experiment-specific runbook once setup is done.

---

## 1. Provision the box

Rent any of: Lambda Labs (~$1.10/hr A100 80GB on-demand), vast.ai (~$0.80-1.50/hr spot), RunPod (similar). For multi-GPU runs: **4× A100 80GB** is the sweet spot. For single-GPU, 1× A100 80GB is enough.

Spec: Ubuntu 22.04, Python 3.11+, CUDA 12.x. Default ephemeral disk OK (~80 GB free needed).

SSH in. All commands below run on the remote box unless noted.

## 2. Clone repo + submodule

```bash
git clone https://github.com/danielyoon2013/policy_pred.git
cd policy_pred
git submodule update --init --recursive
```

If the box already has the repo from a previous session: `git pull && git submodule update --init --recursive`.

## 3. Python env (CUDA torch + training/inference deps)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# CUDA torch (NOT the CPU build the local box uses)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Training extras — versions pinned because newer transformers (>=4.55) ships
# a MoE module that calls torch.library.custom_op with a signature the cu121
# torch wheel doesn't accept, and peft transitively imports transformers at
# `from peft import LoraConfig`, which crashes training before it starts.
pip install 'transformers==4.46.3' 'peft==0.13.2' accelerate

# Eval/inference + general deps
pip install tiktoken huggingface-hub pyyaml pandas pyarrow numpy openai
```

## 4. Download Talkie weights (~5-10 min)

```bash
hf download talkie-lm/talkie-1930-13b-base --local-dir ~/talkie_base
```

(`huggingface-cli` was renamed to `hf` in late 2025.)

Sanity check: `ls -la ~/talkie_base/` should show `final.ckpt` (~53 GB) + `vocab.txt` (~4.6 MB).

## 5. Point the code at remote paths

The codebase respects two env vars for redirection on remote boxes (D:/ doesn't exist there). Set both AND persist them so a fresh SSH session picks them up:

```bash
# Persist (for future SSH sessions, login shells)
echo 'export TALKIE_WEIGHTS_DIR=$HOME/talkie_base' >> ~/.bashrc
echo 'export POLICY_PRED_DATA_ROOT=$HOME/policy_pred_data' >> ~/.bashrc

# Apply now (current shell) — .bashrc isn't auto-sourced by an already-open shell
export TALKIE_WEIGHTS_DIR=$HOME/talkie_base
export POLICY_PRED_DATA_ROOT=$HOME/policy_pred_data
mkdir -p $POLICY_PRED_DATA_ROOT ~/data

# Verify — both files MUST exist or training/eval will fail with FileNotFoundError
ls -la $TALKIE_WEIGHTS_DIR/final.ckpt $TALKIE_WEIGHTS_DIR/vocab.txt
```

`config.py` reads both env vars. With these set, all script outputs land at `$HOME/policy_pred_data/...` instead of `D:/hist_LLM/policy_pred/...`.

**If you open a new SSH session later, re-export both before running anything** (the `.bashrc` line covers interactive logins, but some pods use non-interactive shells that skip it).

## 6. Smoke test the loader (~5-10 min)

Cheap check that Talkie loads correctly before paying for a full training run:

```bash
python scripts/smoke_talkie.py
```

Expected: Paris > London on the first probe (PASS), some preference on the second probe (often opposite-direction is fine — the absolute log-probs are tiny and noisy for very-low-prob tokens), and clear directionality on the SS-1930 probe (P(False) >> P(True), confirming the cutoff effect).

If the smoke test errors out (not just a content sanity-check fail), do not proceed; fix loading first.

---

# Experiment A: math re-train with MC4 format (~$5-10, ~30 min on 4 GPUs)

**Goal**: test whether the GSM-MC chance-level result (24.6%) was a format-mismatch artifact. Train on MC4-format math synth (matches the GSM-MC eval format) instead of CoT-format.

**Decision criterion**:
- Accuracy ≥ 30% → format was the issue, pipeline picks up math signal when format-matched
- Accuracy still ~25% → math is OOD for Talkie regardless of format; move on

## A.1. From local box: scp the data

```bash
# Replace REMOTE_USER@REMOTE_IP with your pod's connection string
scp "D:/hist_LLM/periods/1900_1949/posttraining_data/synthetic/by_generator/gen_d_quantitative_mc4.jsonl" \
    REMOTE_USER@REMOTE_IP:~/data/

scp "D:/hist_LLM/eval_data/gsm_mc.jsonl" \
    REMOTE_USER@REMOTE_IP:~/data/
```

## A.2. On remote: update YAML test_set path + train + eval

```bash
cd ~/policy_pred
source .venv/bin/activate

# Update YAML eval test_set to remote path (D:/ -> /root/data/)
sed -i 's|D:/hist_LLM/eval_data/gsm_mc.jsonl|/root/data/gsm_mc.jsonl|' \
    experiments/math_lora_histllm.yaml
# (use $HOME/data instead of /root/data if your username isn't root)

# 4-GPU DDP train. Per-rank batch 2 keeps effective batch ~8.
accelerate launch --num_processes=4 --mixed_precision=no \
    train.py \
    --experiment experiments/math_lora_histllm.yaml \
    --data ~/data/gen_d_quantitative_mc4.jsonl \
    --batch-size 2

# Full GSM-MC eval (1319 problems) on the trained adapter
python eval.py --experiment experiments/math_lora_histllm.yaml
```

**Expected wall clock**: ~25-30 min total on 4 GPUs (vs ~2 hrs single-GPU).

**Compare to**:
- 18.2% — nanochat-1900-1949 SFT
- 24.6% — our previous CoT-trained adapter
- 24.0% — always-A baseline

## A.3. Pull the result back to local

```bash
# From local box
scp REMOTE_USER@REMOTE_IP:~/policy_pred_data/experiments/math_lora_histllm/eval.json \
    "D:/hist_LLM/policy_pred/experiments/math_lora_histllm/"
```

---

# Experiment B: Policy POC for Social Security 1935

**Goal**: end-to-end policy pipeline. Generate synth from 1931 legal corpus via BOTH `synth_naive` and `synth_2step` (~1000 seeds each, the same seed pool). Train two LoRAs, eval each on the SS-1935 probe (yes_no + likert5 modes), compare.

## B.1. Local: slice corpus + run both synth methods

This step happens on **your local box** (uses `D:/hist_LLM/...` raw corpus + the `synth_naive` and `synth_2step` packages we already shipped to Leland).

### Phased rollout (smoke first, scale only if smoke is clean)

**Smoke (~$2-4, ~10 min)** — 50 seeds, both methods:
```bash
python sources/year_slice_legal.py --year 1931 \
    --out D:/hist_LLM/policy_pred/years/1931/legal.parquet

cd synth_naive
python run.py --seeds D:/hist_LLM/policy_pred/years/1931/legal.parquet \
    --out D:/hist_LLM/policy_pred/years/1931/naive_smoke/ \
    --limit 50 --n-per-seed 4
cd ..

cd synth_2step
python run.py --seeds D:/hist_LLM/policy_pred/years/1931/legal.parquet \
    --out D:/hist_LLM/policy_pred/years/1931/2step_smoke/ \
    --limit 50 --n-ideas 8
cd ..

# Hand-read 5-10 records from each smoke output. Are they era-faithful and
# capturing moral/social discourse? If yes, scale up. If no, iterate prompts.
```

**Scale (~$25-40, ~30-60 min)** — 1000 seeds, both methods:
```bash
cd synth_naive
python run.py --seeds D:/hist_LLM/policy_pred/years/1931/legal.parquet \
    --out D:/hist_LLM/policy_pred/years/1931/naive/ \
    --limit 1000 --n-per-seed 4
cd ..

cd synth_2step
python run.py --seeds D:/hist_LLM/policy_pred/years/1931/legal.parquet \
    --out D:/hist_LLM/policy_pred/years/1931/2step/ \
    --limit 1000 --n-ideas 8
cd ..
```

## B.2. scp synth + battery CSV to remote

```bash
# From local box
scp D:/hist_LLM/policy_pred/years/1931/naive/synth.jsonl \
    REMOTE_USER@REMOTE_IP:~/data/policy_1931_naive.jsonl

scp D:/hist_LLM/policy_pred/years/1931/2step/synth.jsonl \
    REMOTE_USER@REMOTE_IP:~/data/policy_1931_2step.jsonl

# Plus the policy battery CSV (the evaluator reads it)
scp us_policy_event_battery_v4.csv \
    REMOTE_USER@REMOTE_IP:~/policy_pred/
```

## B.3. On remote: eval base + train two LoRAs + eval each (~$15-30, ~1.5 hrs)

```bash
cd ~/policy_pred && source .venv/bin/activate

# 1. Baseline: eval Talkie base (no adapter) on SS-1935 probe (~10 min)
python eval.py --experiment experiments/policy_1931_base.yaml --no-adapter

# 2. Train LoRA on naive synth (~25-30 min on 4 GPUs)
accelerate launch --num_processes=4 --mixed_precision=no \
    train.py \
    --experiment experiments/policy_1931_naive.yaml \
    --data ~/data/policy_1931_naive.jsonl \
    --batch-size 2

# 3. Eval the naive-trained adapter (~10 min)
python eval.py --experiment experiments/policy_1931_naive.yaml

# 4. Train LoRA on 2step synth (~25-30 min)
accelerate launch --num_processes=4 --mixed_precision=no \
    train.py \
    --experiment experiments/policy_1931_2step.yaml \
    --data ~/data/policy_1931_2step.jsonl \
    --batch-size 2

# 5. Eval the 2step-trained adapter (~10 min)
python eval.py --experiment experiments/policy_1931_2step.yaml
```

## B.4. Pull eval.json files + plot (local)

```bash
# From local box
for exp in policy_1931_base policy_1931_naive policy_1931_2step; do
  scp REMOTE_USER@REMOTE_IP:~/policy_pred_data/experiments/$exp/eval.json \
      "D:/hist_LLM/policy_pred/experiments/$exp/eval.json"
done

# Generate the comparison plot
python scripts/plot_policy_compare.py \
    --policy social_security_1935 \
    --base D:/hist_LLM/policy_pred/experiments/policy_1931_base/eval.json \
    --naive D:/hist_LLM/policy_pred/experiments/policy_1931_naive/eval.json \
    --twostep D:/hist_LLM/policy_pred/experiments/policy_1931_2step/eval.json \
    --out figures/ss1935_naive_vs_2step.png
```

The plot has 6 bars (3 models × 2 modes [yes_no, likert5]) showing P(SS implemented).

---

# Stop the box

Don't forget. Lambda / RunPod don't auto-stop billing when you close the SSH session.

---

## Troubleshooting

**`midtraining requires CUDA, got device=cpu`** — torch fell back to CPU build. `pip uninstall torch && pip install torch --index-url https://download.pytorch.org/whl/cu121`.

**OOM during training** — drop `--batch-size 2` to `1`. If still OOM, also drop `--seq-len 1024`. Grad checkpointing is on by default.

**`talkie.model` ImportError** — submodule not initialized. `git submodule update --init --recursive`.

**HF download is slow** — set `HF_HUB_ENABLE_HF_TRANSFER=1` and `pip install hf_transfer` before re-running `hf download`.

**`accelerate launch` hangs at startup** — port conflict with a previous run. `pkill -f "python train.py"` and retry.

**Multi-GPU saves only one process's adapter** — that's by design. Only rank 0 (main process) writes the checkpoint; other ranks call `accelerator.wait_for_everyone()` and exit. The adapter at `$POLICY_PRED_DATA_ROOT/experiments/<name>/checkpoint/` is the canonical output.

**`scaled_dot_product_attention() got an unexpected keyword argument 'enable_gqa'`** — the pod's torch is too old. The nanochat vendor (`flash_attention.py`) uses `enable_gqa`, added in **torch 2.5**. Some pod images ship torch 2.4.x. Fix: `pip install 'torch==2.5.1' --index-url https://download.pytorch.org/whl/cu124` (match your CUDA; the driver already supports it). Verify with `python -c "import torch; print(torch.__version__)"` → ≥2.5.

**Window matrix: CUDA OOM with `JOBS_PER_GPU=2`** — for the rolling-window scripts (`train_window_matrix.sh`), a single nanochat cell at X=5k uses ~25–53 GB, so two per 80 GB GPU OOM. Use **`JOBS_PER_GPU=1`** (the default). Parallelism comes from multiple GPUs, not packing one. Note: at X=5k a roll10 cell is ~50k records → ~20 min/cell; the full matrix at higher X is much heavier — prefer a 4-GPU box and the smaller X levels first.
