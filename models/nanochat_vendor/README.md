# nanochat_vendor

Vendored copy of the nanochat training/inference framework.

## Provenance

Source: `C:/Users/danielyoon/Dropbox/hist_LLM/nanochat/` (local-only, no
git remote). Copied here so `policy_pred` is self-contained and clonable.

Only the `nanochat/` package and `tasks/` task definitions are vendored —
not the top-level scripts (`mid_train.py`, `chat_sft.py`, etc.), which
are not needed at policy_pred's level of abstraction. We use
`policy_pred/train.py` and `policy_pred/eval.py` instead.

## What's inside

- `nanochat/` — the model code: `gpt.py` (the GPT class), `tokenizer.py`
  (BPE), `checkpoint_manager.py` (load/save), `engine.py`, `optim.py`,
  etc.
- `tasks/` — benchmark task definitions (ARC, GSM8K, MMLU, etc.). Useful
  reference, but policy_pred has its own evaluators in `evaluators/` and
  doesn't import these.

## How policy_pred uses it

`backends/nanochat.py` adds `models/nanochat_vendor/` to `sys.path` and
imports `nanochat.checkpoint_manager.load_model`. The vendor tree is
treated as read-only with one exception — see "Local patches" below.

## Local patches

One minimal patch was applied to make this vendor tree usable for
pure-inference setups (no training):

- `nanochat/tokenizer.py`: moved `import rustbpe` from the module top to
  a lazy import inside `RustBPETokenizer.train_from_iterator`. rustbpe is
  a Rust library only needed for tokenizer *training*; inference works
  with just `tiktoken` (which loads from `tokenizer.pkl`). Without this
  patch the whole module fails to import on machines that don't have
  rustbpe installed.

If the vendor tree is overwritten via `cp -r` (see "Updating" below),
re-apply this patch.

## Updating

If `hist_LLM/nanochat/` gets new commits worth pulling in, the cheapest
update is a `cp -r` overwrite:

```bash
rm -rf models/nanochat_vendor/nanochat
cp -r ~/Dropbox/hist_LLM/nanochat/nanochat models/nanochat_vendor/
```

If `hist_LLM` ever gets a GitHub remote, replace this vendor copy with a
proper git submodule (mirroring how `models/talkie_vendor/` works) so we
get version pinning + history.
