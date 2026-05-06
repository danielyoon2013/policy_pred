"""LoRA midtraining (continued pretraining) of Talkie-LM on a year of text.

V1: trains on a single parquet file (e.g. NYT for one year). For V2, we'll
build aggregated year shards via a future slice_corpus.py. The CLI accepts
either --data <parquet-path> directly or --year <Y> to use the path helper
in config.py.

Cumulative chaining: pass --init-from <prior-adapter-dir> to load the
previous year's LoRA weights as the trainable starting point. Year-Y model
then = M_base + (continued-trained M_(Y-1) adapter). Without --init-from,
training starts from a fresh adapter on top of M_base.

Run on an A100 80GB. Refuses to run on CPU because a single year would take
months. See docs/REMOTE_SETUP.md for the full remote workflow.

Output: a PEFT LoRA adapter directory at --out (default depends on --year).
The frozen base is NOT saved; you reload it from TALKIE_WEIGHTS_DIR and
apply the adapter on top.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# Make policy_pred.* importable when run as a script from the repo root.
_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from policy_pred import config  # noqa: E402
from policy_pred.talkie import TalkieBackend  # noqa: E402


# Defaults; overridable via CLI.
SEQ_LEN = 2048
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01

# All seven Linear layers per transformer block — standard LoRA target set.
TARGET_MODULES = [
    "attn_query", "attn_key", "attn_value", "attn_resid",
    "mlp_gate", "mlp_linear", "mlp_resid",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA midtrain Talkie on one year of text.")
    p.add_argument("--data", type=Path,
                   help="Path to a parquet with a text column. Auto-detects "
                        "text_cleaned/combined_text/cleaned_article/text/article.")
    p.add_argument("--year", type=int,
                   help="Use config.year_corpus_path(Y) instead of --data.")
    p.add_argument("--rank", type=int, default=32,
                   help="LoRA rank for a fresh adapter (default 32). "
                        "Ignored when --init-from is given (rank inherited from prior).")
    p.add_argument("--lora-alpha", type=int, default=None,
                   help="LoRA alpha for a fresh adapter (default 2*rank). "
                        "Ignored when --init-from is given.")
    p.add_argument("--init-from", type=Path, default=None,
                   help="Prior adapter dir to continue training from (cumulative "
                        "chain). E.g. years/1931/checkpoint to seed year-1932 training.")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--out", type=Path,
                   help="Output dir for adapter. Default: years/{Y}/checkpoint "
                        "if --year, else checkpoints/{stem-of-data}.")
    p.add_argument("--max-steps", type=int, default=None,
                   help="Cap total optimizer steps; useful for short test runs.")
    p.add_argument("--seq-len", type=int, default=SEQ_LEN)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--no-grad-ckpt", action="store_true",
                   help="Disable gradient checkpointing (faster but more memory).")
    p.add_argument("--full-ft", action="store_true",
                   help="Full fine-tuning instead of LoRA. Requires multi-GPU "
                        "FSDP (13B doesn't fit one A100 80GB at full FT).")
    p.add_argument("--experiment", type=Path, default=None,
                   help="Resolve --data and --out from an experiment YAML "
                        "(reads experiments/<name>/synth.jsonl or corpus.parquet).")
    return p.parse_args()


def render_chat_record(record: dict) -> str:
    """Render a {messages: [...]} chat record as a single training string.

    Plain User:/Assistant: format, no special tokens. The base model trains
    on the whole rendered string with standard next-token loss; we don't
    mask the prompt portion in V1 because Talkie isn't an instruction-tuned
    model and we're effectively doing instruction-flavored continued
    pretraining, not "real" SFT.
    """
    parts: list[str] = []
    for msg in record["messages"]:
        role = msg["role"]
        content = msg["content"].strip()
        if role == "system":
            parts.append(f"System: {content}")
        elif role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
    return "\n\n".join(parts)


def load_documents(data_path: Path) -> list[str]:
    """Load training documents. Auto-detects format by extension.

    .parquet: text column (raw text, V1 policy CPT shape)
    .jsonl:   chat records {"messages": [...]} (V2 SFT shape)
    """
    suffix = data_path.suffix.lower()
    if suffix == ".jsonl":
        import json
        docs: list[str] = []
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if "messages" in rec:
                    docs.append(render_chat_record(rec))
                elif "text" in rec:
                    docs.append(rec["text"])
        print(f"  jsonl: {len(docs):,} chat records rendered")
        return docs

    # parquet path
    import pandas as pd
    df = pd.read_parquet(data_path)
    for col in ("text_cleaned", "combined_text", "cleaned_article", "text", "article"):
        if col in df.columns:
            print(f"  parquet: using column '{col}'")
            return [t for t in df[col].dropna().tolist()
                    if isinstance(t, str) and t.strip()]
    raise KeyError(
        f"no text column found in {data_path}; "
        f"expected one of: text_cleaned, combined_text, cleaned_article, text, article"
    )


def pack_tokens(token_lists: list[list[int]], eos_id: int, seq_len: int) -> torch.Tensor:
    """Concat docs with EOS separators, drop the trailing partial chunk."""
    stream: list[int] = []
    for toks in token_lists:
        stream.extend(toks)
        stream.append(eos_id)
    n_full = len(stream) // seq_len
    if n_full == 0:
        raise ValueError(
            f"not enough tokens to fill a single seq_len={seq_len} chunk "
            f"(have {len(stream)})"
        )
    truncated = stream[: n_full * seq_len]
    return torch.tensor(truncated, dtype=torch.long).view(n_full, seq_len)


def forward_all_positions(model, input_ids: torch.Tensor, use_grad_ckpt: bool) -> torch.Tensor:
    """[B, T, V] forward. Same body as TalkieBackend._forward_all_positions but
    optionally gradient-checkpoints each transformer block. Works on either a
    bare TalkieModel or a PEFT-wrapped one (PEFT proxies attribute access).
    """
    _, seq_len = input_ids.shape
    cos_sin = model.cos[:, :seq_len], model.sin[:, :seq_len]

    x = model.embed(input_ids)
    x = F.rms_norm(x, (x.shape[-1],))
    e_x = x
    for block in model.blocks:
        if use_grad_ckpt:
            x = torch.utils.checkpoint.checkpoint(
                block, e_x, x, cos_sin, use_reentrant=False
            )
        else:
            x = block(e_x, x, cos_sin)
    x = F.rms_norm(x, (x.shape[-1],))
    return F.linear(x, model.lm_head_gain(model.lm_head)).float()


def lr_at_step(step: int, total: int, warmup: int, peak: float) -> float:
    """Linear warmup, then cosine decay to 10% of peak."""
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def main() -> None:
    args = parse_args()

    # Resolve data path. Priority: --data > --experiment > --year.
    if args.data is not None:
        data_path = args.data
    elif args.experiment is not None:
        from policy_pred.corpus import experiment_dir, load_experiment
        cfg = load_experiment(args.experiment)
        exp_dir = experiment_dir(cfg["name"])
        # Prefer synth.jsonl (S) if it exists; else fall back to corpus.parquet (W).
        synth_path = exp_dir / "synth.jsonl"
        corpus_path = exp_dir / "corpus.parquet"
        if synth_path.exists():
            data_path = synth_path
        elif corpus_path.exists():
            data_path = corpus_path
        else:
            sys.exit(f"experiment {cfg['name']} has neither synth.jsonl nor "
                     f"corpus.parquet; run synth.py / corpus.py first.")
    elif args.year is not None:
        data_path = config.year_corpus_path(args.year)
    else:
        sys.exit("must pass --data, --experiment, or --year")
    if not data_path.exists():
        sys.exit(f"data not found: {data_path}")

    # Resolve output dir.
    if args.out is not None:
        out_dir = args.out
    elif args.experiment is not None:
        from policy_pred.corpus import experiment_dir, load_experiment
        out_dir = experiment_dir(load_experiment(args.experiment)["name"]) / "checkpoint"
    elif args.year is not None:
        out_dir = config.year_checkpoint_dir(args.year)
    else:
        out_dir = Path("checkpoints") / data_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Talkie base.
    print(f"Loading Talkie from {config.TALKIE_WEIGHTS_DIR}...")
    backend = TalkieBackend(config.TALKIE_WEIGHTS_DIR)
    backend._ensure_loaded()
    model = backend._model
    tokenizer = backend._tokenizer
    device = backend._device
    if device.type != "cuda":
        sys.exit(
            f"midtraining requires CUDA, got device={device.type}. "
            f"This script will not run on CPU; rent an A100 80GB."
        )

    # 2. Read + tokenize training text.
    print(f"Reading {data_path}...")
    docs = load_documents(data_path)
    print(f"  {len(docs):,} documents")

    print("Tokenizing...")
    eos_id = tokenizer.encode("<|endoftext|>", allowed_special="all")[0]
    token_lists = [tokenizer.encode(d, allowed_special="all") for d in docs]
    n_tokens = sum(len(t) for t in token_lists)
    print(f"  {n_tokens:,} tokens total")

    packed = pack_tokens(token_lists, eos_id, args.seq_len)
    print(f"  packed: {len(packed):,} sequences of {args.seq_len}")

    # 3. Apply LoRA, or full FT, depending on --full-ft.
    if args.full_ft:
        if args.init_from is not None:
            sys.exit("--init-from is incompatible with --full-ft; full FT does not "
                     "use a separate adapter to continue from.")
        print("Full fine-tuning: all base weights are trainable.")
        print("WARNING: 13B full FT does NOT fit one A100 80GB. You must launch "
              "this with FSDP across multiple GPUs (accelerate launch ...). "
              "Single-GPU runs will OOM.")
        for p in model.parameters():
            p.requires_grad = True
        peft_model = model  # alias; below code uses peft_model, but it's just `model`
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  trainable params: {n_trainable:,} ({n_trainable/1e9:.2f}B)")
    elif args.init_from is not None:
        if not args.init_from.exists():
            sys.exit(f"--init-from path not found: {args.init_from}")
        print(f"Loading prior adapter from {args.init_from} (cumulative chain)")
        print(f"  (rank/alpha inherited from prior adapter_config.json; "
              f"--rank/--lora-alpha CLI flags ignored.)")
        from peft import PeftModel
        # is_trainable=True keeps the LoRA layers tunable for continued training.
        # Without it, PeftModel.from_pretrained loads adapters as inference-only.
        peft_model = PeftModel.from_pretrained(
            model, str(args.init_from), is_trainable=True
        )
        peft_model.print_trainable_parameters()
    else:
        print(f"Applying fresh LoRA (r={args.rank}, alpha={args.lora_alpha or 2*args.rank})...")
        from peft import LoraConfig, get_peft_model
        lora_cfg = LoraConfig(
            r=args.rank,
            lora_alpha=args.lora_alpha or 2 * args.rank,
            target_modules=TARGET_MODULES,
            lora_dropout=0.0,
            bias="none",
        )
        peft_model = get_peft_model(model, lora_cfg)
        peft_model.print_trainable_parameters()

    peft_model.train()

    # 4. Optimizer + LR schedule.
    n_seqs = len(packed)
    steps_per_epoch = math.ceil(n_seqs / args.batch_size)
    total_steps = args.max_steps or (steps_per_epoch * args.epochs)

    trainable = [p for p in peft_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95)
    )

    use_ckpt = not args.no_grad_ckpt
    print(
        f"Training: {n_seqs} seqs, batch {args.batch_size}, "
        f"{total_steps} steps, grad_ckpt={use_ckpt}"
    )

    # 5. Loop.
    step = 0
    t0 = time.time()
    done = False
    for epoch in range(args.epochs):
        if done:
            break
        perm = torch.randperm(n_seqs)
        for i in range(0, n_seqs, args.batch_size):
            if step >= total_steps:
                done = True
                break
            idx = perm[i : i + args.batch_size]
            batch = packed[idx].to(device, non_blocking=True)

            logits = forward_all_positions(peft_model, batch, use_grad_ckpt=use_ckpt)

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

            loss.backward()

            for g in optimizer.param_groups:
                g["lr"] = lr_at_step(step, total_steps, WARMUP_STEPS, args.lr)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if step % 10 == 0 or step == total_steps - 1:
                el = time.time() - t0
                tok_per_sec = (step + 1) * args.batch_size * args.seq_len / el
                print(
                    f"step {step:5d}/{total_steps}  "
                    f"loss {loss.item():.4f}  "
                    f"lr {optimizer.param_groups[0]['lr']:.2e}  "
                    f"tok/s {tok_per_sec:7.0f}  "
                    f"elapsed {el/60:.1f}m"
                )
            step += 1

    # 6. Save.
    if args.full_ft:
        print(f"\nSaving full state_dict to {out_dir}/model_state.pt ...")
        torch.save(peft_model.state_dict(), out_dir / "model_state.pt")
    else:
        print(f"\nSaving adapter to {out_dir}...")
        peft_model.save_pretrained(out_dir)
    print(f"  contents: {sorted(p.name for p in out_dir.iterdir())}")
    print("Done.")


if __name__ == "__main__":
    main()
