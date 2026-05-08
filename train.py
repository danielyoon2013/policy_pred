"""LoRA midtraining (continued pretraining) of Talkie-LM on a year of text.

Single-GPU: `python train.py --experiment <yaml> --data <path>`
Multi-GPU:  `accelerate launch --num_processes=N train.py --experiment ... --data ...`

The script uses HuggingFace `accelerate` to wrap the model, optimizer, and
dataloader. With one process accelerate is essentially transparent (same
behavior as the original single-GPU loop). With N processes, accelerate
sets up DDP: each rank loads its own copy of Talkie on its own GPU, the
DataLoader is sharded by DistributedSampler, and gradients all-reduce
after backward. Total wall clock scales ~linearly with N for LoRA (the
small adapter gradients all-reduce cheaply).

Effective batch in multi-GPU mode is (per-rank batch_size) * N. To keep
the same effective batch as a 1-GPU run, divide --batch-size by N. (Or
just keep --batch-size and accept the larger effective batch; for LoRA
fine-tuning the LR is forgiving.)

Cumulative chaining: pass --init-from <prior-adapter-dir> to load the
previous year's LoRA weights as the trainable starting point. Year-Y model
then = M_base + (continued-trained M_(Y-1) adapter). Without --init-from,
training starts from a fresh adapter on top of M_base.

Run on an A100 80GB or larger. Refuses to run on CPU because a single year
would take months. See docs/REMOTE_SETUP.md for the full remote workflow.

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
        # Resolution order:
        #   1. cfg.train.data (explicit external path; bypasses corpus/synth stages)
        #   2. exp_dir/synth.jsonl  (output of synth.py)
        #   3. exp_dir/corpus.parquet  (output of corpus.py if synth was skipped)
        train_data = cfg.get("train", {}).get("data")
        synth_path = exp_dir / "synth.jsonl"
        corpus_path = exp_dir / "corpus.parquet"
        if train_data is not None:
            data_path = Path(train_data)
        elif synth_path.exists():
            data_path = synth_path
        elif corpus_path.exists():
            data_path = corpus_path
        else:
            sys.exit(f"experiment {cfg['name']} has neither train.data, "
                     f"synth.jsonl, nor corpus.parquet. Either set "
                     f"train.data in the YAML or run synth.py / corpus.py first.")
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

    # 0. Initialize Accelerator (transparent for 1 GPU, handles DDP for N GPUs).
    from accelerate import Accelerator
    accelerator = Accelerator()
    is_main = accelerator.is_main_process
    n_proc = accelerator.num_processes

    def mp(*a, **kw):
        """Print only on the main process."""
        if is_main:
            print(*a, **kw)

    if n_proc > 1:
        mp(f"Multi-GPU run: {n_proc} ranks (DDP). "
           f"Per-rank batch={args.batch_size}, "
           f"effective batch={args.batch_size * n_proc}.")

    # 1. Load Talkie base. Each rank loads its own copy on its own GPU
    # (CUDA_VISIBLE_DEVICES is set per rank by `accelerate launch`).
    mp(f"Loading Talkie from {config.TALKIE_WEIGHTS_DIR}...")
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

    # 2. Read + tokenize training text. Done on every rank (cheap relative
    # to training). DataLoader will shard via DistributedSampler.
    mp(f"Reading {data_path}...")
    docs = load_documents(data_path)
    mp(f"  {len(docs):,} documents")

    mp("Tokenizing...")
    eos_id = tokenizer.encode("<|endoftext|>", allowed_special="all")[0]
    token_lists = [tokenizer.encode(d, allowed_special="all") for d in docs]
    n_tokens = sum(len(t) for t in token_lists)
    mp(f"  {n_tokens:,} tokens total")

    packed = pack_tokens(token_lists, eos_id, args.seq_len)
    mp(f"  packed: {len(packed):,} sequences of {args.seq_len}")

    # 3. Apply LoRA, or full FT, depending on --full-ft.
    if args.full_ft:
        if args.init_from is not None:
            sys.exit("--init-from is incompatible with --full-ft; full FT does not "
                     "use a separate adapter to continue from.")
        mp("Full fine-tuning: all base weights are trainable.")
        mp("NOTE: 13B full FT requires FSDP, not just DDP. This branch is "
           "lightly tested; for production full-FT runs use a dedicated FSDP "
           "config (e.g. via `accelerate config`).")
        for p in model.parameters():
            p.requires_grad = True
        peft_model = model
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        mp(f"  trainable params: {n_trainable:,} ({n_trainable/1e9:.2f}B)")
    elif args.init_from is not None:
        if not args.init_from.exists():
            sys.exit(f"--init-from path not found: {args.init_from}")
        mp(f"Loading prior adapter from {args.init_from} (cumulative chain)")
        mp(f"  (rank/alpha inherited from prior adapter_config.json; "
           f"--rank/--lora-alpha CLI flags ignored.)")
        from peft import PeftModel
        peft_model = PeftModel.from_pretrained(
            model, str(args.init_from), is_trainable=True
        )
        if is_main:
            peft_model.print_trainable_parameters()
    else:
        mp(f"Applying fresh LoRA (r={args.rank}, alpha={args.lora_alpha or 2*args.rank})...")
        from peft import LoraConfig, get_peft_model
        lora_cfg = LoraConfig(
            r=args.rank,
            lora_alpha=args.lora_alpha or 2 * args.rank,
            target_modules=TARGET_MODULES,
            lora_dropout=0.0,
            bias="none",
        )
        peft_model = get_peft_model(model, lora_cfg)
        if is_main:
            peft_model.print_trainable_parameters()

    peft_model.train()

    # 4. Build a DataLoader so accelerator.prepare can shard for DDP.
    from torch.utils.data import TensorDataset, DataLoader
    dataset = TensorDataset(packed)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # 5. Optimizer.
    trainable = [p for p in peft_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95)
    )

    # 6. Wrap with Accelerator. Single-GPU = transparent. Multi-GPU = DDP.
    peft_model, optimizer, dataloader = accelerator.prepare(
        peft_model, optimizer, dataloader
    )

    n_seqs = len(packed)
    # Each rank sees ~n_seqs/n_proc sequences per epoch via DistributedSampler.
    steps_per_epoch = math.ceil(n_seqs / (args.batch_size * n_proc))
    total_steps = args.max_steps or (steps_per_epoch * args.epochs)

    use_ckpt = not args.no_grad_ckpt
    mp(f"Training: {n_seqs} seqs total, per-rank batch={args.batch_size}, "
       f"effective batch={args.batch_size * n_proc}, "
       f"{total_steps} steps/rank, grad_ckpt={use_ckpt}")

    # 7. Loop.
    step = 0
    t0 = time.time()
    done = False
    for epoch in range(args.epochs):
        if done:
            break
        for batch_tuple in dataloader:
            if step >= total_steps:
                done = True
                break
            # TensorDataset wraps in tuple; accelerator.prepare moves to device.
            batch = batch_tuple[0]

            logits = forward_all_positions(peft_model, batch, use_grad_ckpt=use_ckpt)

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

            accelerator.backward(loss)

            for g in optimizer.param_groups:
                g["lr"] = lr_at_step(step, total_steps, WARMUP_STEPS, args.lr)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if step % 10 == 0 or step == total_steps - 1:
                el = time.time() - t0
                # Effective tok/s aggregates across all ranks.
                tok_per_sec = (step + 1) * args.batch_size * args.seq_len * n_proc / el
                mp(f"step {step:5d}/{total_steps}  "
                   f"loss {loss.item():.4f}  "
                   f"lr {optimizer.param_groups[0]['lr']:.2e}  "
                   f"tok/s {tok_per_sec:7.0f}  "
                   f"elapsed {el/60:.1f}m")
            step += 1

    # 8. Save (only main process; other ranks wait for it to finish).
    accelerator.wait_for_everyone()
    if is_main:
        unwrapped = accelerator.unwrap_model(peft_model)
        if args.full_ft:
            mp(f"\nSaving full state_dict to {out_dir}/model_state.pt ...")
            torch.save(unwrapped.state_dict(), out_dir / "model_state.pt")
        else:
            mp(f"\nSaving adapter to {out_dir}...")
            unwrapped.save_pretrained(out_dir)
        mp(f"  contents: {sorted(p.name for p in out_dir.iterdir())}")
    accelerator.wait_for_everyone()
    mp("Done.")


if __name__ == "__main__":
    main()
