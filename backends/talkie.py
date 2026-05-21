"""Talkie-LM loader and scoring (wrapper around vendored model code).

Loader is memory-efficient: streams the fp32 ckpt via mmap and casts to bf16
on the fly into a pre-allocated bf16 model. Steady peak ~28 GB instead of
the vendored loader's ~100 GB intermediate spike (which is fine on a GPU box
but uncomfortable on a 128 GB workstation with other apps running).

`score_continuations` re-implements TalkieModel.forward without the last-token
slice so we get [B, T, V] logits for scoring multi-token continuations. The
vendored package at models/talkie_vendor/ stays unmodified.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F


# Make the vendored package importable without requiring `pip install -e`.
# This file lives at backends/talkie.py, so the repo root is one level up.
_VENDOR_SRC = Path(__file__).parent.parent / "models" / "talkie_vendor" / "src"
if _VENDOR_SRC.exists() and str(_VENDOR_SRC) not in sys.path:
    sys.path.insert(0, str(_VENDOR_SRC))


# Names of nn.Linear modules within each TalkieModel transformer block that
# LoRA should attach to. Architecture-specific — lives here, not in train.py,
# because every backend has different module names.
LORA_TARGET_MODULES = [
    "attn_query", "attn_key", "attn_value", "attn_resid",
    "mlp_gate", "mlp_linear", "mlp_resid",
]


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_ckpt_streamed(weights_dir: Path, device: torch.device):
    """Load Talkie's fp32 ckpt into a bf16 model with low peak memory.

    The vendored loader (talkie.model.load_checkpoint) materialises the full
    fp32 state_dict alongside the model and peaks near ~100 GB on the 13B
    weights. Here we use torch.load(mmap=True) so the state_dict is a view
    into the file (paged in on access) and copy each tensor into a
    pre-allocated bf16 model. Steady peak ~28 GB.
    """
    from talkie.model import GPTConfig, TalkieModel

    ckpt_path = weights_dir / "final.ckpt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Talkie checkpoint not found at {ckpt_path}")

    raw = torch.load(ckpt_path, map_location="cpu", mmap=True)
    if isinstance(raw, dict) and "model_state_dict" in raw:
        state_dict = raw["model_state_dict"]
    elif isinstance(raw, dict) and "model" in raw:
        state_dict = raw["model"]
    else:
        state_dict = raw
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    vocab_size = state_dict["embed.weight"].shape[0]
    config = GPTConfig(vocab_size=vocab_size)

    cpu = torch.device("cpu")
    model = TalkieModel(config, cpu).to(dtype=torch.bfloat16)

    with torch.no_grad():
        for name, param in model.named_parameters():
            if name not in state_dict:
                raise KeyError(f"missing key in checkpoint: {name}")
            param.copy_(state_dict[name].to(torch.bfloat16))

    del raw, state_dict

    if device.type != "cpu":
        model = model.to(device)
        model.device = device

    model.eval()
    return model


class TalkieBackend:
    # Mirror the module-level constant on the class so the dispatcher /
    # train.py can read `backend.LORA_TARGET_MODULES` without a separate import.
    LORA_TARGET_MODULES = LORA_TARGET_MODULES

    def __init__(self, weights_dir):
        self.weights_dir = Path(weights_dir)
        self._model = None
        self._tokenizer = None
        self._device = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from talkie.tokenizer import build_tokenizer

        self._device = _pick_device()
        self._tokenizer = build_tokenizer(
            self.weights_dir / "vocab.txt", style="base"
        )
        self._model = _load_ckpt_streamed(self.weights_dir, self._device)

    def prepare_for_peft(self) -> None:
        """Make Talkie's GPTConfig dict-like enough for peft >= 0.10.

        peft's inject_adapter does `model_config.get("tie_word_embeddings")`
        and `"_name_or_path" in model_config`. Talkie's GPTConfig is a plain
        dataclass without either method; this attaches the missing surface.
        Idempotent and safe to call multiple times.
        """
        self._ensure_loaded()
        cfg = self._model.config
        if not hasattr(cfg, "get"):
            cfg.get = lambda k, default=None: getattr(cfg, k, default)
        cls = type(cfg)
        if not hasattr(cls, "__contains__"):
            cls.__contains__ = lambda self, k: hasattr(self, k)

    def tokenize(self, text: str) -> list[int]:
        """Encode text to token IDs. Talkie's tokenizer is a raw tiktoken
        Encoding, so we forward `allowed_special="all"` to let <|endoftext|>
        and other specials encode as their special-token ids.
        """
        self._ensure_loaded()
        return self._tokenizer.encode(text, allowed_special="all")

    @property
    def eos_id(self) -> int:
        """<|endoftext|>'s token id under Talkie's vocab."""
        self._ensure_loaded()
        return self._tokenizer.encode("<|endoftext|>", allowed_special="all")[0]

    def _forward_all_positions(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward returning [B, T, V] logits.

        Mirrors TalkieModel.forward (talkie_vendor/src/talkie/model.py:184-196)
        but skips the last-token slice on the lm_head projection so every
        position gets a logit vector. Needed for scoring multi-token
        continuations and (later) training.
        """
        m = self._model
        _, seq_len = input_ids.shape
        cos_sin = m.cos[:, :seq_len], m.sin[:, :seq_len]

        x = m.embed(input_ids)
        x = F.rms_norm(x, (x.shape[-1],))
        e_x = x
        for block in m.blocks:
            x = block(e_x, x, cos_sin)
        x = F.rms_norm(x, (x.shape[-1],))
        return F.linear(x, m.lm_head_gain(m.lm_head)).float()

    @torch.no_grad()
    def score_continuations(
        self, prompt: str, continuations: Sequence[str]
    ) -> list[float]:
        """Length-normalized log-prob of each continuation given the prompt.

        Batches all continuations into a single padded forward pass. At
        batch=1 the GPU is memory-bandwidth bound (one full 26 GB weight
        read per pass to score a few dozen tokens), so batching multiple
        continuations through one read is nearly free — typical 5-7x
        speedup at the continuation counts used by the policy battery
        (2 for yes/no, 5 for likert). Right-padding is safe because
        Talkie is causal-only: padded tokens after the continuation
        can't influence log-probs at the continuation positions.
        """
        self._ensure_loaded()
        tok = self._tokenizer
        prompt_ids = tok.encode(prompt)
        prompt_len = len(prompt_ids)

        # Tokenize all continuations up front; reject any that vanish.
        cont_ids_list: list[list[int]] = []
        for cont in continuations:
            cont_ids = tok.encode(cont)
            if not cont_ids:
                raise ValueError(f"continuation tokenises to empty: {cont!r}")
            cont_ids_list.append(cont_ids)

        # Build a right-padded [B, T] batch of prompt+continuation sequences.
        seqs = [prompt_ids + ci for ci in cont_ids_list]
        max_len = max(len(s) for s in seqs)
        # Pad with 0 — irrelevant which token id we pick since right-padded
        # positions are never read (we only score positions inside the
        # continuation, which are strictly before any padding).
        padded = torch.zeros((len(seqs), max_len), dtype=torch.long,
                             device=self._device)
        for i, s in enumerate(seqs):
            padded[i, :len(s)] = torch.tensor(s, dtype=torch.long,
                                              device=self._device)

        logits = self._forward_all_positions(padded)  # [B, T, V]
        log_probs = F.log_softmax(logits, dim=-1)     # [B, T, V]

        results: list[float] = []
        for i, cont_ids in enumerate(cont_ids_list):
            # Position prompt_len+j in seq i is the j-th continuation
            # token, predicted by the logits at position prompt_len+j-1.
            total = 0.0
            for j, tok_id in enumerate(cont_ids):
                pos = prompt_len + j
                total += log_probs[i, pos - 1, tok_id].item()
            results.append(total / len(cont_ids))
        return results

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        stop_sequences: Sequence[str] = (),
    ) -> str:
        """Greedy decoding. Slow (no KV cache) — fine for eval, not for serving.

        Returns the generated continuation (NOT including the prompt). Stops
        on EOS, on max_new_tokens, or when any stop_sequences string appears
        in the decoded output.
        """
        self._ensure_loaded()
        tok = self._tokenizer
        eos_id = tok.encode("<|endoftext|>", allowed_special="all")[0]
        prompt_ids = tok.encode(prompt, allowed_special="all")
        x = torch.tensor([prompt_ids], dtype=torch.long, device=self._device)

        new_ids: list[int] = []
        for _ in range(max_new_tokens):
            logits = self._forward_all_positions(x)  # [1, T, V]
            next_id = int(logits[0, -1, :].argmax().item())
            if next_id == eos_id:
                break
            new_ids.append(next_id)
            x = torch.cat(
                [x, torch.tensor([[next_id]], dtype=torch.long, device=self._device)],
                dim=1,
            )
            if stop_sequences:
                # Decode only the newly-generated portion to check for stops.
                decoded = tok.decode(new_ids)
                if any(s in decoded for s in stop_sequences):
                    break
        return tok.decode(new_ids)
