"""Nanochat backend (1900-1949 era base model).

Mirrors TalkieBackend's surface so train.py / eval.py treat both
identically through `load_backend(cfg)`. Set `model_type: nanochat` in an
experiment YAML to route here.

The nanochat source lives at `policy_pred/models/nanochat_vendor/` —
vendored by direct copy from `hist_LLM/nanochat/` since that repo has no
git remote we can submodule. Treat the vendor dir as read-only; any
customisations live in this module.

Weights expected at `config.NANOCHAT_WEIGHTS_DIR`, which must contain:
    <weights_dir>/
        tokenizer/tokenizer.pkl            <- BPE tokenizer
        base_checkpoints/d<depth>/
            model_<step>.pt
            meta_<step>.json

The directory layout above is what nanochat's `load_model("base", ...)`
expects. We set `NANOCHAT_BASE_DIR` env var to the weights_dir so its
internal `get_base_dir()` resolves correctly.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F


# Make the vendored nanochat package importable. The vendored tree is at
# models/nanochat_vendor/, with the importable `nanochat` package one level
# below. This file lives at backends/nanochat.py, so the repo root is one
# parent up.
_VENDOR_SRC = Path(__file__).parent.parent / "models" / "nanochat_vendor"
if _VENDOR_SRC.exists() and str(_VENDOR_SRC) not in sys.path:
    sys.path.insert(0, str(_VENDOR_SRC))


# Names of nn.Linear modules within each nanochat transformer block that
# LoRA should attach to. Found by reading gpt.py:
#   attn submodule:  c_q, c_k, c_v, c_proj
#   mlp submodule:   c_fc, c_proj
# Listing c_proj once is fine — peft matches by suffix, attaching to both
# attn.c_proj and mlp.c_proj.
LORA_TARGET_MODULES = ["c_q", "c_k", "c_v", "c_proj", "c_fc"]


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class NanochatBackend:
    LORA_TARGET_MODULES = LORA_TARGET_MODULES

    def __init__(self, weights_dir):
        self.weights_dir = Path(weights_dir)
        self._model = None
        self._tokenizer = None
        self._device = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        # nanochat's loader resolves the tokenizer + checkpoint dir via its
        # own get_base_dir(), which honors $NANOCHAT_BASE_DIR. Point it at
        # our weights_dir so it finds <weights_dir>/tokenizer/ and
        # <weights_dir>/base_checkpoints/.
        os.environ.setdefault("NANOCHAT_BASE_DIR", str(self.weights_dir))

        self._device = _pick_device()

        # Build the model + tokenizer via nanochat's standard loader.
        # phase="eval" puts the model in eval mode; we'll flip to train()
        # when peft attaches a trainable adapter on top.
        from nanochat.checkpoint_manager import load_model
        model, tokenizer, _meta = load_model("base", self._device, phase="eval")
        # Nanochat hardcodes its rotary cos/sin buffers as bf16 (gpt.py
        # asserts this in forward), but on CUDA the loaded state_dict
        # weights may stay as fp32 — causing a dtype mismatch in c_q/c_k
        # at the first linear op. Cast the whole model to bf16 so
        # weights + activations + cos/sin all agree. (On CPU, load_model
        # has already converted to float32 for stability, and we don't
        # want to break that path — only cast on CUDA.)
        if self._device.type == "cuda":
            model = model.to(torch.bfloat16)
        self._model = model
        self._tokenizer = tokenizer

    def prepare_for_peft(self) -> None:
        """Make nanochat's GPTConfig dict-like enough for peft >= 0.10.

        nanochat's GPTConfig is a plain @dataclass, same gotcha as Talkie's
        — peft.inject_adapter calls `model_config.get("tie_word_embeddings")`
        and `"_name_or_path" in model_config`. Add the missing surface.
        Idempotent.
        """
        self._ensure_loaded()
        cfg = self._model.config
        if not hasattr(cfg, "get"):
            cfg.get = lambda k, default=None: getattr(cfg, k, default)
        cls = type(cfg)
        if not hasattr(cls, "__contains__"):
            cls.__contains__ = lambda self, k: hasattr(self, k)

    @torch.no_grad()
    def score_continuations(
        self, prompt: str, continuations: Sequence[str]
    ) -> list[float]:
        """Length-normalized log-prob of each continuation given the prompt.

        Same batched approach as TalkieBackend: right-pad all
        prompt+continuation sequences into a single [B, T] forward pass.
        nanochat's `model(idx)` returns logits directly (no need for the
        Talkie-style `_forward_all_positions` helper).
        """
        self._ensure_loaded()
        tok = self._tokenizer
        prompt_ids = tok.encode(prompt)
        prompt_len = len(prompt_ids)

        cont_ids_list: list[list[int]] = []
        for cont in continuations:
            cont_ids = tok.encode(cont)
            if not cont_ids:
                raise ValueError(f"continuation tokenises to empty: {cont!r}")
            cont_ids_list.append(cont_ids)

        seqs = [prompt_ids + ci for ci in cont_ids_list]
        max_len = max(len(s) for s in seqs)
        padded = torch.zeros((len(seqs), max_len), dtype=torch.long,
                             device=self._device)
        for i, s in enumerate(seqs):
            padded[i, :len(s)] = torch.tensor(s, dtype=torch.long,
                                              device=self._device)

        logits = self._model(padded)               # [B, T, V]
        log_probs = F.log_softmax(logits, dim=-1)  # [B, T, V]

        results: list[float] = []
        for i, cont_ids in enumerate(cont_ids_list):
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
        """Greedy decoding. No KV cache — fine for eval, slow for serving."""
        self._ensure_loaded()
        tok = self._tokenizer
        # nanochat tokenizer exposes special tokens via its bos/eos lookup.
        # We use <|bos|> as a delimiter for end-of-sequence; the tokenizer's
        # encode_special is the safest way to get its id.
        prompt_ids = tok.encode(prompt)
        try:
            eos_id = tok.encode_special("<|bos|>")
        except Exception:
            eos_id = None

        x = torch.tensor([prompt_ids], dtype=torch.long, device=self._device)
        new_ids: list[int] = []
        for _ in range(max_new_tokens):
            logits = self._model(x)
            next_id = int(logits[0, -1, :].argmax().item())
            if eos_id is not None and next_id == eos_id:
                break
            new_ids.append(next_id)
            x = torch.cat(
                [x, torch.tensor([[next_id]], dtype=torch.long,
                                 device=self._device)],
                dim=1,
            )
            if stop_sequences:
                decoded = tok.decode(new_ids)
                if any(s in decoded for s in stop_sequences):
                    break
        return tok.decode(new_ids)
