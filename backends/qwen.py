"""Qwen backend (stub).

Modern HF model used as a methodological baseline. Because Qwen is a stock
transformers model, the implementation is the smallest of the three backends:
no peft compat shims needed (HF configs are dict-like), and tokenizer/model
load via AutoTokenizer + AutoModelForCausalLM.

Not yet implemented. When wiring this up:
- weights_dir points at config.QWEN_WEIGHTS_DIR (HF cache or local snapshot).
- AutoTokenizer.from_pretrained + AutoModelForCausalLM.from_pretrained(
      weights_dir, torch_dtype=torch.bfloat16
  ) handles loading.
- score_continuations / generate logic is identical to TalkieBackend (modulo
  using the HF tokenizer's encode and the model's forward(input_ids).logits).
- prepare_for_peft is a no-op (HF configs already satisfy peft's interface).
- LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"] is the standard
  Qwen target list — verify against the specific Qwen variant chosen.
"""
from __future__ import annotations

from pathlib import Path


class QwenBackend:
    LORA_TARGET_MODULES: list[str] = []  # TODO: populate when implementing

    def __init__(self, weights_dir: Path):
        raise NotImplementedError(
            "QwenBackend not yet implemented. See backends/qwen.py for "
            "integration steps — should be ~50 lines following the "
            "TalkieBackend pattern (minus the peft shims)."
        )
