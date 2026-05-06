"""GSM8K-style math word-problem generator.

Calls an OpenAI chat model with system + user prompts (in prompts/) to
produce grade-school math problems with step-by-step solutions, loosely
inspired by themes/numbers in the seed document. Output is chat-format
JSONL records compatible with HF chat templates.

Modes:
    sync (default): ThreadPoolExecutor, instant results. Good for testing
                    and small runs (<5K seeds).
    batch: OpenAI Batch API. ~50% cheaper, ~24h turnaround. Used for
           production runs at >5K seeds. Set use_batch_api: true in cfg.

Per-generator cfg keys (from experiment YAML synth.generators.math_word):
    n_per_seed: int           how many problems per seed doc (default 3)
    model: str                OpenAI model name (default "gpt-4o-mini")
    temperature: float        default 0.7
    max_tokens: int           default 1024
    use_batch_api: bool       default false (sync mode)
    api_key_env: str          env var name for API key (default OPENAI_API_KEY)
    seed: int                 deterministic shuffle seed (default 42)
    max_doc_chars: int        truncate seed text to this many chars (default 4000)
    max_workers: int          sync mode only (default 8)

OpenAI API key: read from env var (default OPENAI_API_KEY). NOT committed.
"""
from __future__ import annotations

import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable

from .base import Generator


class MathWordGenerator(Generator):
    name = "math_word"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.model = cfg.get("model", "gpt-4o-mini")
        self.temperature = cfg.get("temperature", 0.7)
        self.max_tokens = cfg.get("max_tokens", 1024)
        self.use_batch = cfg.get("use_batch_api", False)
        self.api_key_env = cfg.get("api_key_env", "OPENAI_API_KEY")
        self.rng_seed = cfg.get("seed", 42)
        self.max_doc_chars = cfg.get("max_doc_chars", 4000)
        self.max_workers = cfg.get("max_workers", 8)

        self.system_prompt = self.load_prompt("system")
        self.user_template = self.load_prompt("user")

    # -- Prompt assembly ---------------------------------------------------

    def _user_message(self, seed: dict) -> str:
        """Render the user-prompt template for one seed doc."""
        text = seed["text"]
        if len(text) > self.max_doc_chars:
            text = text[: self.max_doc_chars] + "\n[... truncated]"
        return self.user_template.format(
            document=text,
            n_per_seed=self.n_per_seed,
        )

    # -- OpenAI calls ------------------------------------------------------

    def _client(self):
        from openai import OpenAI
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"OpenAI API key not found in env var {self.api_key_env!r}. "
                f"Set it before running synth.py."
            )
        return OpenAI(api_key=api_key)

    def _call_one(self, client, seed: dict) -> list[dict]:
        """Synchronous call for one seed. Returns list of chat-format records."""
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": self._user_message(seed)},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            return self._parse(content, seed)
        except Exception as e:
            print(f"  warning: seed {seed.get('doc_id')} failed: {e}")
            return []

    @staticmethod
    def _parse(content: str, seed: dict) -> list[dict]:
        """Parse JSON response into chat-format training records."""
        try:
            obj = json.loads(content)
        except json.JSONDecodeError:
            return []
        # Accept either bare list or {"problems": [...]} or {"questions": [...]}.
        if isinstance(obj, list):
            items = obj
        elif isinstance(obj, dict):
            items = obj.get("problems") or obj.get("questions") or obj.get("data") or []
        else:
            return []

        out: list[dict] = []
        for item in items:
            q = item.get("question")
            a = item.get("answer")
            if not isinstance(q, str) or not isinstance(a, str):
                continue
            if "####" not in a:
                # Skip outputs missing the GSM8K final-answer marker.
                continue
            out.append({
                "messages": [
                    {"role": "user", "content": q.strip()},
                    {"role": "assistant", "content": a.strip()},
                ],
                "metadata": {
                    "generator": self.name,
                    "seed_doc_id": seed.get("doc_id"),
                    "seed_source": seed.get("source"),
                },
            })
        return out

    # -- Public entry point ------------------------------------------------

    def generate(self, seeds: Iterable[dict]) -> list[dict]:
        seeds = list(seeds)
        if self.use_batch:
            raise NotImplementedError(
                "Batch API path not yet implemented; pass use_batch_api: false "
                "in the experiment YAML for now."
            )
        return self._generate_sync(seeds)

    def _generate_sync(self, seeds: list[dict]) -> list[dict]:
        client = self._client()
        rng = random.Random(self.rng_seed)
        rng.shuffle(seeds)  # so partial runs cover diverse sources

        out: list[dict] = []
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = {ex.submit(self._call_one, client, s): s for s in seeds}
            for i, fut in enumerate(as_completed(futures), 1):
                out.extend(fut.result())
                if i % 50 == 0 or i == len(seeds):
                    elapsed = time.time() - t0
                    rate = i / elapsed if elapsed > 0 else 0
                    print(f"  [{i}/{len(seeds)}] {len(out)} records "
                          f"({rate:.1f} seeds/s)")
        return out
