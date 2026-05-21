# synth/sft_format

Generates **format-teaching SFT pairs** for the policy probe.

This is the third synth package, parallel to `synth/naive` and
`synth/2step` but with a fundamentally different output schema and
purpose:

| Package | Output | Used for |
|---|---|---|
| `synth/naive` | `{"text": "..."}` legal-prose synth | Policy CPT (cumulative chain) |
| `synth/2step` | same, two-stage variant | Policy CPT (more diverse) |
| `synth/sft_format` | `{"messages": [user, assistant]}` chat-format | Policy **SFT** (format teaching) |

## What it does

Produces 10K SFT pairs in the **exact prompt shape the policy battery
evaluator uses at eval time**:

- **5K yes/no pairs** — balanced 50/50 between target answer " Yes" and
  " No".
- **5K likert pairs** — balanced 20% each across {Strongly agree, Agree,
  Uncertain, Disagree, Strongly disagree}.

The prompts come from `prompts/yesno_template.md` and
`prompts/likert_template.md`, which steer gpt-4o-mini to produce
questions whose **natural** answer matches a target label. The output
labels are then attached deterministically (we know which label we asked
for), so the resulting dataset has perfect label balance by construction.

Topic diversity comes entirely from gpt-4o-mini at temperature 0.9 — we
don't seed with explicit topics. The prompts steer the model to "safe"
generic topics (geography, weather, food, generic ethics, abstract
education, technology trade-offs) and **explicitly avoid named U.S.
policies**, so there is no overlap with the 211 events in the policy
benchmark CSV.

## Why this matters

The policy probe asks the model:
```
Should X happen?

Answer:
```
…and reads the log-probabilities of `[" Yes", " No"]` (or the 5 likert
options). A base model trained on long-form legal prose has weak signal
on these probes because it was never taught to answer in this 1-token-ish
format. The SFT pass on this data teaches the model **the shape of the
answer** without teaching it **what to think about any specific policy**
— that's the methodological line we're walking.

## Usage

```bash
cd synth/sft_format
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...

# Full run (10K pairs, ~5-10 min, ~$1-2 on gpt-4o-mini)
python run.py

# Smoke run (200 pairs total, < 1 min, < $0.10)
python run.py --n-yesno 100 --n-likert 100
```

Idempotent: skips generation if the output file exists. Pass `--force` to
regenerate.

## Output schema (one line per record)

```json
{
  "messages": [
    {"role": "user", "content": "Should children learn to read?\n\nAnswer:"},
    {"role": "assistant", "content": " Yes"}
  ],
  "metadata": {"target_label": "Yes", "generator": "sft_format"}
}
```

The `messages` field is what `policy_pred/train.py --sft` consumes
directly via `render_chat_record` — the train loop tokenizes the user
content, masks its tokens out of the loss, and trains only on the
assistant content. This is the standard prompt-masked-loss SFT pattern.

## Why labels are " Yes" / " No" with a leading space

The policy battery evaluator scores continuations `" Yes"` and `" No"`
(see `evaluators/policy_battery.py:_build_yesno_prompt` —
`f"{question}\n\nAnswer:"`). The leading space matters: the model is
predicting the *next token* after the colon, and the tokenizer
encodes `" Yes"` and `"Yes"` as different tokens. Training with the
leading space ensures the SFT-shifted distribution lines up with what
the eval-time probe reads.

Same applies to the likert labels.

## Cost & timing

| Config | Time | Cost |
|---|---|---|
| 10K pairs, max_workers=16, batch_size=10 | ~5-10 min | ~$1-2 |
| 200-pair smoke (100/100) | < 1 min | < $0.10 |

The bottleneck is OpenAI's rate limit. gpt-4o-mini tier-2 allows 5K
RPM — we're well under that at max_workers=16.
