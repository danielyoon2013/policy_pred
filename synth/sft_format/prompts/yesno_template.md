You are generating training data for a language model. Each example
teaches the model to answer simple Yes/No normative questions in a
specific format.

Your task: produce {n_pairs} (question, answer) pairs where the natural
answer is exactly "{target_label}". The questions should:

1. Be of the form "Should X?" — normative, not factual.
2. Be on **diverse generic topics** (not about specific U.S. laws or
   named historical legislation). Examples of safe topics: everyday
   safety, geography, weather, food preferences, sports, generic ethics,
   household decisions, technology habits, abstract education questions.
3. **Have a clear natural answer of "{target_label}"** — either the
   subject is broadly good and uncontroversial (target=Yes) or broadly
   bad/dangerous/silly (target=No). Avoid ambiguous policy questions.
4. Be paraphrased — vary syntax and vocabulary. Don't repeat the same
   sentence structure.
5. Each question must be a single sentence ending with "?".

Output format: a JSON array of {n_pairs} objects, each with a single
"question" field. NO answer field in the JSON — we'll attach the label
"{target_label}" later.

Example output for target_label="Yes":
```json
[
  {"question": "Should children learn to read before middle school?"},
  {"question": "Should drivers stop at red traffic lights?"},
  {"question": "Should pedestrians look both ways before crossing?"}
]
```

Example output for target_label="No":
```json
[
  {"question": "Should employees ignore fire alarms during evacuations?"},
  {"question": "Should bicyclists ride against the flow of traffic?"},
  {"question": "Should food handlers skip washing their hands?"}
]
```

Generate {n_pairs} fresh, diverse questions for target_label="{target_label}".
Be creative — vary topics, sentence length, framing. Output ONLY the JSON
array, no preamble or explanation.
