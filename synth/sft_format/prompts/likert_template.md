You are generating training data for a language model. Each example
teaches the model to respond to normative questions on a 5-point Likert
agreement scale.

Your task: produce {n_pairs} (question, answer) pairs where the natural
answer is exactly "{target_label}". The 5 possible labels and what they
mean:

- **Strongly agree** — the proposition is broadly seen as very good /
  uncontroversially correct / essential.
  Example: "Should children have access to clean drinking water?"
- **Agree** — the proposition is generally seen as good but not extreme.
  Example: "Should libraries provide free books to local residents?"
- **Uncertain** — the proposition is genuinely debatable, depends on
  context, or has no obvious right answer.
  Example: "Should pizza be considered a national dish?"
- **Disagree** — the proposition is broadly seen as a bad idea but not
  extreme.
  Example: "Should employers skip optional safety drills?"
- **Strongly disagree** — the proposition is broadly seen as very bad,
  dangerous, or absurd.
  Example: "Should children operate heavy machinery?"

Each question should:

1. Be of the form "Should X?" — normative, not factual.
2. Be on **diverse generic topics** (not about specific U.S. laws or
   named historical legislation). Stay away from anything that could
   match a real U.S. policy by name.
3. **Have a clear natural answer of "{target_label}"** — pick subjects
   whose moral / sensible / debatable status maps cleanly to that level.
4. Be paraphrased — vary syntax, vocabulary, framing.
5. Be a single sentence ending with "?".

Output format: a JSON array of {n_pairs} objects, each with a single
"question" field. NO answer field — we'll attach "{target_label}" later.

Example output for target_label="Strongly agree":
```json
[
  {"question": "Should hospitals maintain sterile operating rooms?"},
  {"question": "Should building inspections be required before occupancy?"}
]
```

Example output for target_label="Uncertain":
```json
[
  {"question": "Should pets be allowed in office workplaces?"},
  {"question": "Should public art include political themes?"}
]
```

Generate {n_pairs} fresh, diverse questions for
target_label="{target_label}". Vary topics, sentence length, framing.
Output ONLY the JSON array, no preamble or explanation.
