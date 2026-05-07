You are brainstorming a diverse set of document concepts inspired by a seed text. Your output is the *ideas* stage of a two-stage synthetic-data pipeline — these ideas will later be expanded into full documents by another model call. Variety is the point.

# Rules

1. Read the seed text. Identify themes, vocabulary, characters, conflicts, institutions, places, and concerns visible in it.
2. Generate {n_ideas} doc-idea records. Each record specifies:
   - **doc_type**: pick from a wide range of period-appropriate forms — essay, opinion piece, news article, court opinion, dissent, legal brief, policy memorandum, congressional speech, lecture excerpt, sermon, editorial, letter to the editor, private letter, journal entry, business report, government statement, biographical sketch, book review, technical bulletin. Use a different doc_type for each idea where possible.
   - **concept**: one or two sentences describing what THIS specific document is about. Should be distinct from the seed (not a paraphrase of the seed). Inspired by the seed's themes but pursues a fresh angle.
   - **tone**: 1-3 adjectives describing the register (formal, polemical, sober, alarmist, scholarly, plainspoken, etc.).
3. Use ONLY information present in or directly inferable from the seed era. Do NOT introduce events, people, or concepts that the seed text's period would not know about.
4. Vary substantially across the {n_ideas} ideas. Different doc_types, different angles, different tones. The expansion stage will rely on each idea standing alone.

# Output format

Return a JSON object with one key, "ideas", whose value is an array of records:

```json
{
  "ideas": [
    {
      "doc_type": "court opinion",
      "concept": "An appeal in which a junior partner challenges a partnership accounting after the senior partner's death. Court must decide whether goodwill is divisible.",
      "tone": "formal, sober"
    },
    {
      "doc_type": "editorial",
      "concept": "A newspaper editorial questioning the wisdom of the proposed schedule of import duties on cotton manufactures, arguing protectionism harms downstream industries.",
      "tone": "polemical, plainspoken"
    },
    ...
  ]
}
```

Do NOT add commentary outside the JSON.
