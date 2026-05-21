You write one full synthetic document from a doc-idea record. Your output is continued-pretraining data for a language model studying historical period text.

# Rules

1. Write the document as a self-contained piece of period writing — no meta-commentary, no preamble, no postscript referencing the prompt.
2. Match the requested **doc_type** exactly. A court opinion reads like a court opinion; an editorial reads like an editorial; a private letter reads like a private letter.
3. Use the requested **tone**.
4. Use ONLY information present in the seed text or directly inferable from the seed text's era. Do NOT introduce events, people, places, technologies, or concepts that would not exist in the seed's period.
5. Length: 200-500 words for most doc types. Court opinions and policy memos may run longer.
6. Do NOT mention the seed text directly. The output should stand alone.
7. Do NOT include obvious anachronisms.

# Output format

Return a JSON object with one key, "document", whose value is a single string containing the full document text:

```json
{
  "document": "Full text of the document, as a single string with paragraph breaks via \\n\\n. No commentary outside the document itself."
}
```

Do NOT add commentary outside the JSON.
