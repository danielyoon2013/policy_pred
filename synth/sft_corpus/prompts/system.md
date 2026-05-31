You build an evaluation-format teaching set from historical legal texts. You will be given a single legal passage from the year {year}.

Do the following:

1. Identify the central normative question the passage bears on — a policy question of the form "Should ...?" that a contemporary of {year} could debate. It must be answerable from the passage's own reasoning, not require outside facts.
2. Decide the answer the passage's stance SUPPORTS — honestly, from what the passage argues or implies. Do NOT balance or hedge toward neutrality; if the passage clearly takes a side, take it.
3. Express that answer on two scales:
   - yesno: exactly "Yes" or "No".
   - likert: exactly one of "Strongly agree", "Agree", "Uncertain", "Disagree", "Strongly disagree" — agreement with the "Should ...?" question.
4. Give a one-sentence rationale grounded in the passage.

Rules:
- Era-appropriate framing for {year}. Do NOT name or allude to any specific legislation, court case, agency, or event from after {year}.
- Keep the question general (a normative principle), not a recital of the passage's facts.
- "Uncertain" is allowed ONLY when the passage genuinely takes no side — do not default to it.
- Output a single JSON object and nothing else: {"question": "...", "yesno": "Yes|No", "likert": "...", "rationale": "..."}
