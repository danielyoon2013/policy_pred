You build an evaluation-format teaching set from historical legal texts. You will be given a single legal passage from the year {year}, plus a FRAMING instruction.

Do the following:

1. Identify a normative policy question of the form "Should ...?" that the passage bears on — debatable by a contemporary of {year}, answerable from the passage's own reasoning (not requiring outside facts).
2. Apply the FRAMING:
   - frame = affirm: phrase the question so the passage's own reasoning supports answering it YES / agreeing.
   - frame = oppose: phrase the question as the OPPOSITE of what the passage concludes, so the passage's reasoning supports answering it NO / disagreeing (e.g. if the passage upholds a duty, ask "Should parties be free from that duty?").
3. Give the answer the passage's stance actually implies for the question as framed — honestly. Do NOT hedge toward neutrality; if the passage takes a clear side, take it. The Likert STRENGTH should reflect how strongly the passage commits (a firm holding → "Strongly agree"/"Strongly disagree"; a close or qualified one → "Agree"/"Disagree"; genuinely balanced → "Uncertain").
4. Give a one-sentence rationale grounded in the passage.

Scales:
- yesno: exactly "Yes" or "No".
- likert: exactly one of "Strongly agree", "Agree", "Uncertain", "Disagree", "Strongly disagree" (agreement with the "Should ...?" question as framed).

Rules:
- Era-appropriate framing for {year}. Do NOT name or allude to any specific legislation, court case, agency, or event from after {year}.
- Keep the question a general normative principle, not a recital of the passage's facts.
- "Uncertain" only when the passage genuinely takes no side — do not default to it.
- Output a single JSON object and nothing else: {"question": "...", "yesno": "Yes|No", "likert": "...", "rationale": "..."}
