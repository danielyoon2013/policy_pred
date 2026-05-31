You build an evaluation-format teaching set from historical legal texts. You are given one legal passage from the year {year}.

Your job: turn the SPECIFIC dispute in the passage into the BROAD normative policy question it instantiates, and answer it as a thoughtful contemporary of {year} would, judging only from the era's own reasoning.

Steps:
1. Read the passage and identify the general PRINCIPLE it bears on — the kind of broad "Should ...?" policy question a legislature or informed public of {year} might debate. State the question at that POLICY level, NOT about the specific parties, facts, or procedure.
   - Narrow (DO NOT do this): "Should the burden of proof for contributory negligence lie with the defendant?"
   - Broad principle (DO this): "Should the law make it easier for injured people to recover from those who harm them?"
   Target the altitude of these examples:
     - "Should the law hold businesses responsible for harms their operations cause to others?"
     - "Should courts enforce clearly written contracts over later claims of unfairness?"
     - "Should the state strictly punish the sale of prohibited goods?"
     - "Should government regulate the terms on which credit is extended to ordinary people?"
2. Answer whether the era's reasoning, as reflected in the passage, supports the principle:
   - yesno: exactly "Yes" or "No".
   - likert: agreement with the question, with STRENGTH graded by how SETTLED the principle was in the era's reasoning:
       firmly / near-universally settled  -> "Strongly agree" or "Strongly disagree"
       generally accepted but debatable    -> "Agree" or "Disagree"
       genuinely contested or unsettled    -> "Uncertain"
3. Give a one-sentence rationale grounded in the passage.

Rules:
- Phrase the question in ONE natural direction (the way the principle is normally stated). Do NOT negate it artificially or ask the opposite.
- Era-appropriate for {year}. Do NOT name or allude to any specific statute, act, amendment, court case, agency, program, or named event (of ANY year). Keep it a general principle a person could debate from first principles.
- Derive everything from the passage's own reasoning — no outside facts and no later (post-{year}) knowledge.
- Output a single JSON object and nothing else: {"question": "...", "yesno": "Yes|No", "likert": "...", "rationale": "..."}
