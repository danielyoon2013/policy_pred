# Safe seed topics for SFT-format question generation

A pool of generic, non-CSV-overlapping topics used to seed format-teaching
question generation. The goal is **diversity**, not coverage — the LLM
will produce ~10 paraphrased questions per topic, so we get ~1K topics ×
~10 paraphrases = ~10K examples.

CRITICAL: none of these topics should overlap with the 211 U.S. policy
events in `us_policy_event_battery_v4.csv`. Stay away from anything that
could match an `event_id` in that file (no "Social Security", no "Glass-
Steagall", no specific U.S. statutes by name, etc.).

Topics span: everyday life, geography, weather, science basics, common
sense, personal preferences, generic foreign policy (non-U.S.-statute),
hypothetical scenarios, household decisions, transportation, food, sports,
etiquette, education abstractions, environmental ethics, technology
trade-offs, etc.

## Topic categories with examples

### Everyday safety / common sense
- wearing helmets while cycling
- looking both ways before crossing
- locking the front door at night
- washing hands before meals
- driving slowly in school zones

### Geography / weather (mundane facts)
- whether rivers flow downhill
- whether the sky is generally blue at noon
- whether rain falls down rather than up
- whether mountains are taller than valleys
- whether deserts are dry by definition

### Common preferences / cuisine
- whether pizza should have pineapple
- whether coffee tastes bitter
- whether ice cream melts in the sun
- whether soup should be served hot
- whether pasta needs sauce

### Generic ethics / hypotheticals (non-U.S.-policy)
- whether children should learn to read
- whether dogs make good companions
- whether honesty is generally a virtue
- whether kindness should be encouraged
- whether sleep is necessary for health

### Environmental abstractions (non-U.S.-statute)
- whether trees produce oxygen
- whether plastic pollution is harmful to oceans
- whether energy conservation is wise in general
- whether endangered species deserve protection
- whether clean water access is important globally

### Foreign / international generic (NOT U.S. policy)
- whether the United Nations should promote cooperation
- whether literacy should be encouraged worldwide
- whether vaccination programs help in low-income countries
- whether multilingual education benefits travelers
- whether international trade reduces global poverty in general

### Technology trade-offs (generic, not U.S.-specific)
- whether smartphones can distract drivers
- whether social media affects sleep patterns
- whether passwords should be unique per account
- whether software updates should be installed promptly
- whether two-factor authentication improves account security

### Sports / entertainment / arts
- whether warm-up exercises reduce injury risk
- whether teamwork helps in soccer
- whether classical music can be relaxing
- whether plot twists make stories more memorable
- whether stretching after a workout is helpful

### Education abstractions (no U.S. statute)
- whether reading expands vocabulary
- whether practice improves skill
- whether sleep deprivation hurts test performance
- whether learning languages broadens perspective
- whether mentorship benefits new employees

### Anti-cases (questions where the natural answer is "No"/"Disagree")
- whether children should drive cars on highways
- whether employees should ignore fire alarms
- whether food preparation should skip handwashing
- whether bicyclists should ride against traffic
- whether bridges should be built without inspection

(The generator will mix these and add 10x more variants by paraphrasing.)
