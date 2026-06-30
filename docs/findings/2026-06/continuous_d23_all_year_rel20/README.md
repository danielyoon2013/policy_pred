# Continuous D23 Policy Prediction: rel [-20,+20]

## Summary

This run evaluates policy-belief trajectories for continual D23 point-in-time base models. For each selected model year, the base model is frozen and a LoRA adapter is trained on roll10 caselaw-derived policy synthetic data. The 211-policy battery is evaluated with direct `yes_no` and `likert5` probes over `rel_year = year_model - enactment_year` in `[-20,+20]`.

## Model Years

Model years: 1931, 1932, 1933, 1934, 1935, 1936, 1937, 1938, 1939, 1940, 1941, 1942, 1943, 1944, 1945, 1946, 1947, 1948, 1949, 1950, 1951, 1952, 1953, 1954, 1955, 1956, 1957, 1958, 1959, 1960, 1961, 1962, 1963, 1964, 1965, 1966, 1967, 1968, 1969, 1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020

## Outputs

- `continuous_d23_rel20_tidy.csv`
- `control_nanochat_rel20_tidy.csv`
- `figures/continuous_vs_control_nshape.png`
- `figures/continuous_d23_raw_and_znorm.png`
- `figures/continuous_d23_all_year_rel10_nshape.png`

## Counts

- Continuous rows: 7,541; policies: 211; z-normalized policies kept: 211
- Control rows: 0; policies: 0; z-normalized policies kept: 0
- Continuous rel range: -20 to 20
- Control rel range: NA to NA

## Control Status

The old nanochat control was not regenerated in this tidy pass; `control_nanochat_rel20_tidy.csv` is present as an empty placeholder.

## Normalization

For each run, each policy's observed trajectory is normalized as `z = (p_yes - mean_policy) / std_policy`. Policies with fewer than 5 observed points or zero standard deviation are excluded from z-normalized curves. Raw plots average `p_yes` directly by relative year.

## Storage

Only LoRA adapter weights are saved for this run. Full continuous base checkpoints are not duplicated; they are referenced by year and checkpoint step.

## Professor-Facing Summary

We tested whether the new continual point-in-time models recover the same policy-belief trajectory seen in the earlier nanochat policy experiments. For each selected year-model, we froze the continuous D23 base checkpoint, trained a small roll10 policy LoRA on caselaw-derived synthetic policy data, and evaluated direct yes/no policy beliefs over a symmetric enactment window. The figures compare raw belief levels and within-policy normalized trajectory shape against the old nanochat control when available.
