"""Belief-elicitation prompt templates.

Goal: synthetic instructions that train a year-Y model to be a fluent 'year-Y mind' --
current events, prevailing economic / political / social views, ongoing debates of the
year -- WITHOUT naming or anticipating any policy in policies/catalog.yaml. Policy
leakage in synthesis invalidates the trajectory at eval time.

Design these only after a manual review of sample documents from a representative year
(e.g. 1935) so the templates reflect what the corpus can actually support.
"""

WORLDVIEW_TEMPLATES: list[str] = []
