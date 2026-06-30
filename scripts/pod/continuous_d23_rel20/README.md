# Continuous D23 rel20 run helpers

This folder records the RunPod orchestration used for the continuous D23 policy-prediction run completed on 2026-06-27.

Core policy code lives in the repository root:

- `train.py`
- `eval.py`
- `windowing.py`
- `evaluators/policy_battery_variants.py`
- `backends/nanochat.py`

Run-specific helpers:

- `remote_policy_run.sh`: remote driver for smoke, sparse-grid, and arbitrary-year waves.
- `stage_policy_wave_bases.ps1`: stages selected D23 year checkpoints from local D: to a RunPod.
- `stage_policy_main_bases.ps1`: earlier staging helper for the sparse grid.
- `patch_flash_attention.py`: compatibility patch used on the pod when needed.
- `year_step_map.csv`: sparse-grid checkpoint map from the pilot phase.

Final all-year results are documented in:

- `docs/findings/2026-06/continuous_d23_all_year_rel20/`

Heavyweight files are intentionally not committed. They remain archived locally at:

- `D:\hist_LLM\policy_pred\continuous_d23_all_year_rel20\remote_policy_archive.tar`
- `D:\hist_LLM\policy_pred\continuous_d23_all_year_rel20\continuous_data\experiments\`

That archive contains 90 LoRA adapter checkpoints and 90 eval JSON files for model years 1931-2020.
