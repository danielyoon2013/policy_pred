"""Rolling-window training assembly: weights, naming, dataset assembly, YAML emit.

Round 2 replaces the cumulative LoRA chain with parallel rolling-window training:
each year-Y model is the frozen base + a LoRA trained on a *recency-weighted
window* of per-year synthetic pools, with no `--init-from` chain — so every year
trains independently and in parallel. This module is the single home for all of
that logic (kept in one file by design; no new scripts/data_prep/ files).

Four pieces:
  - WeightSpec / year_weights: which source years feed a given model-year, and in
    what proportion. Pluggable (rolling / exponential / cumulative / single) with
    a no-future-leakage guarantee.
  - allocate_counts: turn weights + a budget into an integer record count per year.
  - assemble: read the per-year pools, take a deterministic prefix from each,
    concatenate into one training jsonl, and write a reproducibility manifest.
  - experiment_name / write_window_yaml: the naming convention and the tiny
    experiment YAML that train.py / eval.py route on (since train.py has no
    --model-type flag, the YAML is the only reason a YAML is needed).

The pool files (built in Phase 1a) are authoritative and already ordered
distinct-records-first with any resample-with-replacement duplicates appended to
reach the uniform 40k. assemble therefore just takes the first-`count` lines of a
pool — prefixes are nested, so the {5k,10k,20k,40k} sweep levels are nested
subsets by construction. stable_order is the deterministic permutation the pool
builder uses to fix that order.

CLI (called by scripts/pod/train_window_matrix.sh):
    python -m windowing name     --year 1950 --backend nanochat --spec roll10 --per-year 40000
    python -m windowing assemble --year 1950 --spec roll10 --per-year 40000 --pool-dir ~/data/pools --out /tmp/x.jsonl
    python -m windowing yaml     --year 1950 --backend nanochat --spec roll10 --per-year 40000 --out-dir experiments/windows
"""
from __future__ import annotations

import argparse
import json
import random
import re
import zlib
from dataclasses import dataclass
from pathlib import Path

# LoRA / eval defaults baked into the emitted YAML — match the Round-1
# experiments/policy_<Y>_naive.yaml so the only deliberate change is the
# training data and the aggregation scheme.
_RANK = 32
_LORA_ALPHA = 64
_EPOCHS = 1
_LR = 1.0e-4
_BATCH_SIZE = 8
_SEQ_LEN = 2048
_EVAL_WINDOW_YEARS = 10
_CSV_BATTERY = "us_policy_event_battery_v4.csv"


@dataclass(frozen=True)
class WeightSpec:
    """A recency-weighting scheme, round-tripped through a compact token.

    The token (e.g. "roll10", "exp5", "cum", "single") is both the CLI value and
    the slug embedded in the experiment name, so it is the single serialized form.
    """
    kind: str            # "roll" | "exp" | "cum" | "single"
    param: float | None  # roll: window width W; exp: half-life H; else None
    token: str


def parse_spec(token: str) -> WeightSpec:
    """Parse a spec token. 'roll10' -> rolling width 10; 'exp5' -> half-life 5."""
    token = token.strip().lower()
    if token == "cum":
        return WeightSpec("cum", None, "cum")
    if token == "single":
        return WeightSpec("single", None, "single")
    m = re.fullmatch(r"(roll|exp)(\d+(?:\.\d+)?)", token)
    if not m:
        raise ValueError(
            f"bad spec {token!r}; expected roll<W> | exp<H> | cum | single")
    kind, num = m.group(1), float(m.group(2))
    if num <= 0:
        raise ValueError(f"spec param must be > 0, got {num}")
    return WeightSpec(kind, num, token)


def year_weights(model_year: int,
                 eligible_source_years: list[int],
                 spec: WeightSpec) -> dict[int, float]:
    """Normalized weight per source year feeding `model_year`.

    Guarantees: never includes a year > model_year (no future leakage); the
    window is clipped to the AVAILABLE eligible years (so an early model with a
    truncated window simply renormalizes over what exists); weights sum to 1 and
    only years with weight > 0 are returned.
    """
    eligible = sorted(y for y in eligible_source_years if y <= model_year)
    if not eligible:
        return {}

    if spec.kind == "single":
        chosen = [eligible[-1]]                       # model_year, or nearest <=
        raw = {chosen[0]: 1.0}
    elif spec.kind == "roll":
        width = int(spec.param)
        window = eligible[-width:]                    # the W most-recent available
        raw = {y: 1.0 for y in window}
    elif spec.kind == "cum":
        raw = {y: 1.0 for y in eligible}
    elif spec.kind == "exp":
        half_life = spec.param
        raw = {y: 0.5 ** ((model_year - y) / half_life) for y in eligible}
    else:  # pragma: no cover - parse_spec guards this
        raise ValueError(f"unknown spec kind {spec.kind!r}")

    total = sum(raw.values())
    return {y: w / total for y, w in raw.items()}


def allocate_counts(weights: dict[int, float],
                    pool_sizes: dict[int, int],
                    *,
                    total_n: int | None = None,
                    per_year: int | None = None) -> dict[int, int]:
    """Split a record budget across window years by weight (largest-remainder).

    Exactly one of `total_n` / `per_year` is given. per_year mode (the
    professor-decided framing) sets the budget to per_year * (number of window
    years), so a uniform rolling window yields exactly `per_year` records per
    year. Counts are clipped to pool_sizes; freed budget is NOT redistributed
    (pools are the uniform 40k so clipping only bites in degenerate cases, and
    redistribution would break the clean per-year-X interpretation).
    """
    if (total_n is None) == (per_year is None):
        raise ValueError("pass exactly one of total_n / per_year")
    years = sorted(weights)
    budget = total_n if total_n is not None else per_year * len(years)

    exact = {y: weights[y] * budget for y in years}
    floors = {y: int(exact[y]) for y in years}
    remainder = budget - sum(floors.values())
    # Hand out the leftover by largest fractional part; ties -> most recent year.
    order = sorted(years, key=lambda y: (exact[y] - floors[y], y), reverse=True)
    for y in order[:remainder]:
        floors[y] += 1
    # Clip to what each pool can supply.
    return {y: min(floors[y], pool_sizes.get(y, floors[y])) for y in years}


def stable_order(pool_year: int, n_records: int, seed: int = 0) -> list[int]:
    """Deterministic permutation of range(n_records), keyed on (seed, year) only.

    Keying on (seed, year) — and crucially NOT on n_records — is what makes the
    scaling levels nested: first-5k is a prefix of first-10k of first-20k, etc.
    Uses crc32 (stable across processes/Python builds), not the salted built-in
    hash().
    """
    base = zlib.crc32(f"{seed}:{pool_year}".encode())
    rng = random.Random(base)
    idx = list(range(n_records))
    rng.shuffle(idx)
    return idx


def build_pool(distinct_jsonl: Path, out_pool: Path, year: int, target: int,
               *, seed: int = 0) -> dict:
    """Build a uniform-size year pool from distinct synth records.

    Reads the distinct synth (one record per legal seed, output of
    synth/naive at n_per_seed=1), orders it deterministically via stable_order
    (distinct-records-first), then — per the professor-decided equalization —
    appends identical resample-WITH-REPLACEMENT duplicates until the pool
    reaches `target` (e.g. 40k). Rich years (>= target distinct) are truncated to
    the first `target`. Writes the pool jsonl + a sibling .manifest.json carrying
    `n_distinct` so downstream assembly can report effective_unique.

    Distinct-first ordering is what keeps the {5k,10k,20k,40k} sweep honest: any
    level <= n_distinct draws only distinct records; duplicates appear only in the
    high levels of sparse years.
    """
    lines = [ln.rstrip("\n") for ln in open(distinct_jsonl, encoding="utf-8")
             if ln.strip()]
    n_distinct = len(lines)
    order = stable_order(year, n_distinct, seed=seed)
    ordered = [lines[i] for i in order]

    if n_distinct >= target:
        pool = ordered[:target]
    else:
        # Resample-with-replacement to fill the gap, deterministically.
        rng = random.Random(zlib.crc32(f"{seed}:{year}:resample".encode()))
        pool = list(ordered)
        pool.extend(rng.choices(ordered, k=target - n_distinct))

    out_pool.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pool, "w", encoding="utf-8") as f:
        f.write("\n".join(pool) + ("\n" if pool else ""))
    manifest = {"year": year, "n_distinct": n_distinct, "n_total": len(pool),
                "target": target, "resampled": max(0, len(pool) - n_distinct),
                "seed": seed, "source": str(distinct_jsonl)}
    out_pool.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def experiment_name(year: int, backend: str, spec: WeightSpec,
                    per_year: int, sft: bool) -> str:
    """The single naming source: policy_<Y>_<backend>_<spec>_n<X>[_sft]."""
    name = f"policy_{year}_{backend}_{spec.token}_n{per_year}"
    return f"{name}_sft" if sft else name


def _read_jsonl(path: Path, limit: int) -> list[str]:
    """First `limit` raw lines of a jsonl pool (lines kept verbatim)."""
    out: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(line.rstrip("\n"))
                if len(out) >= limit:
                    break
    return out


def _pool_distinct_count(pool_path: Path) -> int | None:
    """Distinct-record count from a sibling <pool>.manifest.json, if present.

    Phase 1a writes that manifest when it resamples-with-replacement to reach the
    uniform target; it lets us report effective_unique per (year, level). Absent
    manifest → treat the whole pool as distinct.
    """
    man = pool_path.with_suffix(".manifest.json")
    if man.exists():
        try:
            return int(json.loads(man.read_text(encoding="utf-8"))["n_distinct"])
        except (KeyError, ValueError):
            return None
    return None


def assemble(model_year: int, spec: WeightSpec, pool_dir: Path, out_jsonl: Path,
             *, total_n: int | None = None, per_year: int | None = None,
             seed: int = 0) -> dict:
    """Build one model's training jsonl from the per-year pools; write a manifest.

    Reads pools at pool_dir/<Y>.jsonl (authoritative, already distinct-first +
    resampled). Takes the first allocated count from each window year, concatenates,
    shuffles the combined set (order doesn't affect which records are chosen, so
    nesting across sweep levels is preserved), writes out_jsonl and a manifest.
    """
    pools = {int(p.stem): p for p in pool_dir.glob("*.jsonl")
             if p.stem.isdigit()}
    eligible = [y for y in pools if y <= model_year]
    weights = year_weights(model_year, eligible, spec)
    if not weights:
        raise SystemExit(f"no eligible source years <= {model_year} in {pool_dir}")

    pool_sizes = {y: sum(1 for ln in open(pools[y], encoding="utf-8") if ln.strip())
                  for y in weights}
    counts = allocate_counts(weights, pool_sizes, total_n=total_n, per_year=per_year)

    records: list[str] = []
    per_year_manifest: dict[str, dict] = {}
    for y in sorted(counts):
        take = counts[y]
        lines = _read_jsonl(pools[y], take)
        records.extend(lines)
        distinct = _pool_distinct_count(pools[y])
        per_year_manifest[str(y)] = {
            "count": len(lines),
            "effective_unique": min(len(lines), distinct) if distinct else len(lines),
            "weight": round(weights[y], 6),
        }

    random.Random(zlib.crc32(f"{seed}:{model_year}:assemble".encode())).shuffle(records)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        f.write("\n".join(records) + ("\n" if records else ""))

    manifest = {
        "model_year": model_year,
        "spec": spec.token,
        "total_records": len(records),
        "effective_unique_total": sum(v["effective_unique"]
                                       for v in per_year_manifest.values()),
        "seed": seed,
        "per_year": per_year_manifest,
        "pool_dir": str(pool_dir),
        "out": str(out_jsonl),
    }
    out_jsonl.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def write_window_yaml(year: int, backend: str, spec: WeightSpec, per_year: int,
                      sft: bool, out_dir: Path, *,
                      modes: list[str] | None = None,
                      variants_per_policy: int | None = None,
                      chat_format: bool | None = None,
                      cot_max_new_tokens: int = 256,
                      rel_min: int | None = None,
                      rel_max: int | None = None) -> Path:
    """Emit the tiny experiment YAML train.py / eval.py route on (data left blank
    so the CLI --data, i.e. the assembled jsonl, wins).

    Eval knobs (read by policy_battery_variants.run):
      modes                which probes to score; default [yes_no, likert5].
                           Add yes_no_cot/likert5_cot for the reason-then-answer
                           probe (the model generates reasoning, then we score
                           the answer) — pair these with a CoT-SFT adapter.
      chat_format          wrap the direct probes in the User:/Assistant:
                           rendering the SFT models were trained on. Defaults to
                           `sft` (on for SFT-on-top, off for the LoRA-only
                           baseline read bare). CoT modes always chat-render.
      variants_per_policy  cap paraphrases/policy; defaults to 20 when any CoT
                           mode is present (generation is slow), else uncapped.
      cot_max_new_tokens   reasoning-generation budget for the CoT modes.
    """
    name = experiment_name(year, backend, spec, per_year, sft)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.yaml"

    modes = modes or ["yes_no", "likert5"]
    has_cot = any(m.endswith("_cot") for m in modes)
    if chat_format is None:
        chat_format = sft
    if variants_per_policy is None and has_cot:
        variants_per_policy = 20

    eval_lines = [
        "eval:",
        "  evaluator: policy_battery_variants",
        "  source: csv",
        f"  csv_path: {_CSV_BATTERY}",
    ]
    # Asymmetric rel-window (rel = model_year - E) takes precedence over the
    # symmetric window_years when given — e.g. rel_min=-10, rel_max=+20 for the
    # extended post-enactment trajectory.
    if rel_min is not None or rel_max is not None:
        if rel_min is not None:
            eval_lines.append(f"  rel_min: {int(rel_min)}")
        if rel_max is not None:
            eval_lines.append(f"  rel_max: {int(rel_max)}")
    else:
        eval_lines.append(f"  window_years: {_EVAL_WINDOW_YEARS}")
    eval_lines += [
        f"  modes: [{', '.join(modes)}]",
        f"  chat_format: {str(chat_format).lower()}",
    ]
    if variants_per_policy is not None:
        eval_lines.append(f"  variants_per_policy: {variants_per_policy}")
    if has_cot:
        eval_lines.append(f"  cot_max_new_tokens: {cot_max_new_tokens}")

    text = (
        f"# Auto-generated by windowing.py — rolling-window {spec.token}, "
        f"per-year={per_year}{', SFT-on-top' if sft else ''}.\n"
        f"name: {name}\n"
        f"model_type: {backend}\n"
        f"\n"
        f"train:\n"
        f"  method: lora\n"
        f"  rank: {_RANK}\n"
        f"  lora_alpha: {_LORA_ALPHA}\n"
        f"  epochs: {_EPOCHS}\n"
        f"  lr: {_LR}\n"
        f"  batch_size: {_BATCH_SIZE}\n"
        f"  seq_len: {_SEQ_LEN}\n"
        f"\n"
        + "\n".join(eval_lines) + "\n"
    )
    path.write_text(text, encoding="utf-8")
    return path


def _cli() -> None:
    p = argparse.ArgumentParser(description="rolling-window assembly utilities")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp):
        sp.add_argument("--year", type=int, required=True)
        sp.add_argument("--spec", required=True, help="roll10 | exp5 | cum | single")
        sp.add_argument("--per-year", type=int, default=None)
        sp.add_argument("--total-n", type=int, default=None)

    s = sub.add_parser("name")
    add_common(s)
    s.add_argument("--backend", required=True)
    s.add_argument("--sft", action="store_true")

    s = sub.add_parser("assemble")
    add_common(s)
    s.add_argument("--pool-dir", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--seed", type=int, default=0)

    s = sub.add_parser("yaml")
    add_common(s)
    s.add_argument("--backend", required=True)
    s.add_argument("--sft", action="store_true")
    s.add_argument("--out-dir", type=Path, default=Path("experiments/windows"))
    s.add_argument("--modes", default=None,
                   help="comma-separated eval modes (e.g. yes_no,likert5 or "
                        "yes_no_cot,likert5_cot). Default: yes_no,likert5.")
    s.add_argument("--variants-per-policy", type=int, default=None,
                   help="cap paraphrases/policy (default: 20 if a CoT mode is set, else uncapped).")
    s.add_argument("--chat-format", action=argparse.BooleanOptionalAction, default=None,
                   help="force chat-format eval on/off (default: on iff --sft).")
    s.add_argument("--cot-max-new-tokens", type=int, default=256)
    s.add_argument("--rel-min", type=int, default=None,
                   help="asymmetric window lower bound on rel=model_year-E (e.g. -10).")
    s.add_argument("--rel-max", type=int, default=None,
                   help="asymmetric window upper bound on rel=model_year-E (e.g. +20). "
                        "If either rel bound is set it overrides window_years.")

    s = sub.add_parser("build-pool")
    s.add_argument("--year", type=int, required=True)
    s.add_argument("--distinct", type=Path, required=True,
                   help="Distinct synth jsonl (synth/naive output, n_per_seed=1).")
    s.add_argument("--out", type=Path, required=True, help="Pool jsonl to write.")
    s.add_argument("--target", type=int, default=40000)
    s.add_argument("--seed", type=int, default=0)

    args = p.parse_args()
    if args.cmd == "build-pool":
        man = build_pool(args.distinct, args.out, args.year, args.target,
                         seed=args.seed)
        print(args.out)
        print(f"  {man['n_distinct']} distinct -> {man['n_total']} pool "
              f"({man['resampled']} resampled)")
        return

    spec = parse_spec(args.spec)

    if args.cmd == "name":
        print(experiment_name(args.year, args.backend, spec,
                              args.per_year or args.total_n, args.sft))
    elif args.cmd == "assemble":
        man = assemble(args.year, spec, args.pool_dir, args.out,
                       total_n=args.total_n, per_year=args.per_year, seed=args.seed)
        print(args.out)  # so the bash dispatcher can capture the data path
        print(f"  {man['total_records']} records "
              f"({man['effective_unique_total']} unique) from {len(man['per_year'])} years")
    elif args.cmd == "yaml":
        modes = [m.strip() for m in args.modes.split(",")] if args.modes else None
        print(write_window_yaml(args.year, args.backend, spec,
                                args.per_year or args.total_n, args.sft, args.out_dir,
                                modes=modes,
                                variants_per_policy=args.variants_per_policy,
                                chat_format=args.chat_format,
                                cot_max_new_tokens=args.cot_max_new_tokens,
                                rel_min=args.rel_min, rel_max=args.rel_max))


if __name__ == "__main__":
    _cli()
