from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


YEARS = list(range(1931, 2020, 4))
RELS = list(range(-20, 21))

TAXONOMY = {
    "Civil Rights": ["civil", "lgbtq", "gender", "native", "disability"],
    "Civil Liberties": ["speech", "religion", "privacy", "morals", "separation"],
    "Democracy & Governance": [
        "democracy",
        "campaign",
        "federalism",
        "state",
        "administrative",
        "executive",
        "transparency",
        "public",
        "crisis",
        "regional",
        "communications",
    ],
    "Criminal Justice & Guns": ["criminal", "guns", "drug"],
    "Immigration": ["immigration"],
    "Labor": ["labor"],
    "Economy & Finance": [
        "finance",
        "macroeconomic",
        "consumer",
        "banking",
        "credit",
        "tax",
        "trade",
        "industrial",
        "deregulation",
        "spending",
        "property",
    ],
    "Health/Welfare/Educ": ["health", "education", "welfare", "poverty", "social", "veterans", "housing"],
    "Environment & Energy": ["environment", "climate", "energy", "infrastructure", "rural", "utilities", "science"],
    "Natl Security & Foreign": ["security", "foreign", "war", "military", "nuclear", "wartime", "international"],
}
BASE2CAT = {base: cat for cat, bases in TAXONOMY.items() for base in bases}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--continuous-root", type=Path, required=True)
    p.add_argument("--control-root", type=Path)
    p.add_argument("--battery", type=Path, required=True)
    p.add_argument("--step-map", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--years", help="Comma-separated model years. Overrides --year-start/--year-end.")
    p.add_argument("--year-start", type=int)
    p.add_argument("--year-end", type=int)
    p.add_argument("--rel-min", type=int, default=-20)
    p.add_argument("--rel-max", type=int, default=20)
    p.add_argument("--run-label", default="continuous D23")
    return p.parse_args()


def resolve_years(args: argparse.Namespace) -> list[int]:
    if args.years:
        return [int(x.strip()) for x in args.years.split(",") if x.strip()]
    if args.year_start is not None or args.year_end is not None:
        if args.year_start is None or args.year_end is None:
            raise SystemExit("--year-start and --year-end must be provided together")
        return list(range(args.year_start, args.year_end + 1))
    return YEARS


def load_policy_categories(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            base = str(row["domain"]).split("_")[0]
            out[row["event_id"]] = BASE2CAT.get(base, "UNMAPPED")
    unmapped = sorted({p for p, cat in out.items() if cat == "UNMAPPED"})
    if unmapped:
        raise SystemExit(f"unmapped policy domains for {len(unmapped)} policies")
    return out


def load_step_map(path: Path) -> dict[int, int]:
    out = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[int(row["year"])] = int(row["step"])
    return out


def read_eval_rows(root: Path, run: str, pid2cat: dict[str, str], step_map: dict[int, int] | None, years: list[int]) -> list[dict]:
    rows: list[dict] = []
    exp_root = root / "experiments"
    for year in years:
        name = f"policy_{year}_nanochat_roll10_n10000"
        ev = exp_root / name / "eval.json"
        if not ev.exists():
            continue
        data = json.loads(ev.read_text(encoding="utf-8"))
        for pol in data.get("results", {}).get("policies", []):
            enact = pol.get("implementation_year")
            if enact is None:
                continue
            rel = year - int(enact)
            scores = pol.get("scores", {})
            yn = scores.get("yes_no", {})
            lk = scores.get("likert5", {})
            p_yes = yn.get("mean_p_yes", yn.get("p_yes"))
            likert = lk.get("mean_score", lk.get("score"))
            if p_yes is None:
                continue
            rows.append(
                {
                    "run": run,
                    "year_model": year,
                    "base_step": step_map.get(year, "") if step_map else "",
                    "policy_id": pol["id"],
                    "enactment_year": int(enact),
                    "rel_year": rel,
                    "p_yes": float(p_yes),
                    "likert": "" if likert is None else float(likert),
                    "category": pid2cat.get(pol["id"], "UNKNOWN"),
                    "n_variants": yn.get("n_variants", ""),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["run", "year_model", "base_step", "policy_id", "enactment_year", "rel_year", "p_yes", "likert", "category", "n_variants"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def raw_curve(rows: list[dict], rels: list[int]) -> list[float]:
    by_rel = defaultdict(list)
    for r in rows:
        by_rel[int(r["rel_year"])].append(float(r["p_yes"]))
    return [st.mean(by_rel[k]) if by_rel.get(k) else math.nan for k in rels]


def z_curve(rows: list[dict], rels: list[int]) -> tuple[list[float], int]:
    by_policy = defaultdict(list)
    for r in rows:
        by_policy[r["policy_id"]].append(r)
    by_rel = defaultdict(list)
    kept = 0
    for prs in by_policy.values():
        vals = [float(r["p_yes"]) for r in prs]
        if len(vals) < 5:
            continue
        sd = st.pstdev(vals)
        if sd == 0:
            continue
        kept += 1
        m = st.mean(vals)
        for r in prs:
            by_rel[int(r["rel_year"])].append((float(r["p_yes"]) - m) / sd)
    return [st.mean(by_rel[k]) if by_rel.get(k) else math.nan for k in rels], kept


def plot_curves(continuous: list[dict], control: list[dict], out_dir: Path, rels: list[int], years: list[int], run_label: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    c_raw = raw_curve(continuous, rels)
    k_raw = raw_curve(control, rels)
    c_z, c_n = z_curve(continuous, rels)
    k_z, k_n = z_curve(control, rels)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    axes[0].plot(rels, c_raw, "-o", ms=3, label=run_label)
    if control:
        axes[0].plot(rels, k_raw, "-o", ms=3, label="old nanochat control")
    axes[0].set_ylabel("mean P(yes)")
    axes[0].set_title("Raw policy belief trajectory")
    axes[1].plot(rels, c_z, "-o", ms=3, label=f"{run_label} (n={c_n})")
    if control:
        axes[1].plot(rels, k_z, "-o", ms=3, label=f"old nanochat control (n={k_n})")
    axes[1].set_ylabel("mean per-policy z-score of P(yes)")
    axes[1].set_title("Per-policy z-normalized trajectory")
    for ax in axes:
        ax.axvline(0, color="k", lw=1, ls="--", alpha=0.6, label="enactment")
        ax.axhline(0, color="gray", lw=0.8, alpha=0.35)
        ax.set_xlabel("year_model - enactment_year")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "continuous_vs_control_nshape.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    axes[0].plot(rels, c_raw, "-o", ms=3, color="#1f77b4")
    axes[0].set_ylabel("mean P(yes)")
    axes[0].set_title(f"{run_label}: raw")
    axes[1].plot(rels, c_z, "-o", ms=3, color="#1f77b4")
    axes[1].set_ylabel("mean per-policy z-score of P(yes)")
    axes[1].set_title(f"{run_label}: per-policy z-normalized (n={c_n})")
    for ax in axes:
        ax.axvline(0, color="k", lw=1, ls="--", alpha=0.6)
        ax.axhline(0, color="gray", lw=0.8, alpha=0.35)
        ax.set_xlabel("year_model - enactment_year")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "continuous_d23_raw_and_znorm.png", dpi=140)
    plt.close(fig)

    rel10 = [r for r in rels if -10 <= r <= 10]
    if rel10:
        c10 = [c_z[rels.index(r)] for r in rel10]
        mean_p = st.mean(float(r["p_yes"]) for r in continuous if -10 <= int(r["rel_year"]) <= 10) if continuous else math.nan
        policies = len({r["policy_id"] for r in continuous})
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(rel10, c10, "-o", ms=5, label=f"{run_label} LoRA (P(yes)={mean_p:.2f})")
        ax.axvline(0, color="k", lw=1, ls="--", alpha=0.6)
        ax.axhline(0, color="gray", lw=0.8, alpha=0.45)
        ax.set_xlabel("year_model - enactment_year")
        ax.set_ylabel("mean per-policy z-score of P(yes)")
        ax.set_title(
            f"{run_label} @ X=10k: per-policy z-normalized trajectory\n"
            f"LoRA policy adapters ({min(years)}-{max(years)}, {policies} policies)"
        )
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left")
        fig.tight_layout()
        fig.savefig(out_dir / "continuous_d23_all_year_rel10_nshape.png", dpi=140)
        plt.close(fig)

    return {"continuous_z_policies": c_n, "control_z_policies": k_n}


def write_readme(out_dir: Path, continuous: list[dict], control: list[dict], stats: dict, years_list: list[int], rel_min: int, rel_max: int) -> None:
    years = ", ".join(str(y) for y in years_list)
    c_policies = len({r["policy_id"] for r in continuous})
    k_policies = len({r["policy_id"] for r in control})
    c_rel = Counter(int(r["rel_year"]) for r in continuous)
    k_rel = Counter(int(r["rel_year"]) for r in control)
    control_note = (
        "The old nanochat control was regenerated on the same `[-20,+20]` window."
        if control
        else "The old nanochat control was not regenerated in this tidy pass; `control_nanochat_rel20_tidy.csv` is present as an empty placeholder."
    )
    text = f"""# Continuous D23 Policy Prediction: rel [{rel_min:+d},{rel_max:+d}]

## Summary

This run evaluates policy-belief trajectories for continual D23 point-in-time base models. For each selected model year, the base model is frozen and a LoRA adapter is trained on roll10 caselaw-derived policy synthetic data. The 211-policy battery is evaluated with direct `yes_no` and `likert5` probes over `rel_year = year_model - enactment_year` in `[{rel_min:+d},{rel_max:+d}]`.

## Model Years

Model years: {years}

## Outputs

- `continuous_d23_rel20_tidy.csv`
- `control_nanochat_rel20_tidy.csv`
- `figures/continuous_vs_control_nshape.png`
- `figures/continuous_d23_raw_and_znorm.png`
- `figures/continuous_d23_all_year_rel10_nshape.png`

## Counts

- Continuous rows: {len(continuous):,}; policies: {c_policies}; z-normalized policies kept: {stats['continuous_z_policies']}
- Control rows: {len(control):,}; policies: {k_policies}; z-normalized policies kept: {stats['control_z_policies']}
- Continuous rel range: {min(c_rel) if c_rel else 'NA'} to {max(c_rel) if c_rel else 'NA'}
- Control rel range: {min(k_rel) if k_rel else 'NA'} to {max(k_rel) if k_rel else 'NA'}

## Control Status

{control_note}

## Normalization

For each run, each policy's observed trajectory is normalized as `z = (p_yes - mean_policy) / std_policy`. Policies with fewer than 5 observed points or zero standard deviation are excluded from z-normalized curves. Raw plots average `p_yes` directly by relative year.

## Storage

Only LoRA adapter weights are saved for this run. Full continuous base checkpoints are not duplicated; they are referenced by year and checkpoint step.

## Professor-Facing Summary

We tested whether the new continual point-in-time models recover the same policy-belief trajectory seen in the earlier nanochat policy experiments. For each selected year-model, we froze the continuous D23 base checkpoint, trained a small roll10 policy LoRA on caselaw-derived synthetic policy data, and evaluated direct yes/no policy beliefs over a symmetric enactment window. The figures compare raw belief levels and within-policy normalized trajectory shape against the old nanochat control when available.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    years = resolve_years(args)
    rels = list(range(args.rel_min, args.rel_max + 1))
    pid2cat = load_policy_categories(args.battery)
    step_map = load_step_map(args.step_map)
    continuous = read_eval_rows(args.continuous_root, "continuous_d23", pid2cat, step_map, years)
    control = []
    if args.control_root:
        control = read_eval_rows(args.control_root, "old_nanochat_control", pid2cat, None, years)
    bad = [r for r in continuous + control if not (args.rel_min <= int(r["rel_year"]) <= args.rel_max)]
    if bad:
        raise SystemExit(f"{len(bad)} rows outside rel [{args.rel_min:+d},{args.rel_max:+d}]")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "continuous_d23_rel20_tidy.csv", continuous)
    write_csv(args.out_dir / "control_nanochat_rel20_tidy.csv", control)
    stats = plot_curves(continuous, control, args.out_dir / "figures", rels, years, args.run_label)
    write_readme(args.out_dir, continuous, control, stats, years, args.rel_min, args.rel_max)
    print(f"wrote {args.out_dir}")
    print(f"continuous rows={len(continuous)} control rows={len(control)}")


if __name__ == "__main__":
    main()
