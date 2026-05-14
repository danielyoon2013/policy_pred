"""Generate per-year policy_<Y>_naive.yaml files for the cumulative-chain training.

Copies the structure of an existing reference YAML (e.g. policy_1932_naive.yaml)
and substitutes the year-specific fields (`name`, `train.data`, and the comment
header). Idempotent — skips files that already exist unless --force is set.

Usage:
    # Generate 1936..1970 (35 new files) using 1932 as the template:
    python scripts/data_prep/make_year_yaml.py \\
        --template experiments/policy_1932_naive.yaml \\
        --years 1936-1970 \\
        --out-dir experiments

    # Override a single year:
    python scripts/data_prep/make_year_yaml.py --template ... --years 1942 --force
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def parse_years_spec(spec: str) -> list[int]:
    """Parse '1936-1970' or '1942' or '1936,1937,1940-1942' into a list of ints."""
    out: list[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if "-" in chunk:
            a, b = chunk.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(chunk))
    return sorted(set(out))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--template", type=Path, required=True,
                   help="Path to a reference YAML (e.g. experiments/policy_1932_naive.yaml).")
    p.add_argument("--years", required=True,
                   help="Years to generate: '1936-1970', '1942', '1936,1940-1942'.")
    p.add_argument("--out-dir", type=Path, default=Path("experiments"),
                   help="Output dir for the generated YAMLs.")
    p.add_argument("--name-template", default="policy_{year}_naive",
                   help="Format for the YAML's `name:` field. Default: policy_{year}_naive.")
    p.add_argument("--data-template",
                   default="C:/tmp/policy_pred/years/{year}/naive/synth.jsonl",
                   help="Format for the YAML's train.data path. Default: "
                        "C:/tmp/policy_pred/years/{year}/naive/synth.jsonl")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing YAML files.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.template.exists():
        sys.exit(f"template not found: {args.template}")

    template_text = args.template.read_text(encoding="utf-8")

    # Pull the reference year from the template's name: line.
    m = re.search(r"^name:\s+(\S+)", template_text, flags=re.MULTILINE)
    if not m:
        sys.exit(f"could not find `name:` line in template {args.template}")
    template_name = m.group(1)

    m = re.search(r"(19|20)\d{2}", template_name)
    if not m:
        sys.exit(f"could not parse a year out of template name {template_name!r}")
    template_year = m.group(0)
    print(f"Template: {args.template}  (year={template_year}, name={template_name})")

    years = parse_years_spec(args.years)
    print(f"Generating YAMLs for {len(years)} years: {years[0]}..{years[-1]}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    n_written = n_skipped = 0
    for y in years:
        out_path = args.out_dir / f"policy_{y}_naive.yaml"
        if out_path.exists() and not args.force:
            n_skipped += 1
            continue

        # Substitute year in the template text.
        text = template_text
        # Replace the template year in name + data path everywhere it appears
        # (it appears in `name:`, `train.data:`, comments).
        text = text.replace(template_year, str(y))

        # Ensure the name and data lines match the configured formats.
        new_name = args.name_template.format(year=y)
        new_data = args.data_template.format(year=y)
        text = re.sub(r"^name:.*$", f"name: {new_name}", text, flags=re.MULTILINE)
        text = re.sub(r"^(\s+data:)\s.*$",
                      lambda mm: f"{mm.group(1)} {new_data}",
                      text, flags=re.MULTILINE)

        out_path.write_text(text, encoding="utf-8")
        n_written += 1

    print()
    print(f"  wrote:    {n_written}")
    print(f"  skipped:  {n_skipped} (already existed; use --force to overwrite)")


if __name__ == "__main__":
    main()
