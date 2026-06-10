"""Build the S1/S2 LoRA pools from caselaw + discourse synth blocks (additive design).

  S1 (pools_swmgst): first 5k caselaw pool lines + 5k GST synth          = 10k
  S2 (pools_swmall): first 5k caselaw + 5k GST + ~3k Econ/FOMC synth      = ~13k

Taking the FIRST 5k of each year's caselaw pool means S1/S2 share their caselaw half
with Arm A's 5k ladder level (stable_order prefix), so the matched-volume comparison
(S1 vs caselaw-10k, S2 vs caselaw-13k) is exact. Records are deterministically
shuffled; there is NO resampling — the blocks ARE the pool (all distinct). Writes the
per-arm pool jsonl + a sibling manifest, ready for windowing.assemble (roll10).

Run AFTER the discourse synth batch has been parsed into
years/<Y>/naive_swmgst/synth.jsonl and years/<Y>/naive_swmecofomc/synth.jsonl.

Usage:
  python scripts/data_prep/build_swm_pools.py --start-year 1932 --end-year 2016
"""
from __future__ import annotations

import argparse
import json
import random
import zlib
from pathlib import Path

POOLS = Path("C:/tmp/policy_pred/pools")          # existing caselaw pools
YEARS = Path("C:/tmp/policy_pred/years")          # discourse synth lives here
OUT_S1 = Path("C:/tmp/policy_pred/pools_swmgst")  # Arm S1
OUT_S2 = Path("C:/tmp/policy_pred/pools_swmall")  # Arm S2


def read_lines(path: Path, limit: int | None = None) -> list[str]:
    if not path.exists():
        return []
    out: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(line.rstrip("\n"))
                if limit and len(out) >= limit:
                    break
    return out


def write_pool(records: list[str], out_path: Path, year: int) -> int:
    recs = list(records)
    random.Random(zlib.crc32(f"swm:{year}".encode())).shuffle(recs)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(recs) + ("\n" if recs else ""))
    man = {"year": year, "n_distinct": len(recs), "n_total": len(recs),
           "resampled": 0, "target": len(recs)}
    out_path.with_suffix(".manifest.json").write_text(
        json.dumps(man, indent=2), encoding="utf-8")
    return len(recs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start-year", type=int, required=True)
    ap.add_argument("--end-year", type=int, required=True)
    ap.add_argument("--caselaw-block", type=int, default=5000)
    args = ap.parse_args()

    print(f"{'year':>5} {'S1 case+gst':>12} {'S2 +eco/fomc':>13}")
    missing = []
    for y in range(args.start_year, args.end_year + 1):
        case = read_lines(POOLS / f"{y}.jsonl", limit=args.caselaw_block)
        gst = read_lines(YEARS / str(y) / "naive_swmgst" / "synth.jsonl")
        eco = read_lines(YEARS / str(y) / "naive_swmecofomc" / "synth.jsonl")
        n1 = write_pool(case + gst, OUT_S1 / f"{y}.jsonl", y) if (case and gst) else 0
        n2 = write_pool(case + gst + eco, OUT_S2 / f"{y}.jsonl", y) if (case and gst and eco) else 0
        if not (case and gst):
            missing.append((y, "S1", f"case={len(case)} gst={len(gst)}"))
        print(f"{y:>5} {n1:>12} {n2:>13}")
    if missing:
        print("\n  MISSING inputs (pool not written):")
        for m in missing:
            print("   ", m)


if __name__ == "__main__":
    main()
