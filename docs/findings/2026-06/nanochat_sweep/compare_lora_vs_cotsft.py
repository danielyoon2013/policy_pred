"""Compare nanochat LoRA-only vs LoRA+CoT-SFT at X=10k (1931-2020).

LoRA-only: yes_no P(yes), read bare (from nano_tidy.csv, X=10000).
CoT-SFT  : yes_no_cot P(yes), generate-reasoning-then-score (from the pulled _sftcot eval.jsons).
Both: per-policy z-normalized over each policy's +/-10yr window; aggregated by years-to-enactment.

Inputs : nano_tidy.csv (this dir); C:/tmp/cotsft_results/policy_*_n10000_sftcot/eval.json
Outputs: figures/compare_{nshape,domain}_lora_vs_cotsft.png
"""
import csv, json, glob, re, collections, statistics as st
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
COTSFT_DIR = "C:/tmp/cotsft_results"

TAXONOMY = {
    "Civil Rights": ["civil","lgbtq","gender","native","disability"],
    "Civil Liberties": ["speech","religion","privacy","morals","separation"],
    "Democracy & Governance": ["democracy","campaign","federalism","state","administrative","executive","transparency","public","crisis","regional","communications"],
    "Criminal Justice & Guns": ["criminal","guns","drug"],
    "Immigration": ["immigration"], "Labor": ["labor"],
    "Economy & Finance": ["finance","macroeconomic","consumer","banking","credit","tax","trade","industrial","deregulation","spending","property"],
    "Health/Welfare/Educ": ["health","education","welfare","poverty","social","veterans","housing"],
    "Environment & Energy": ["environment","climate","energy","infrastructure","rural","utilities","science"],
    "Natl Security & Foreign": ["security","foreign","war","military","nuclear","wartime","international"],
}
BASE2CAT = {b: c for c, bs in TAXONOMY.items() for b in bs}
bat = pd.read_csv(REPO / "us_policy_event_battery_v4.csv")
pid2cat = {str(r.event_id): BASE2CAT.get(str(r.domain).split("_")[0], "?") for r in bat.itertuples()}

# LoRA-only @ X=10000 from nano_tidy
lora = []
for r in csv.DictReader(open(HERE / "nano_tidy.csv", encoding="utf-8")):
    if r["X"] == "10000":
        lora.append((int(r["year_model"]), r["policy_id"], int(r["rel_year"]), float(r["p_yes"])))

# CoT-SFT @ X=10000 from pulled eval.jsons (scores.yes_no_cot)
cot = []
for d in glob.glob(COTSFT_DIR + "/policy_*_n10000_sftcot/eval.json"):
    ym = int(re.search(r"policy_(\d+)_", d.replace("\\", "/")).group(1))
    data = json.load(open(d, encoding="utf-8"))
    for p in data["results"]["policies"]:
        enact = p.get("implementation_year")
        sc = (p.get("scores", {}).get("yes_no_cot") or {}).get("mean_p_yes")
        if enact is None or sc is None: continue
        cot.append((ym, p["id"], ym - int(enact), sc))

RELS = list(range(-10, 11))
def per_policy_z(recs):  # recs: (ym, pid, rel, p)
    out = collections.defaultdict(list); bp = collections.defaultdict(list)
    for r in recs: bp[r[1]].append(r)
    for prs in bp.values():
        vs = [r[3] for r in prs]
        if len(vs) < 5: continue
        m, sd = st.mean(vs), st.pstdev(vs)
        if sd == 0: continue
        for r in prs: out[r[2]].append((r[3]-m)/sd)
    return out
def curve(z): return [st.mean(z[k]) if z.get(k) else float("nan") for k in RELS]
def preinc(z):
    pre = [st.mean(z[k]) for k in range(-10,1) if z.get(k)]
    return (pre[-1]-pre[0]) if len(pre) >= 2 else float("nan")

zL, zC = per_policy_z(lora), per_policy_z(cot)
print(f"LoRA-only: overall P(yes)={st.mean([r[3] for r in lora]):.3f}  pre-rise={preinc(zL):+.3f}z")
print(f"CoT-SFT  : overall P(yes)={st.mean([r[3] for r in cot]):.3f}  pre-rise={preinc(zC):+.3f}z")

# ---- FIG 1: n-shape comparison ----
fig, ax = plt.subplots(figsize=(9.5, 5.5))
ax.plot(RELS, curve(zL), "-o", ms=4, color="#1f77b4", label=f"LoRA-only (P(yes)={st.mean([r[3] for r in lora]):.2f})")
ax.plot(RELS, curve(zC), "-o", ms=4, color="#d62728", label=f"LoRA + CoT-SFT (P(yes)={st.mean([r[3] for r in cot]):.2f})")
ax.axvline(0, color="k", lw=1, ls="--", alpha=.6); ax.axhline(0, color="gray", lw=.8, alpha=.5)
ax.set_xlabel("year_model - enactment_year"); ax.set_ylabel("mean per-policy z-score of P(yes)")
ax.set_title("nanochat @ X=10k: per-policy z-normalized trajectory\nLoRA-only vs +CoT-SFT (1931-2020, 211 policies)")
ax.grid(alpha=.25); ax.legend()
fig.tight_layout(); fig.savefig(HERE / "figures" / "compare_nshape_lora_vs_cotsft.png", dpi=130); plt.close(fig)
print("wrote compare_nshape_lora_vs_cotsft.png")

# ---- FIG 2: per-domain pre-enactment z-rise, grouped ----
cats = list(TAXONOMY)
incL, incC, ns = [], [], []
for cat in cats:
    lr = [r for r in lora if pid2cat.get(r[1]) == cat]; cr = [r for r in cot if pid2cat.get(r[1]) == cat]
    incL.append(preinc(per_policy_z(lr))); incC.append(preinc(per_policy_z(cr)))
    ns.append(len({r[1] for r in lr}))
order = sorted(range(len(cats)), key=lambda i: incL[i], reverse=True)
cats=[cats[i] for i in order]; incL=[incL[i] for i in order]; incC=[incC[i] for i in order]; ns=[ns[i] for i in order]
import numpy as np
y = np.arange(len(cats)); h = 0.38
fig, ax = plt.subplots(figsize=(11, 6.5))
ax.barh(y+h/2, incL, h, color="#1f77b4", label="LoRA-only")
ax.barh(y-h/2, incC, h, color="#d62728", label="LoRA + CoT-SFT")
ax.set_yticks(y); ax.set_yticklabels([f"{c} (n={n})" for c,n in zip(cats,ns)], fontsize=9); ax.invert_yaxis()
ax.axvline(0, color="k", lw=1); ax.set_xlabel("pre-enactment z-rise (rel -10 -> 0), X=10k")
ax.set_title("nanochat @ X=10k: per-policy z-normalized pre-enactment rise by domain\nLoRA-only vs +CoT-SFT", fontsize=11)
ax.legend(); ax.grid(axis="x", alpha=.25); fig.tight_layout()
fig.savefig(HERE / "figures" / "compare_domain_lora_vs_cotsft.png", dpi=130); plt.close(fig)
print("wrote compare_domain_lora_vs_cotsft.png")

print("\nper-domain pre-enactment z-rise [LoRA | CoT-SFT]:")
for c, a, b, n in zip(cats, incL, incC, ns):
    print(f"  {c:<26} n={n:>2}  {a:+.2f} | {b:+.2f}")
