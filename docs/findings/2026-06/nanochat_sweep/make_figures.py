"""Figures for the nanochat LoRA-only ladder sweep (1931-2020, X=5k/10k/20k).

All trajectories are PER-POLICY Z-NORMALIZED: each policy's P(yes) is centered and
scaled by its own mean/std over its +/-10yr window, so the SHAPE (rise toward
enactment) is visible despite different per-policy baselines. We show RAW z and
CALENDAR-DETRENDED z side by side: the calendar drift (later year-models rate every
policy higher; rel_year == year_model - const) inflates the raw rise, so detrended
is the honest signal.

Inputs : nano_tidy.csv (this dir), us_policy_event_battery_v4.csv (repo root)
Outputs: figures/{nshape_znorm,domain_znorm_smallmult,domain_increment_bar}.png
Run    : python make_figures.py
"""
import csv, collections, statistics as st
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]

# --- meaningful 10-category taxonomy: every domain base-token mapped (all 211) ---
TAXONOMY = {
    "Civil Rights":            ["civil", "lgbtq", "gender", "native", "disability"],
    "Civil Liberties":         ["speech", "religion", "privacy", "morals", "separation"],
    "Democracy & Governance":  ["democracy", "campaign", "federalism", "state", "administrative",
                                "executive", "transparency", "public", "crisis", "regional", "communications"],
    "Criminal Justice & Guns": ["criminal", "guns", "drug"],
    "Immigration":             ["immigration"],
    "Labor":                   ["labor"],
    "Economy & Finance":       ["finance", "macroeconomic", "consumer", "banking", "credit", "tax",
                                "trade", "industrial", "deregulation", "spending", "property"],
    "Health/Welfare/Educ":     ["health", "education", "welfare", "poverty", "social", "veterans", "housing"],
    "Environment & Energy":    ["environment", "climate", "energy", "infrastructure", "rural", "utilities", "science"],
    "Natl Security & Foreign": ["security", "foreign", "war", "military", "nuclear", "wartime", "international"],
}
BASE2CAT = {b: cat for cat, bases in TAXONOMY.items() for b in bases}

# policy_id -> category (via domain first token)
import pandas as pd
bat = pd.read_csv(REPO / "us_policy_event_battery_v4.csv")
pid2cat, pid2base = {}, {}
for _, r in bat.iterrows():
    base = str(r["domain"]).split("_")[0]
    pid2cat[str(r["event_id"])] = BASE2CAT.get(base, "UNMAPPED")
    pid2base[str(r["event_id"])] = base
unmapped = sorted({pid2base[p] for p, c in pid2cat.items() if c == "UNMAPPED"})
assert not unmapped, f"UNMAPPED base tokens: {unmapped}"
print("category coverage (all 211):", dict(collections.Counter(pid2cat.values())))

# --- load tidy + compute per-policy z (raw + calendar-detrended), per X ---
rows = list(csv.DictReader(open(HERE / "nano_tidy.csv", encoding="utf-8")))
for r in rows:
    r["ym"], r["rel"], r["p"] = int(r["year_model"]), int(r["rel_year"]), float(r["p_yes"])
    r["cat"] = pid2cat.get(r["policy_id"], "UNMAPPED")

XS = [5000, 10000, 20000]

def per_policy_z(recs, valkey):
    """Return {rel: [z...]} aggregating per-policy z-scores (>=5 yr/policy)."""
    out = collections.defaultdict(list)
    bypol = collections.defaultdict(list)
    for r in recs:
        bypol[r["policy_id"]].append(r)
    for prs in bypol.values():
        vs = [r[valkey] for r in prs]
        if len(vs) < 5:
            continue
        m, sd = st.mean(vs), st.pstdev(vs)
        if sd == 0:
            continue
        for r in prs:
            out[r["rel"]].append((r[valkey] - m) / sd)
    return out

def detrend(recs):
    """Add r['pres'] = p residualized on global p~year_model (removes calendar drift)."""
    yms = [r["ym"] for r in recs]; ps = [r["p"] for r in recs]
    n = len(yms); mym = st.mean(yms); mp = st.mean(ps)
    b = sum((yms[i]-mym)*(ps[i]-mp) for i in range(n)) / sum((y-mym)**2 for y in yms)
    a = mp - b*mym
    for r in recs:
        r["pres"] = r["p"] - (a + b*r["ym"])

byX = {x: [r for r in rows if int(r["X"]) == x] for x in XS}
for x in XS:
    detrend(byX[x])

RELS = list(range(-10, 11))
def curve(zmap):
    return [st.mean(zmap[k]) if zmap.get(k) else float("nan") for k in RELS]

# =================== FIG 1: overall n-shape (z-normalized) ===================
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
colors = {5000: "#1f77b4", 10000: "#d62728", 20000: "#2ca02c"}
for ax, (valkey, title) in zip(axes, [("p", "RAW per-policy z"), ("pres", "CALENDAR-DETRENDED z")]):
    for x in XS:
        zmap = per_policy_z(byX[x], valkey)
        ax.plot(RELS, curve(zmap), "-o", ms=3, color=colors[x], label=f"X={x//1000}k")
    ax.axvline(0, color="k", lw=1, ls="--", alpha=.6)
    ax.axhline(0, color="gray", lw=.8, alpha=.5)
    ax.set_title(title); ax.set_xlabel("year_model - enactment_year"); ax.grid(alpha=.25)
    ax.legend(title="data/yr")
axes[0].set_ylabel("mean per-policy z-score of P(yes)")
fig.suptitle("nanochat LoRA-only: per-policy z-normalized P(yes) vs years-to-enactment (n=211 policies, 1931-2020)", fontsize=11)
fig.tight_layout()
fig.savefig(HERE / "figures" / "nshape_znorm.png", dpi=130); plt.close(fig)
print("wrote nshape_znorm.png")

# =================== per-category z curves (X=10000) ===================
X = 10000
recs = byX[X]
cat_order_data = {}
for cat in TAXONOMY:
    cr = [r for r in recs if r["cat"] == cat]
    npol = len({r["policy_id"] for r in cr})
    zraw = per_policy_z(cr, "p"); zdet = per_policy_z(cr, "pres")
    # pre-enactment increment (rel -10 -> 0) on detrended
    pre = [st.mean(zdet[k]) for k in range(-10, 1) if zdet.get(k)]
    inc = (pre[-1] - pre[0]) if len(pre) >= 2 else float("nan")
    cat_order_data[cat] = dict(npol=npol, zraw=zraw, zdet=zdet, inc=inc)
ordered = sorted(TAXONOMY, key=lambda c: cat_order_data[c]["inc"], reverse=True)

# =================== FIG 2: domain small-multiples (z-normalized) ===================
fig, axes = plt.subplots(2, 5, figsize=(19, 7.5), sharex=True, sharey=True)
for ax, cat in zip(axes.flat, ordered):
    d = cat_order_data[cat]
    ax.plot(RELS, curve(d["zraw"]), "-o", ms=2.5, color="#999999", label="raw z")
    ax.plot(RELS, curve(d["zdet"]), "-o", ms=2.5, color="#d62728", label="detrended z")
    ax.axvline(0, color="k", lw=.8, ls="--", alpha=.6); ax.axhline(0, color="gray", lw=.6, alpha=.5)
    ax.set_title(f"{cat}\n(n={d['npol']} policies, pre-rise {d['inc']:+.2f}z)", fontsize=9)
    ax.grid(alpha=.2)
axes[0, 0].legend(fontsize=7)
fig.supxlabel("year_model - enactment_year"); fig.supylabel("mean per-policy z-score of P(yes)")
fig.suptitle("nanochat LoRA-only @ X=10k: z-normalized P(yes) trajectory by policy domain "
             "(all 211 policies in 10 categories; sorted by detrended pre-enactment rise)", fontsize=12)
fig.tight_layout(rect=[0.01, 0.02, 1, 0.95])
fig.savefig(HERE / "figures" / "domain_znorm_smallmult.png", dpi=130); plt.close(fig)
print("wrote domain_znorm_smallmult.png")

# =================== FIG 3: per-domain pre-enactment increment bar ===================
fig, ax = plt.subplots(figsize=(11, 6))
cats = ordered
incs = [cat_order_data[c]["inc"] for c in cats]
ns = [cat_order_data[c]["npol"] for c in cats]
bars = ax.barh(range(len(cats)), incs, color=["#2ca02c" if v > 0 else "#d62728" for v in incs])
ax.set_yticks(range(len(cats))); ax.set_yticklabels([f"{c}  (n={n})" for c, n in zip(cats, ns)], fontsize=9)
ax.invert_yaxis(); ax.axvline(0, color="k", lw=1)
ax.set_xlabel("detrended pre-enactment z-rise (rel -10 -> 0), X=10k")
ax.set_title("nanochat: which policy domains show a pre-enactment rise after removing calendar drift?\n"
             "(per-policy z-normalized; positive = belief rises approaching enactment)", fontsize=11)
for i, v in enumerate(incs):
    ax.text(v + (0.01 if v >= 0 else -0.01), i, f"{v:+.2f}", va="center",
            ha="left" if v >= 0 else "right", fontsize=8)
ax.grid(axis="x", alpha=.25); fig.tight_layout()
fig.savefig(HERE / "figures" / "domain_increment_bar.png", dpi=130); plt.close(fig)
print("wrote domain_increment_bar.png")

# print the per-domain table for the writeup
print("\nper-domain detrended pre-enactment z-rise (X=10k), ranked:")
for c in ordered:
    print(f"  {c:<26} n={cat_order_data[c]['npol']:>2}  rise={cat_order_data[c]['inc']:+.3f}z")
