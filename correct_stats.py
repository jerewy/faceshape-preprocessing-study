import numpy as np
from scipy.stats import binomtest
from math import sqrt

data = np.load("runs/test_predictions.npz")
T = data["targets"]; P = {k: data[k] for k in data.files if k != "targets"}
n = len(T)
MODELS = ["resnet50", "efficientnetv2s", "mobilenetv3small", "swint"]

def mcnemar(ak, bk):
    ca, cb = (P[ak] == T), (P[bk] == T)
    b = int(np.sum(ca & ~cb)); c = int(np.sum(~ca & cb))
    p = 1.0 if (b + c) == 0 else binomtest(min(b, c), b + c, 0.5).pvalue
    diff = (b - c) / n
    var = (b + c - (b - c) ** 2 / n) / n ** 2
    h = 1.96 * sqrt(max(var, 0))
    return dict(p=p, accA=ca.mean(), accB=cb.mean(), diff=diff, lo=diff - h, hi=diff + h)

comps = []
for m in MODELS:
    for va, vb in [("D1", "D2"), ("D2", "D3"), ("D3", "D4"), ("D1", "D3")]:
        comps.append((f"{m}: {vb} vs {va}", f"{m}_{vb}", f"{m}_{va}"))
comps.append(("swint_D4 vs effnetv2s_D3", "swint_D4", "efficientnetv2s_D3"))

rows = [dict(name=nm, **mcnemar(a, b)) for nm, a, b in comps]
pvals = np.array([r["p"] for r in rows]); mt = len(pvals)
order = np.argsort(pvals); run = 0; adj = np.empty(mt)
for rank, idx in enumerate(order):
    run = max(run, (mt - rank) * pvals[idx]); adj[idx] = min(run, 1.0)
for i, r in enumerate(rows): r["holm"] = adj[i]

print(f"n_test={n}, {mt} tests, Holm-Bonferroni\n")
print(f"{'comparison':30s} {'diff%':>6s} {'95% CI':>15s} {'p_raw':>8s} {'p_holm':>8s}  sig")
for r in rows:
    ci = f"[{r['lo']*100:+.1f},{r['hi']*100:+.1f}]"
    print(f"{r['name']:30s} {r['diff']*100:+6.1f} {ci:>15s} {r['p']:8.4f} {r['holm']:8.4f}  {'YES' if r['holm']<0.05 else 'no'}")

def wilson(k, n, z=1.96):
    p = k/n; d = 1+z*z/n
    return ((p+z*z/(2*n))/d - z*sqrt(p*(1-p)/n+z*z/(4*n*n))/d,
            (p+z*z/(2*n))/d + z*sqrt(p*(1-p)/n+z*z/(4*n*n))/d)
print("\nPer-cell accuracy [95% Wilson CI]:")
for m in MODELS:
    s = m.ljust(16)
    for v in ["D1","D2","D3","D4"]:
        k = int((P[f"{m}_{v}"]==T).sum()); lo,hi = wilson(k,n)
        s += f"  {v} {k/n*100:.1f}[{lo*100:.0f}-{hi*100:.0f}]"
    print(s)

L = [f"# Significance with Holm correction + 95% CIs (n_test={n})\n",
     f"Exact McNemar (two-sided), Holm-Bonferroni across all {mt} pairwise tests.\n",
     "| Comparison | diff (95% CI) | p raw | p Holm | sig after correction |",
     "|---|---|---|---|---|"]
for r in rows:
    L.append(f"| {r['name']} | {r['diff']*100:+.1f} [{r['lo']*100:+.1f}, {r['hi']*100:+.1f}] | {r['p']:.4f} | {r['holm']:.4f} | {'YES' if r['holm']<0.05 else 'no'} |")
open("../paper_assets/significance_corrected.md", "w", encoding="utf-8").write("\n".join(L))
print("\nwrote paper_assets/significance_corrected.md")
