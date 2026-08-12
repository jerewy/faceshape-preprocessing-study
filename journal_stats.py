"""Journal-level evaluation: per-sample predictions -> McNemar significance,
efficiency metrics, and per-class breakdowns. Writes a report to paper_assets/."""
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data import CLASSES, FaceShapeDataset, load_split
from src.models import build_model
from src.preprocessing import build_transform, source_root_for

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS = ["resnet50", "efficientnetv2s", "mobilenetv3small", "swint"]
VERSIONS = ["D1", "D2", "D3", "D4"]
SPLIT, DATA_ROOT, PRE_ROOT = "splits/split.csv", "data/raw", "data/preprocessed"
OUT = Path("../paper_assets")


@torch.no_grad()
def predict(model_name, version):
    src = source_root_for(version, DATA_ROOT, PRE_ROOT)
    items = load_split(SPLIT, "test", source_root=src)
    ds = FaceShapeDataset(items, build_transform(version, "test", 224))
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    m = build_model(model_name, num_classes=len(CLASSES), pretrained=False).to(DEVICE).eval()
    m.load_state_dict(torch.load(f"runs/{model_name}_{version}/best.pt",
                                 map_location=DEVICE, weights_only=True))
    preds, tgts = [], []
    for x, y in dl:
        preds.append(m(x.to(DEVICE)).argmax(1).cpu().numpy()); tgts.append(y.numpy())
    del m; torch.cuda.empty_cache()
    return np.concatenate(preds), np.concatenate(tgts)


# ---- 1. collect predictions + sanity-check against results.json ----
print("== predictions (sanity vs results.json) ==")
P, T = {}, None
for mdl in MODELS:
    for v in VERSIONS:
        if not Path(f"runs/{mdl}_{v}/best.pt").exists():
            continue
        pr, tg = predict(mdl, v)
        P[(mdl, v)] = pr
        T = tg if T is None else T
        assert np.array_equal(T, tg), f"target order mismatch at {mdl}_{v}"
        acc = (pr == tg).mean()
        rj = json.load(open(f"runs/{mdl}_{v}/results.json"))["test_accuracy"]
        print(f"  {mdl}_{v}: {acc*100:5.1f}%  (json {rj*100:5.1f}%)"
              + ("" if abs(acc - rj) < 0.01 else "  <-- MISMATCH"))
N = len(T)
np.savez("runs/test_predictions.npz", targets=T, **{f"{m}_{v}": P[(m, v)] for (m, v) in P})
print(f"  test samples: {N}\n")

# ---- 2. McNemar ----
try:
    from scipy.stats import binomtest
    def pval(b, c):
        n = b + c
        return 1.0 if n == 0 else binomtest(min(b, c), n, 0.5).pvalue
except Exception:
    from scipy.stats import chi2
    def pval(b, c):
        n = b + c
        return 1.0 if n == 0 else float(chi2.sf((abs(b - c) - 1) ** 2 / n, 1))


def mcnemar(ka, kb):
    ca, cb = (P[ka] == T), (P[kb] == T)
    b = int(np.sum(ca & ~cb)); c = int(np.sum(~ca & cb))
    return b, c, pval(b, c)


sig_rows = []
for mdl in MODELS:
    for va, vb in [("D1", "D2"), ("D2", "D3"), ("D3", "D4"), ("D1", "D3")]:
        if (mdl, va) in P and (mdl, vb) in P:
            b, c, p = mcnemar((mdl, vb), (mdl, va))
            sig_rows.append((mdl, f"{vb} vs {va}", (P[(mdl, vb)] == T).mean(),
                             (P[(mdl, va)] == T).mean(), b, c, p))
# top-2 model comparison at their best versions
if ("swint", "D4") in P and ("efficientnetv2s", "D3") in P:
    b, c, p = mcnemar(("swint", "D4"), ("efficientnetv2s", "D3"))
    sig_rows.append(("swint_D4 vs effnetv2s_D3", "top-2", (P[("swint", "D4")] == T).mean(),
                     (P[("efficientnetv2s", "D3")] == T).mean(), b, c, p))

# ---- 3. efficiency ----
eff_rows = []
for mdl in MODELS:
    m = build_model(mdl, num_classes=len(CLASSES), pretrained=False).eval()
    params = sum(p.numel() for p in m.parameters())
    mg = m.to(DEVICE)
    x = torch.randn(32, 3, 224, 224, device=DEVICE)
    with torch.no_grad():
        for _ in range(5): mg(x)
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(20): mg(x)
        torch.cuda.synchronize(); gpu_ms = (time.time() - t0) / 20 / 32 * 1000
    mc = m.to("cpu").eval(); xc = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        for _ in range(3): mc(xc)
        t0 = time.time()
        for _ in range(10): mc(xc)
        cpu_ms = (time.time() - t0) / 10 * 1000
    eff_rows.append((mdl, params / 1e6, params * 4 / 1e6, gpu_ms, cpu_ms))
    del m, mg, mc; torch.cuda.empty_cache()

# ---- 4. per-class for best + deployment configs ----
from sklearn.metrics import classification_report
per_class = {}
for cfg in [("efficientnetv2s", "D3"), ("swint", "D4"), ("mobilenetv3small", "D3")]:
    if cfg in P:
        per_class[cfg] = classification_report(T, P[cfg], target_names=CLASSES,
                                                digits=3, zero_division=0)

# ---- 5. write report ----
OUT.mkdir(parents=True, exist_ok=True)
L = []
L.append(f"# Paper 2 — Journal-level statistics (n_test = {N})\n")
L.append(f"Device: {DEVICE}. McNemar = exact (binomial) two-sided on paired test predictions.\n")
L.append("## Significance (McNemar)\n")
L.append("`b` = first config correct & second wrong; `c` = reverse. p<0.05 = bold.\n")
L.append("| Comparison | acc A | acc B | b | c | p | significant |")
L.append("|---|---|---|---|---|---|---|")
for r in sig_rows:
    name, cmp, aA, aB, b, c, p = r
    sig = "**yes**" if p < 0.05 else "no"
    L.append(f"| {name} — {cmp} | {aA*100:.1f} | {aB*100:.1f} | {b} | {c} | {p:.4f} | {sig} |")
L.append("\n## Efficiency\n")
L.append("| Model | Params (M) | Size fp32 (MB) | GPU ms/img | CPU ms/img |")
L.append("|---|---|---|---|---|")
for mdl, pm, mb, g, cpu in eff_rows:
    L.append(f"| {mdl} | {pm:.1f} | {mb:.0f} | {g:.2f} | {cpu:.1f} |")
L.append("\n## Per-class (precision / recall / F1)\n")
for cfg, rep in per_class.items():
    L.append(f"### {cfg[0]} {cfg[1]}\n```\n{rep}\n```")
(OUT / "journal_stats_report.md").write_text("\n".join(L), encoding="utf-8")

# console summary
print("== McNemar ==")
for name, cmp, aA, aB, b, c, p in sig_rows:
    print(f"  {name:34s} {cmp:9s} accA={aA*100:5.1f} accB={aB*100:5.1f} b={b:3d} c={c:3d} p={p:.4f} {'SIG' if p<0.05 else 'ns'}")
print("== efficiency ==")
for mdl, pm, mb, g, cpu in eff_rows:
    print(f"  {mdl:16s} {pm:6.1f}M  {mb:4.0f}MB  GPU {g:5.2f}ms  CPU {cpu:6.1f}ms")
print(f"\nwrote {OUT/'journal_stats_report.md'} and runs/test_predictions.npz")
