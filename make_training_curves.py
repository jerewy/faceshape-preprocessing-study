import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("../paper_assets")
COL = {"D1": "#888888", "D2": "#3b82f6", "D3": "#16a34a", "D4": "#b9772a"}
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

# Panel A: validation accuracy by preprocessing version (EfficientNetV2-S)
for v in ["D1", "D2", "D3", "D4"]:
    df = pd.read_csv(f"runs/efficientnetv2s_{v}/epochs.csv")
    ax[0].plot(df["epoch"], df["val_acc"], label=v, color=COL[v], lw=1.9)
ax[0].set_title("(a) Validation accuracy by preprocessing\n(EfficientNetV2-S)")
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("validation accuracy")
ax[0].legend(title="version"); ax[0].grid(alpha=0.3)

# Panel B: train vs val for the best config (no-overfitting check)
df = pd.read_csv("runs/efficientnetv2s_D3/epochs.csv")
ax[1].plot(df["epoch"], df["train_acc"], label="train", color="#16a34a", lw=1.9)
ax[1].plot(df["epoch"], df["val_acc"], label="validation", color="#b9772a", lw=1.9)
ax[1].set_title("(b) Train vs validation accuracy\n(EfficientNetV2-S, D3)")
ax[1].set_xlabel("epoch"); ax[1].set_ylabel("accuracy")
ax[1].legend(); ax[1].grid(alpha=0.3)

plt.tight_layout()
fig.savefig(OUT / "fig_training_curves.png", dpi=200, bbox_inches="tight")
fig.savefig(OUT / "fig_training_curves.pdf", bbox_inches="tight")
print("saved fig_training_curves.png/.pdf")
