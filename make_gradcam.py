import numpy as np, torch
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from src.models import build_model
from src.data import load_split, CLASSES
from src.preprocessing import build_transform, source_root_for

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = Path("../paper_assets")
VERSIONS = ["D1", "D2", "D3", "D4"]

def load_model(v):
    m = build_model("resnet50", num_classes=len(CLASSES), pretrained=False).to(DEVICE).eval()
    m.load_state_dict(torch.load(f"runs/resnet50_{v}/best.pt", map_location=DEVICE, weights_only=True))
    return m

models = {v: load_model(v) for v in VERSIONS}
cams = {v: GradCAM(model=models[v], target_layers=[models[v].layer4[-1]]) for v in VERSIONS}
tf = build_transform("D1", "test", 224)
items = {v: load_split("splits/split.csv", "test",
         source_root=source_root_for(v, "data/raw", "data/preprocessed")) for v in VERSIONS}

def prep(path):
    img = Image.open(path).convert("RGB").resize((224, 224))
    return np.asarray(img, np.float32) / 255.0, tf(img).unsqueeze(0).to(DEVICE)

ex, seen = [], set()
for idx, (p, lab) in enumerate(items["D1"]):
    if lab not in seen: seen.add(lab); ex.append(idx)
    if len(ex) >= 3: break

n = len(ex)
fig, ax = plt.subplots(n, 4, figsize=(8.4, 2.15 * n))
for r, idx in enumerate(ex):
    for c, v in enumerate(VERSIONS):
        rgb, x = prep(items[v][idx][0])
        ax[r, c].imshow(show_cam_on_image(rgb, cams[v](input_tensor=x, targets=None)[0], use_rgb=True))
        ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
    ax[r, 0].set_ylabel(items["D1"][idx][1], fontsize=12)
for c, v in enumerate(VERSIONS):
    ax[0, c].set_title(v, fontsize=12)
plt.tight_layout()
fig.savefig(OUT / "fig_gradcam.png", dpi=200, bbox_inches="tight")
fig.savefig(OUT / "fig_gradcam.pdf", bbox_inches="tight")
print("saved fig_gradcam: 4 versions D1-D4, no baked-in caption")
