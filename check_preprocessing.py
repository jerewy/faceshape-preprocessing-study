import csv
from pathlib import Path
from collections import defaultdict
import cv2, numpy as np

CLASSES = ["heart","oblong","oval","round","square"]
EXTS = {".jpg",".jpeg",".png",".bmp",".webp"}
CROP, ALIGN, RAW = Path("data/preprocessed/crop"), Path("data/preprocessed/align"), Path("data/raw")
rows = list(csv.DictReader(open("splits/split.csv", encoding="utf-8")))

# ---- split validation ----
miss_crop, miss_align = [], []
for r in rows:
    base = Path(r["path"].replace("\\","/")).name; lab = r["label"]
    if not (CROP/lab/base).exists():  miss_crop.append((lab,base,r["split"]))
    if not (ALIGN/lab/base).exists(): miss_align.append((lab,base,r["split"]))
print(f"split rows: {len(rows)}")
print(f"missing crop: {len(miss_crop)}   missing align: {len(miss_align)}")
for m in (miss_crop+miss_align)[:10]: print("   MISSING:", m)

# ---- near-duplicate scan (dHash, cross-split) ----
raw_index = {}
for lab in CLASSES:
    for p in (RAW/lab).rglob("*"):
        if p.suffix.lower() in EXTS: raw_index[(lab,p.name)] = p

POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
def dhash(p):
    img = cv2.imread(str(p), cv2.IMREAD_REDUCED_GRAYSCALE_8)
    if img is None: img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    img = cv2.resize(img,(9,8))
    return int.from_bytes(np.packbits(img[:,1:]>img[:,:-1]).tobytes(),"big")

hashes, split_of = {}, {}
for r in rows:
    base = Path(r["path"].replace("\\","/")).name; lab = r["label"]
    p = raw_index.get((lab,base))
    if p is None: continue
    h = dhash(p)
    if h is None: continue
    hashes[(lab,base)] = h; split_of[(lab,base)] = r["split"]
print(f"hashed: {len(hashes)} images")

byhash = defaultdict(list)
for k,h in hashes.items(): byhash[h].append(k)
exact = [g for g in byhash.values() if len(g)>1]
exact_cross = [g for g in exact if len({split_of[k] for k in g})>1]
print(f"EXACT perceptual duplicates: {len(exact)} groups; spanning >1 split: {len(exact_cross)}")

THR = 5; near_total = near_cross = 0
for lab in CLASSES:
    keys = [k for k in hashes if k[0]==lab]
    arr = np.array([hashes[k] for k in keys], dtype=np.uint64); n=len(arr)
    if n<2: continue
    dist = POP[(arr[:,None]^arr[None,:]).view(np.uint8).reshape(n,n,8)].sum(-1)
    iu = np.triu_indices(n,1)
    for idx in np.where(dist[iu]<=THR)[0]:
        i,j = iu[0][idx], iu[1][idx]; near_total += 1
        if split_of[keys[i]] != split_of[keys[j]]: near_cross += 1
print(f"NEAR-dup pairs (Hamming<={THR}): {near_total}; cross-split (potential leakage): {near_cross}")
