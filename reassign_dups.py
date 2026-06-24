import csv, shutil
from pathlib import Path
from collections import defaultdict, Counter
import cv2, numpy as np

CLASSES = ["heart","oblong","oval","round","square"]
EXTS = {".jpg",".jpeg",".png",".bmp",".webp"}
RAW, SPLIT = Path("data/raw"), Path("splits/split.csv")
rows = list(csv.DictReader(open(SPLIT, encoding="utf-8")))

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

key_of_row, hashes = {}, {}
for r in rows:
    k = (r["label"], Path(r["path"].replace("\\","/")).name); key_of_row[k] = r
    p = raw_index.get(k)
    if p:
        h = dhash(p)
        if h is not None: hashes[k] = h

# near-dup graph (Hamming<=5, all pairs)
adj = defaultdict(set)
for lab in CLASSES:
    keys = [k for k in hashes if k[0]==lab]
    arr = np.array([hashes[k] for k in keys], dtype=np.uint64); n=len(arr)
    if n<2: continue
    dist = POP[(arr[:,None]^arr[None,:]).view(np.uint8).reshape(n,n,8)].sum(-1)
    iu = np.triu_indices(n,1)
    for idx in np.where(dist[iu]<=5)[0]:
        i,j = iu[0][idx], iu[1][idx]
        adj[keys[i]].add(keys[j]); adj[keys[j]].add(keys[i])

# connected components -> assign each multi-split component to its majority split
seen, comps = set(), []
for k in list(adj):
    if k in seen: continue
    stack, comp = [k], []
    while stack:
        x = stack.pop()
        if x in seen: continue
        seen.add(x); comp.append(x); stack.extend(adj[x]-seen)
    comps.append(comp)

before = Counter(r["split"] for r in rows)
changes = 0
for comp in comps:
    splits = [key_of_row[k]["split"] for k in comp]
    if len(set(splits)) == 1: continue
    target = Counter(splits).most_common(1)[0][0]
    for k in comp:
        if key_of_row[k]["split"] != target:
            key_of_row[k]["split"] = target; changes += 1

shutil.copy(SPLIT, SPLIT.with_name("split_original.csv"))
with open(SPLIT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

# verify: cross-split near-dups now
remaining = 0
for lab in CLASSES:
    keys = [k for k in hashes if k[0]==lab]
    arr = np.array([hashes[k] for k in keys], dtype=np.uint64); n=len(arr)
    if n<2: continue
    dist = POP[(arr[:,None]^arr[None,:]).view(np.uint8).reshape(n,n,8)].sum(-1)
    iu = np.triu_indices(n,1)
    for idx in np.where(dist[iu]<=5)[0]:
        i,j = iu[0][idx], iu[1][idx]
        if key_of_row[keys[i]]["split"] != key_of_row[keys[j]]["split"]: remaining += 1

after = Counter(r["split"] for r in rows)
per_class = defaultdict(Counter)
for r in rows: per_class[r["label"]][r["split"]] += 1
print("near-dup components:", len([c for c in comps if len(c)>1]), " reassignments:", changes)
print("split BEFORE:", dict(before))
print("split AFTER :", dict(after))
print("cross-split near-dups remaining:", remaining)
print("per-class totals:", {c: sum(per_class[c].values()) for c in CLASSES})
print("per-class x split:", {c: dict(per_class[c]) for c in CLASSES})
