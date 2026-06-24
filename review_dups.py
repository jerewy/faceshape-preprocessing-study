import csv
from pathlib import Path
import cv2, numpy as np

CLASSES = ["heart","oblong","oval","round","square"]
EXTS = {".jpg",".jpeg",".png",".bmp",".webp"}
RAW = Path("data/raw")
rows = list(csv.DictReader(open("splits/split.csv", encoding="utf-8")))

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

info = {}
for r in rows:
    base = Path(r["path"].replace("\\","/")).name; lab = r["label"]
    p = raw_index.get((lab,base))
    if p is None: continue
    h = dhash(p)
    if h is None: continue
    info[(lab,base)] = (h, r["split"], p)

pairs = []
for lab in CLASSES:
    keys = [k for k in info if k[0]==lab]
    arr = np.array([info[k][0] for k in keys], dtype=np.uint64); n=len(arr)
    if n<2: continue
    dist = POP[(arr[:,None]^arr[None,:]).view(np.uint8).reshape(n,n,8)].sum(-1)
    iu = np.triu_indices(n,1)
    for idx in np.where(dist[iu]<=5)[0]:
        i,j = iu[0][idx], iu[1][idx]
        ka,kb = keys[i],keys[j]
        if info[ka][1] != info[kb][1]:
            pairs.append((int(dist[i,j]), lab, ka[1], info[ka][1], info[ka][2],
                                              kb[1], info[kb][1], info[kb][2]))
pairs.sort(key=lambda x: x[0])

with open("../paper_assets/dup_pairs.csv","w",newline="",encoding="utf-8") as f:
    w=csv.writer(f); w.writerow(["hamming","class","fileA","splitA","fileB","splitB"])
    for d,lab,fa,sa,pa,fb,sb,pb in pairs: w.writerow([d,lab,fa,sa,fb,sb])

print(f"cross-split pairs (Hamming<=5): {len(pairs)}")
for n,(d,lab,fa,sa,pa,fb,sb,pb) in enumerate(pairs,1):
    print(f"  #{n:2d} H={d} [{lab}] {sa}:{fa}  <->  {sb}:{fb}")

CELL,CAP = 170,26
canvas = np.full((len(pairs)*(CELL+CAP), CELL*2, 3), 255, np.uint8)
def load(p):
    im = cv2.imread(str(p))
    return cv2.resize(im,(CELL,CELL)) if im is not None else np.full((CELL,CELL,3),200,np.uint8)
for idx,(d,lab,fa,sa,pa,fb,sb,pb) in enumerate(pairs):
    y = idx*(CELL+CAP)
    tag = "IDENTICAL" if d==0 else f"near (H={d})"
    cv2.putText(canvas,f"#{idx+1} {lab}  {tag}   L={sa}  R={sb}",(4,y+18),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)
    canvas[y+CAP:y+CAP+CELL, 0:CELL] = load(pa)
    canvas[y+CAP:y+CAP+CELL, CELL:2*CELL] = load(pb)
cv2.imwrite("../paper_assets/dup_pairs_review.png", canvas)
print("wrote dup_pairs_review.png", canvas.shape)
