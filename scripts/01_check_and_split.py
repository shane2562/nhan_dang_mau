"""
BƯỚC 1 - Kiểm tra chất lượng dữ liệu + chia train/val/test + vẽ mẫu bbox.
Chạy:  python scripts/01_check_and_split.py
Yêu cầu: đã giải nén EdgeVision (ảnh + nhãn YOLO .txt + classes.txt) vào data/raw/
"""
import hashlib, random, shutil, sys
from collections import Counter
import cv2, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

SEED, SPLIT = 42, (0.70, 0.15, 0.15)
random.seed(SEED)


def main():
    if not RAW.exists():
        sys.exit(f"Chưa có {RAW}. Xem README mục 'Tải dữ liệu'.")
    RESULTS.mkdir(exist_ok=True)
    imgs = [p for p in RAW.rglob("*") if p.suffix.lower() in IMG_EXT]
    labels = {p.stem: p for p in RAW.rglob("*.txt") if p.name.lower() != "classes.txt"}
    print(f"Tìm thấy {len(imgs)} ảnh, {len(labels)} file nhãn YOLO")

    cf = list(RAW.rglob("classes.txt"))
    if cf:
        names = [l.strip() for l in cf[0].read_text(encoding="utf-8").splitlines() if l.strip()]
        print("classes.txt:", names)
        if names != CLASS_NAMES:
            print("CẢNH BÁO: thứ tự lớp khác giả định -> sửa CLASS_NAMES trong common.py")

    rows, problems, md5 = [], [], {}
    for p in imgs:
        lab = labels.get(p.stem)
        im = cv2.imread(str(p))
        if im is None:
            problems.append((p.name, "ảnh hỏng")); continue
        H, W = im.shape[:2]
        h = hashlib.md5(p.read_bytes()).hexdigest()
        if h in md5:
            problems.append((p.name, f"trùng ảnh với {md5[h]}")); continue
        md5[h] = p.name
        if lab is None:
            problems.append((p.name, "thiếu nhãn")); continue
        boxes, errs = read_yolo_label(lab)
        problems += [(p.name, e) for e in errs]
        ok = True
        for c, xc, yc, w, hh in boxes:
            if c not in range(len(CLASS_NAMES)) or not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < w <= 1 and 0 < hh <= 1):
                problems.append((p.name, f"bbox/lớp không hợp lệ {(c, xc, yc, w, hh)}")); ok = False
        if ok:
            rows.append(dict(img=p, label=lab, W=W, H=H, n=len(boxes),
                             cls=[b[0] for b in boxes], areas=[b[3] * b[4] for b in boxes]))
    pd.DataFrame(problems, columns=["file", "van_de"]).to_csv(RESULTS / "data_problems.csv", index=False, encoding="utf-8-sig")
    print(f"Ảnh hợp lệ: {len(rows)} | vấn đề phát hiện: {len(problems)} (xem results/data_problems.csv)")

    cnt = Counter(c for r in rows for c in r["cls"])
    stats = {CLASS_NAMES[k]: cnt.get(k, 0) for k in range(len(CLASS_NAMES))}
    print("Số bbox theo lớp:", stats)
    areas = [a for r in rows for a in r["areas"]]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].bar(list(stats.keys()), list(stats.values())); ax[0].set_title("So bbox theo lop")
    ax[1].hist(areas, bins=40); ax[1].set_title("Phan bo dien tich bbox (ti le so voi anh)")
    plt.tight_layout(); plt.savefig(RESULTS / "data_stats.png", dpi=150); plt.close()

    random.shuffle(rows)
    n = len(rows); a = int(n * SPLIT[0]); b = a + int(n * SPLIT[1])
    parts = {"train": rows[:a], "val": rows[a:b], "test": rows[b:]}
    if DATASET.exists():
        shutil.rmtree(DATASET)
    for sp, items in parts.items():
        (DATASET / "images" / sp).mkdir(parents=True); (DATASET / "labels" / sp).mkdir(parents=True)
        for r in items:
            shutil.copy2(r["img"], DATASET / "images" / sp / r["img"].name)
            shutil.copy2(r["label"], DATASET / "labels" / sp / (r["img"].stem + ".txt"))
        print(sp, len(items))
    yaml = (f"path: {DATASET.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
            f"names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASS_NAMES)))
    (ROOT / "data" / "data.yaml").write_text(yaml, encoding="utf-8")

    colors = {0: (255, 150, 0), 1: (0, 0, 255), 2: (0, 200, 0)}
    fig, axs = plt.subplots(3, 4, figsize=(16, 12))
    for ax, r in zip(axs.ravel(), random.sample(rows, min(12, len(rows)))):
        im = cv2.imread(str(r["img"]))
        for bx in read_yolo_label(r["label"])[0]:
            x1, y1, x2, y2 = map(int, yolo_to_xyxy(bx, r["W"], r["H"]))
            cv2.rectangle(im, (x1, y1), (x2, y2), colors[bx[0]], max(2, r["W"] // 300))
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); ax.axis("off")
    plt.tight_layout(); plt.savefig(RESULTS / "sample_boxes.png", dpi=100); plt.close()
    print("Xong. Đã tạo data/data.yaml, results/data_stats.png, results/sample_boxes.png")


if __name__ == "__main__":
    main()
