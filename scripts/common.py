"""Hàm dùng chung: đường dẫn, tên lớp, đọc nhãn YOLO, IoU."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"          # giải nén EdgeVision vào đây
DATASET = ROOT / "data" / "dataset"  # thư mục sau khi chia train/val/test
RESULTS = ROOT / "results"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

# Thứ tự lớp cố định theo mô tả của tác giả dataset (classes.txt)
CLASS_NAMES = ["BikeWithRider", "NoHelmet", "Helmet"]
NO_HELMET_ID = 1


def read_yolo_label(path):
    """Trả về (list (cls, xc, yc, w, h) chuẩn hóa 0-1, list lỗi)."""
    rows, errors = [], []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        p = line.split()
        if not p:
            continue
        try:
            if len(p) != 5:
                raise ValueError("không đủ 5 cột")
            c = int(float(p[0]))
            xc, yc, w, h = map(float, p[1:])
            rows.append((c, xc, yc, w, h))
        except ValueError as e:
            errors.append(f"{Path(path).name}:{i}: {e}")
    return rows, errors


def yolo_to_xyxy(box, W, H):
    _, xc, yc, w, h = box
    return ((xc - w / 2) * W, (yc - h / 2) * H, (xc + w / 2) * W, (yc + h / 2) * H)


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0
