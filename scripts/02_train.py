"""
BƯỚC 2 - Huấn luyện. Hai cấu hình để so sánh:
  baseline : YOLOv8n, imgsz 640, augmentation mặc định
  improved : YOLOv8s, imgsz 800, tăng augmentation + early stopping
Chạy:  python scripts/02_train.py --exp baseline
       python scripts/02_train.py --exp improved
"""
import argparse
from ultralytics import YOLO
from common import ROOT

CFG = {
    "baseline": dict(model="yolov8n.pt", imgsz=640, batch=16, epochs=50, patience=15),
    "improved": dict(model="yolov8s.pt", imgsz=800, batch=8, epochs=60, patience=20,
                     scale=0.6, hsv_v=0.5, degrees=5.0, close_mosaic=10),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=CFG, required=True)
    ap.add_argument("--device", default="0", help="0 = GPU đầu tiên, cpu = dùng CPU")
    ap.add_argument("--epochs", type=int, default=None)
    a = ap.parse_args()
    cfg = dict(CFG[a.exp]); model = YOLO(cfg.pop("model"))
    if a.epochs:
        cfg["epochs"] = a.epochs
    model.train(data=str(ROOT / "data" / "data.yaml"), project=str(ROOT / "runs"),
                name=a.exp, device=a.device, seed=42, workers=2, exist_ok=True, **cfg)


if __name__ == "__main__":   # bắt buộc trên Windows (multiprocessing)
    main()
