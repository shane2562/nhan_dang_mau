"""Bảng + biểu đồ so sánh baseline (YOLOv8n) và improved (YOLOv8s). Chạy sau 03_evaluate.py."""
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import RESULTS

m = pd.read_csv(RESULTS / "metrics.csv")
v = pd.read_csv(RESULTS / "violation.csv")
if m["model"].nunique() < 2:
    raise SystemExit("metrics.csv mới có 1 mô hình. Chạy 03_evaluate.py với cả 2 weights.")

pivot = m.pivot(index="cls", columns="model", values=["mAP50", "mAP50_95", "R"]).round(4)
pivot.to_csv(RESULTS / "compare_metrics.csv", encoding="utf-8-sig")
print(pivot.to_string()); print(v.to_string(index=False))

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for a, col in zip(ax, ["mAP50", "mAP50_95", "R"]):
    m.pivot(index="cls", columns="model", values=col).plot.bar(ax=a, rot=15)
    a.set_title(col); a.set_ylim(0, 1)
plt.tight_layout(); plt.savefig(RESULTS / "compare.png", dpi=150)
print("Đã tạo results/compare_metrics.csv và results/compare.png")