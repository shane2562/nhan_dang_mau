"""
BƯỚC 3 - Đánh giá trên TẬP TEST + phân tích lỗi.
Chạy:  python scripts/03_evaluate.py --weights runs/baseline/weights/best.pt runs/improved/weights/best.pt
Sinh ra: results/metrics.csv, results/violation.csv, results/errors_<tên>.csv, results/error_cases_<tên>/
"""
import argparse, time
import cv2, pandas as pd
from ultralytics import YOLO
from common import *

CONF, IOU_T = 0.25, 0.5


def evaluate_one(wpath):
    name = Path(wpath).parent.parent.name
    model = YOLO(wpath)
    r = model.val(data=str(ROOT / "data" / "data.yaml"), split="test", conf=0.001, iou=0.6, verbose=False)
    rows = [dict(model=name, cls="ALL", P=r.box.mp, R=r.box.mr, mAP50=r.box.map50, mAP50_95=r.box.map)]
    for i, ci in enumerate(r.box.ap_class_index):
        rows.append(dict(model=name, cls=CLASS_NAMES[ci], P=r.box.p[i], R=r.box.r[i],
                         mAP50=r.box.ap50[i], mAP50_95=r.box.ap[i]))

    out = RESULTS / f"error_cases_{name}"; out.mkdir(exist_ok=True, parents=True)
    errs, vt = [], dict(tp=0, fp=0, fn=0, tn=0); times = []
    for ip in sorted((DATASET / "images" / "test").iterdir()):
        im = cv2.imread(str(ip)); H, W = im.shape[:2]
        gt = read_yolo_label(DATASET / "labels" / "test" / (ip.stem + ".txt"))[0]
        t = time.perf_counter(); pr = model.predict(im, conf=CONF, iou=0.6, verbose=False)[0]; times.append(time.perf_counter() - t)
        preds = [(int(b.cls), *b.xyxy[0].tolist(), float(b.conf)) for b in pr.boxes]
        used, bad = set(), []
        for g in gt:
            gb = yolo_to_xyxy(g, W, H)
            best = max(((iou(gb, p[1:5]), j) for j, p in enumerate(preds) if j not in used), default=(0, -1))
            if best[0] >= IOU_T:
                used.add(best[1])
                if preds[best[1]][0] != g[0]:
                    bad.append(("nham_lop", CLASS_NAMES[g[0]], CLASS_NAMES[preds[best[1]][0]]))
            else:
                bad.append(("bo_sot_FN", CLASS_NAMES[g[0]], ""))
        bad += [("du_FP", "", CLASS_NAMES[p[0]]) for j, p in enumerate(preds) if j not in used]
        for k, gcls, pcls in bad:
            errs.append(dict(image=ip.name, loai_loi=k, gt=gcls, pred=pcls))
        gv = any(g[0] == NO_HELMET_ID for g in gt); pv = any(p[0] == NO_HELMET_ID for p in preds)
        vt["tp" if gv and pv else "fp" if pv else "fn" if gv else "tn"] += 1
        if bad:
            for g in gt:
                x1, y1, x2, y2 = map(int, yolo_to_xyxy(g, W, H)); cv2.rectangle(im, (x1, y1), (x2, y2), (0, 200, 0), 3)
            for p in preds:
                cv2.rectangle(im, (int(p[1]), int(p[2])), (int(p[3]), int(p[4])), (0, 0, 255), 2)
                cv2.putText(im, f"{CLASS_NAMES[p[0]]} {p[5]:.2f}", (int(p[1]), max(20, int(p[2]) - 5)), 0, 0.8, (0, 0, 255), 2)
            cv2.imwrite(str(out / ip.name), im)
    pd.DataFrame(errs).to_csv(RESULTS / f"errors_{name}.csv", index=False, encoding="utf-8-sig")
    tp, fp, fn = vt["tp"], vt["fp"], vt["fn"]
    prec = tp / (tp + fp) if tp + fp else 0; rec = tp / (tp + fn) if tp + fn else 0
    vio = dict(model=name, **vt, precision=prec, recall=rec, f1=2 * prec * rec / (prec + rec) if prec + rec else 0,
               ms_per_image=1000 * sum(times) / len(times))
    return rows, vio


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--weights", nargs="+", required=True)
    a = ap.parse_args(); allm, allv = [], []
    for w in a.weights:
        m, v = evaluate_one(w); allm += m; allv.append(v)
    pd.DataFrame(allm).round(4).to_csv(RESULTS / "metrics.csv", index=False)
    pd.DataFrame(allv).round(4).to_csv(RESULTS / "violation.csv", index=False)
    print(pd.DataFrame(allm).round(3).to_string()); print(pd.DataFrame(allv).round(3).to_string())


if __name__ == "__main__":
    main()
