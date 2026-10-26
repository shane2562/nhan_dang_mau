"""
DEMO tương tác (Gradio): ảnh / webcam / video, chỉnh ngưỡng tin cậy, xem bbox + cảnh báo vi phạm.
Chạy:  python demo/app_demo.py --weights runs/improved/weights/best.pt
Mở trình duyệt tại http://127.0.0.1:7860
"""
import argparse, sys, tempfile
from pathlib import Path
import cv2, gradio as gr
from ultralytics import YOLO
import subprocess, imageio_ffmpeg
import time

NAMES = ["BikeWithRider", "NoHelmet", "Helmet"]
COL = {0: (255, 150, 0), 1: (0, 0, 255), 2: (0, 200, 0)}   # BGR
ap = argparse.ArgumentParser(); ap.add_argument("--weights", default="runs/improved/weights/best.pt")
args, _ = ap.parse_known_args()
if not Path(args.weights).exists():
    sys.exit(f"Không thấy {args.weights}. Hãy huấn luyện trước (scripts/02_train.py).")
model = YOLO(args.weights)


def draw(frame, conf, show, imgsz=640):
    r = model.predict(frame, conf=conf, iou=0.6, imgsz=imgsz, verbose=False)[0]
    n_no = n_ok = 0
    for b in r.boxes:
        c = int(b.cls)
        n_no += c == 1; n_ok += c == 2
        if c not in show: continue
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), COL[c], 3)
        cv2.putText(frame, f"{float(b.conf):.2f}", (x1, max(22, y1 - 6)), 0, 0.8, COL[c], 2)
    return frame, n_no, n_ok


def run_image(img, conf, show_names):
    if img is None: return None, ""
    show = {NAMES.index(s) for s in show_names}
    bgr, n_no, n_ok = draw(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), conf, show)
    msg = (f"VI PHẠM: Phát hiện {n_no} người không đội mũ" if n_no else "Không phát hiện vi phạm") 
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), msg


def run_video(path, conf, show_names):
    if path is None: return None, ""
    show = {NAMES.index(s) for s in show_names}
    cap = cv2.VideoCapture(path); fps = cap.get(5) or 25
    W, H = int(cap.get(3)), int(cap.get(4))
    raw = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H)); viol = tot = 0
    while True:
        ok, f = cap.read()
        if not ok: break
        f, n_no, _ = draw(f, conf, show); vw.write(f); tot += 1; viol += n_no > 0
    cap.release(); vw.release()
    # đổi sang H.264 để trình duyệt phát được
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", raw, "-vcodec", "libx264",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", out],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out, f"{viol}/{tot} khung hình có người không đội mũ"

def run_stream(frame, conf, show_names):
    if frame is None:
        return None, ""
    show = {NAMES.index(s) for s in show_names}
    t = time.perf_counter()
    bgr, n_no, n_ok = draw(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), conf, show, imgsz=416)
    fps = 1 / max(time.perf_counter() - t, 1e-6)
    msg = (f"VI PHẠM: {n_no} người KHÔNG đội mũ" if n_no else "Không phát hiện vi phạm")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), f"{msg} | {fps:.1f} FPS"


with gr.Blocks(title="Phát hiện không đội mũ bảo hiểm") as ui:
    gr.Markdown("## Phát hiện người đi xe máy không đội mũ bảo hiểm (YOLOv8)")
    conf = gr.Slider(0.05, 0.9, 0.25, step=0.05, label="Ngưỡng tin cậy (confidence)")
    show = gr.CheckboxGroup(NAMES, value=NAMES, label="Lớp hiển thị")
    with gr.Tab("Ảnh / Webcam"):
        with gr.Row():
            i = gr.Image(sources=["upload", "webcam"], type="numpy", label="Ảnh đầu vào", height=420)
            o = gr.Image(label="Kết quả", height=420)
        t = gr.Textbox(label="Kết luận")
        gr.Button("Phát hiện").click(run_image, [i, conf, show], [o, t])
    with gr.Tab("Video"):
        with gr.Row():
            v = gr.Video(label="Video đầu vào", height=420)
            ov = gr.Video(label="Kết quả", height=420)
        tv = gr.Textbox(label="Kết luận")
        gr.Button("Xử lý video").click(run_video, [v, conf, show], [ov, tv])
    with gr.Tab("Realtime"):
        with gr.Row():
            cam = gr.Image(sources=["webcam"], streaming=True, type="numpy", label="Webcam", height=420)
            rt = gr.Image(label="Kết quả realtime", height=420)
        rt_txt = gr.Textbox(label="Kết luận realtime")
        cam.stream(run_stream, [cam, conf, show], [rt, rt_txt],
                   stream_every=0.1, time_limit=None, concurrency_limit=1)
if __name__ == "__main__":
    ui.launch()
