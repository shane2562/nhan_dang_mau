# Phát hiện người tham gia giao thông không đội mũ bảo hiểm (YOLOv8 + EdgeVision)

## 0. Cài môi trường trên Windows + VS Code
1. Cài Python 3.10 hoặc 3.11 (python.org, tick "Add to PATH"), VS Code + extension **Python**.
2. Mở thư mục dự án bằng VS Code, mở Terminal (PowerShell):
   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1      # nếu bị chặn: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
   python -m pip install --upgrade pip
   ```
3. Có GPU NVIDIA: cài PyTorch CUDA trước (xem lệnh đúng tại pytorch.org), ví dụ:
   `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`
   Không có GPU: bỏ qua, dùng `--device cpu` (chậm, nên giảm `--epochs 20`).
4. `pip install -r requirements.txt`
5. Ctrl+Shift+P -> "Python: Select Interpreter" -> chọn `.venv`.

## 1. Tải dữ liệu (có nguồn xác minh, có bounding box)
- Tên: **EdgeVision Dataset**, Mendeley Data, DOI **10.17632/j82bnw7gsr.1**, giấy phép **CC BY 4.0**
- Link: https://data.mendeley.com/datasets/j82bnw7gsr/1  (bấm "Download All")
- 2.392 ảnh, 8.275 bbox, 3 lớp: BikeWithRider(0), NoHelmet(1), Helmet(2); có nhãn YOLO/VOC/COCO.
- Giải nén vào `data/raw/` (cần ảnh + file .txt YOLO + classes.txt).
- Trích dẫn: Gajjar K., Patel N., Patel R., Patel K., Dabhi C., Trivedi H., Goyani M., Bhatt C. (2026), EdgeVision Dataset, Mendeley Data, V1, doi:10.17632/j82bnw7gsr.1

## 2. Chạy lần lượt (trong thư mục gốc dự án)
```powershell
python scripts/01_check_and_split.py
python scripts/02_train.py --exp baseline
python scripts/02_train.py --exp improved
python scripts/03_evaluate.py --weights runs/baseline/weights/best.pt runs/improved/weights/best.pt
python demo/app_demo.py --weights runs/improved/weights/best.pt
```
Hết bộ nhớ GPU: giảm `batch` hoặc `imgsz` trong `scripts/02_train.py`.
Kết quả nằm ở `results/` (metrics.csv, violation.csv, errors_*.csv, error_cases_*/).

## 3. Cấu trúc
scripts/ (code), demo/ (Gradio), results/ (số liệu, hình), docs/ (hướng dẫn báo cáo)
