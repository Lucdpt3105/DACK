#  Vietnamese Traffic Sign Detection

Chương trình nhận diện biển báo giao thông Việt Nam sử dụng YOLOv8.

---

##  ⚙️ Yêu Cầu

- **Python 3.11** (Khuyên) hoặc Python 3.12+ (tương thích)
- **RAM:** 8GB (16GB nếu có GPU)
- **GPU:** NVIDIA (tùy chọn, tăng tốc 8-10x)

### Python Version
- ✅ **Python 3.11:** Khuyến cáo nhất
- ✅ **Python 3.12+:** Tương thích (dùng `requirements.txt` hoặc `requirements-cpu.txt`)
- ❌ **Python 3.10 trở xuống:** Không khuyến cáo

Nếu **không dùng Python 3.11**, cài thêm:
```bash
pip install -U numpy scipy pillow
```

---

##  Cài đặt

### Linux / macOS
```bash
# Tạo virtual environment (nếu chưa)
python -m venv venv
source venv/bin/activate

# Cài đặt dependencies
pip install -r requirements.txt
```

### Windows
```cmd
# Tạo virtual environment (nếu chưa)
python -m venv venv
venv\Scripts\activate

# Cài đặt dependencies
pip install -r requirements.txt
```

**Lưu ý:** 
- GPU CUDA 12.1: dùng `requirements.txt`
- CPU only: dùng `requirements-cpu.txt`

---

##  🚀 Chạy Nhanh

### GUI Chính (Main Interface)
```bash
python main.py
```
Giao diện tkinter cho phép chọn: Ảnh, Video, hoặc Webcam

### Hoặc chạy trực tiếp từng module

---

##  Nhận diện Ảnh

### Nhận diện một ảnh
```bash
python Detection/image.py --image_path path/to/image.jpg
```

### Nhận diện toàn bộ folder
```bash
python Detection/image.py --folder path/to/folder
```

### Tùy chọn
- `--conf 0.25` - Ngưỡng tự tin (0-1, mặc định 0.25)
- `--imgsz 1280` - Kích thước xử lý (640, 832, 1024, 1280)
  - `1280` tốt cho ảnh lớn/biển báo nhỏ
  - `640` nhanh hơn, phù hợp ảnh bình thường

### Ví dụ
```bash
# Ảnh đơn lẻ với ngưỡng cao
python Detection/image.py --image_path test.jpg --conf 0.3

# Folder ảnh với kích thước cao
python Detection/image.py --folder bienbao --imgsz 1280 --conf 0.25
```

**Output:** Ảnh kết quả có bounding box lưu trong `output/`

---

##  Nhận diện Video

```bash
python Detection/inference.py --model model/best.pt --source "Video Project.mp4" --imgsz 1280 --conf 0.2
```

### Tùy chọn
- `--source` - Đường dẫn video
- `--imgsz` - Kích thước xử lý (mặc định 1280)
- `--conf` - Ngưỡng tự tin (mặc định 0.2)

### Ví dụ
```bash
# Nhận diện video với cấu hình tối ưu
python Detection/inference.py --model model/best.pt --source "Video Project.mp4"  --imgsz 1280 --conf 0.25

# Nhận diện nhanh
python Detection/inference.py --model model/best.pt --source "Video Project.mp4"  --imgsz 640 --conf 0.3
```

---

##  Nhận diện Webcam

```bash
python Detection/webcam.py
```

### Tùy chọn
- `--camera 0` - Chọn webcam theo index (mặc định 0)
- `--conf 0.25` - Tuù chọn confident threshold (0-1, mặc định 0.25)
- `--imgsz 640` - Kích thước xử lý, tăng lên 832/1024 nếu biển báo nhỏ nhưng sẽ chậm hơn
- `--width 1280 --height 720` - Độ phân giải webcam mong muốn
- `--skip 1` - Bỏ qua frame giữa các lần nhận diện để tăng FPS
- `--save` - Lưu video kết quả vào `output/`
- `--enhance`, `--clahe`, `--sharpen` - Tăng cường ảnh khi camera tối/mờ

### Ví dụ
```bash
# Chạy webcam mặc định
python Detection/webcam.py

# Nếu máy có nhiều camera
python Detection/webcam.py --camera 1

# Lưu video kết quả
python Detection/webcam.py --save

# Tăng khả năng bắt biển báo nhỏ/xa
python Detection/webcam.py --imgsz 1024 --conf 0.2 --enhance --sharpen

# Chạy nhanh hơn trên máy yếu
python Detection/webcam.py --imgsz 640 --skip 1
```

Trong cửa sổ webcam:
- Nhấn `q` để thoát
- Nhấn `s` để chụp ảnh màn hình vào `output/`

---

##  Kết quả

- **Ảnh:** Lưu trong `output/` với bounding box, tên biển báo, độ tự tin
- **Video:** In trực tiếp lên video hoặc lưu video đầu ra
- **Webcam:** Hiển thị trực tiếp, có thể chụp ảnh hoặc lưu video kết quả

---

## Danh mục biển báo (29 loại)

Các nhóm biển báo được phát hiện:
- **Cấm**: Cấm đỗ, cấm quay đầu, cấm rẽ, cấm xe máy/ô tô...
- **Cảnh báo**: Nguy hiểm, đi chậm...
- **Giới hạn**: Tốc độ, trọng tải, chiều cao...
- **Chỉ dẫn**: Phương hướng, làn xe được phép...
- **Khác**: Các biển báo khác

---

**Mô hình:** YOLOv8 (model/best.pt)
