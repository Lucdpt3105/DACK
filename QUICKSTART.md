# QUICK START GUIDE 🚀

Hướng dẫn nhanh bắt đầu với Traffic Sign Detection System

## ⚡ Bắt đầu nhanh (5 phút)

### Bước 1: Cài đặt Dependencies

```bash
pip install -r requirements.txt
```

### Bước 2: Cài đặt Tesseract-OCR

**Windows:**
- Download: https://github.com/UB-Mannheim/tesseract/wiki
- Chạy installer
- Cập nhật đường dẫn trong `config.py`

**Linux:**
```bash
sudo apt-get install tesseract-ocr
```

### Bước 3: Chạy Setup (Optional)

```bash
python traffic_sign_detection/setup.py
```

### Bước 4: Chạy Detection

```bash
cd traffic_sign_detection
python main.py -c
```

## 💻 Lệnh cơ bản

```bash
# Từ camera thực tế
python main.py -c

# Từ file video
python main.py -v /path/to/video.mp4

# Từ file ảnh
python main.py -i /path/to/image.jpg

# Batch xử lý thư mục ảnh
python batch.py --images /path/to/images

# Xử lý video thành frames
python batch.py --video /path/to/video.mp4

# Advanced demo với metrics
python advanced.py

# Kiểm tra cài đặt
python test.py

# Benchmark mô hình
python advanced.py --benchmark /path/to/image.jpg
```

## 📂 Cấu trúc thư mục

```
traffic_sign_detection/
├── main.py           ← Chạy từ đây
├── test.py           ← Kiểm tra cài đặt
├── setup.py          ← Cài đặt ban đầu
├── batch.py          ← Batch processing
├── advanced.py       ← Demo nâng cao
├── config.py         ← Cấu hình
├── detector.py       ← Lớp detection
├── ocr_utils.py      ← OCR functions
├── utils.py          ← Utility functions
├── USAGE.md          ← Hướng dẫn chi tiết
└── output/           ← Kết quả output
```

## 🎮 Phím tắt

| Phím | Chức năng |
|------|----------|
| `q` | Thoát |
| `s` | Lưu ảnh |
| `r` | Reset counter |

## ⚙️ Cấu hình quan trọng

Chỉnh sửa `config.py`:

```python
# Model (nhanh → chính xác)
MODEL_PATH = 'yolov8n.pt'  # nhanh
MODEL_PATH = 'yolov8m.pt'  # balanced
MODEL_PATH = 'yolov8l.pt'  # chính xác

# Tesseract path (Windows)
PYTESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Ngưỡng detection
CONFIDENCE_THRESHOLD = 0.5  # 0.0-1.0
```

## 🐛 Troubleshooting nhanh

**Camera không mở?**
```python
# Trong config.py, thử:
CAMERA_INDEX = 1  # hoặc 2, 3...
```

**OCR lỗi?**
```bash
# Cài Tesseract
# Windows: https://github.com/UB-Mannheim/tesseract/wiki
# Linux: sudo apt-get install tesseract-ocr
```

**Chậm?**
```python
# Dùng model nhỏ hơn
MODEL_PATH = 'yolov8s.pt'  # Nhanh hơn

# Hoặc giảm độ phân giải
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
```

## 📊 Output

Chương trình tạo ra:
- **Ảnh**: `output/detection_*.jpg`
- **Báo cáo**: `batch_output_*/report.txt`
- **Log**: `logs/traffic_sign.log`

## 📝 Ví dụ code đơn giản

```python
from detector import TrafficSignDetector
import cv2

# Load model
detector = TrafficSignDetector('yolov8m.pt')

# Read image
img = cv2.imread('sign.jpg')

# Detect
detections = detector.detect(img)

# Draw
result = detector.draw_detections(img, detections)

# Show
cv2.imshow('Result', result)
cv2.waitKey(0)

# Print results
for d in detections:
    print(f"{d['class_name']}: {d['text']}")
```

## 🎯 Các bước tiếp theo

1. ✓ Cài đặt xong
2. → Thử chạy từ camera
3. → Test với video/ảnh
4. → Cấu hình cho dự án của bạn
5. → Fine-tune mô hình (advanced)

## 🆘 Cần giúp?

1. Kiểm tra `test.py` để debug
2. Xem `USAGE.md` để hướng dẫn chi tiết
3. Kiểm tra `logs/` folder

## 🌟 Mẹo tối ưu

- **Nhanh nhất**: `yolov8n.pt` + `FRAME_WIDTH=640`
- **Cân bằng**: `yolov8m.pt` + `FRAME_WIDTH=1280`
- **Chính xác nhất**: `yolov8l.pt` + `FRAME_WIDTH=1920`

## 📦 Model sizes

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| n | ~3MB | ⚡⚡⚡ | ⭐⭐ |
| s | ~22MB | ⚡⚡ | ⭐⭐⭐ |
| m | ~49MB | ⚡ | ⭐⭐⭐⭐ |
| l | ~99MB | ⚠️ | ⭐⭐⭐⭐⭐ |
| x | ~168MB | 🚫 | 🌟🌟🌟 |

---

**Happy Detection! 🎉**
