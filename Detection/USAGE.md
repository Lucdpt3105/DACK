# Traffic Sign Detection - Project Structure

## 📁 Cấu trúc thư mục

```
DACK/
├── README.md                          # Hướng dẫn chính
├── requirements.txt                   # Dependencies
└── traffic_sign_detection/
    ├── main.py                        # Script chính
    ├── test.py                        # Script kiểm tra
    ├── config.py                      # Cấu hình
    ├── detector.py                    # Lớp phát hiện YOLOv8
    ├── ocr_utils.py                   # Tiện ích OCR
    ├── utils.py                       # Tiện ích bổ sung
    └── output/                        # Thư mục lưu kết quả
        └── (ảnh kết quả)
```

## 🚀 Cài đặt & Sử dụng

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 2. Cài đặt Tesseract OCR

**Windows:**
- Download: https://github.com/UB-Mannheim/tesseract/wiki
- Mở file `tesseract-ocr-w64-setup-v5.x.exe` và cài đặt
- Cập nhật đường dẫn trong `config.py`:
```python
PYTESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install tesseract-ocr
```

**Mac:**
```bash
brew install tesseract
```

### 3. Chạy chương trình

**Từ Camera (mặc định):**
```bash
cd traffic_sign_detection
python main.py
# Hoặc
python main.py -c
```

**Từ File Video:**
```bash
python main.py -v video.mp4
```

**Từ File Ảnh:**
```bash
python main.py -i image.jpg
```

### 4. Kiểm tra cài đặt

```bash
python test.py
```

## 📊 Tính năng

### ✨ Phát hiện biển báo
- Sử dụng YOLOv8 Medium (yolov8m.pt)
- Real-time detection từ camera
- Support video files
- Support ảnh tĩnh

### 🔍 Nhận dạng chữ
- Sử dụng Tesseract OCR
- Đọc số tốc độ từ biển báo
- Đọc text nhiều dòng
- Tiền xử lý ảnh tự động

### 📦 Bounding Box
- Vẽ bbox xung quanh biển báo
- Hiển thị class name
- Hiển thị confidence score
- Hiển thị text được đọc

## ⌨️ Phím tắt

| Phím | Chức năng |
|------|----------|
| `q` | Thoát chương trình |
| `s` | Lưu ảnh hiện tại |
| `r` | Reset counter |

## ⚙️ Cấu hình

Chỉnh sửa file `config.py` để tùy chỉnh:

```python
# Mô hình
MODEL_PATH = 'yolov8m.pt'  # Hoặc yolov8s, yolov8l, yolov8x

# Ngưỡng detection
CONFIDENCE_THRESHOLD = 0.5  # 0.0 - 1.0

# Camera
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Màu bounding box
BOX_COLOR = (0, 255, 0)  # BGR format
```

## 📝 Ví dụ sử dụng

### Sử dụng DetectorTrafficSign trong code

```python
from detector import TrafficSignDetector
import cv2

# Khởi tạo
detector = TrafficSignDetector(model_path='yolov8m.pt')

# Đọc ảnh
frame = cv2.imread('image.jpg')

# Phát hiện
detections = detector.detect(frame)

# Vẽ kết quả
frame_result = detector.draw_detections(frame, detections)

# Hiển thị
cv2.imshow('Result', frame_result)
cv2.waitKey(0)

# In kết quả
for det in detections:
    print(f"Sign: {det['class_name']}")
    print(f"Confidence: {det['confidence']:.2f}")
    print(f"Text: {det['text']}")
```

### Sử dụng OCR Reader

```python
from ocr_utils import OCRReader, extract_sign_region
import cv2

# Khởi tạo
ocr = OCRReader()

# Đọc ảnh
frame = cv2.imread('sign.jpg')

# Đọc text
text = ocr.read_text(frame)
print(f"Text: {text}")

# Đọc tốc độ
speed = ocr.read_speed_limit(frame)
print(f"Speed: {speed} km/h")
```

## 🔧 Troubleshooting

### Camera không mở
- Kiểm tra camera kết nối
- Thử đổi `CAMERA_INDEX` thành `1` hoặc `2`

### OCR không hoạt động
- Kiểm tra Tesseract đã cài đặt
- Cập nhật `PYTESSERACT_PATH` đúng

### Detection chậm
- Dùng model nhỏ hơn: `yolov8s.pt`
- Giảm `FRAME_WIDTH` và `FRAME_HEIGHT`

### CUDA out of memory
- Chuyển sang CPU bằng cách giảm batch size

## 🎓 Các model YOLOv8 có sẵn

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| yolov8n | ~3MB | Nhanh | Thấp |
| yolov8s | ~22MB | Nhanh | Trung bình |
| yolov8m | ~49MB | Trung bình | Cao |
| yolov8l | ~99MB | Chậm | Rất cao |
| yolov8x | ~168MB | Chậm | Cực cao |

## 📚 Tài liệu tham khảo

- YOLOv8: https://github.com/ultralytics/ultralytics
- OpenCV: https://opencv.org/
- Tesseract OCR: https://github.com/tesseract-ocr/tesseract
- PyTorch: https://pytorch.org/

## 📄 License

MIT License

## 👨‍💻 Tác giả

Traffic Sign Detection System - Vietnamese Edition
