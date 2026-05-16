<<<<<<< HEAD
# 🚗 Traffic Sign Detection & Recognition System

Phần mềm phát hiện, nhận dạng và đọc biển báo giao thông sử dụng **YOLOv8**, **OpenCV**, và **Tesseract OCR**.

## ✨ Tính năng chính

- 🎯 **Phát hiện biển báo** thời gian thực từ camera/video/ảnh
- 🔍 **Nhận dạng loại biển báo** (Stop, Speed Limit, Yield, v.v.)
- 📝 **OCR - Đọc text/số** trên biển báo (giới hạn tốc độ, tên đường, v.v.)
- 📦 **Bounding box** hiển thị chi tiết phát hiện
- 🎬 **Hỗ trợ** camera thực tế, file video, ảnh tĩnh
- ⚡ **GPU support** - Tối ưu hóa với CUDA

## 🛠️ Công nghệ sử dụng

| Công nghệ | Mục đích |
|-----------|---------|
| **YOLOv8** | Phát hiện object (biển báo) |
| **OpenCV** | Xử lý ảnh & video |
| **Tesseract OCR** | Nhận dạng text/số |
| **PyTorch** | Deep learning framework |
| **Python 3.8+** | Ngôn ngữ lập trình |

## 📋 Yêu cầu

- Python 3.8 hoặc cao hơn
- pip (Package installer)
- Camera (nếu dùng camera live) hoặc file ảnh/video
- Tesseract-OCR (cài đặt riêng)

## 🚀 Cài đặt nhanh

### 1️⃣ Cài đặt Python packages

```bash
pip install -r requirements.txt
```

### 2️⃣ Cài đặt Tesseract-OCR

**Windows:**
1. Tải từ: https://github.com/UB-Mannheim/tesseract/wiki
2. Chạy `tesseract-ocr-w64-setup-v5.x.exe`
3. Cập nhật đường dẫn trong `traffic_sign_detection/config.py`:
```python
PYTESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

**Linux/Mac:**
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# Mac
brew install tesseract
```

### 3️⃣ Kiểm tra cài đặt

```bash
cd traffic_sign_detection
python test.py
```

## 💻 Sử dụng

### Từ Camera (Real-time)

```bash
cd traffic_sign_detection
python main.py -c
```

### Từ File Video

```bash
python main.py -v path/to/video.mp4
```

### Từ File Ảnh

```bash
python main.py -i path/to/image.jpg
```

### Batch Processing (Xử lý nhiều ảnh)

```bash
python batch.py --images path/to/images_folder
python batch.py --video path/to/video.mp4
```

## ⌨️ Phím tắt trong chương trình

| Phím | Chức năng |
|------|----------|
| `q` | Thoát chương trình |
| `s` | Lưu ảnh kết quả |
| `r` | Reset bộ đếm frame |

## ⚙️ Cấu hình

Chỉnh sửa `traffic_sign_detection/config.py`:

```python
# Mô hình (yolov8n/s/m/l/x)
MODEL_PATH = 'yolov8m.pt'

# Ngưỡng tin cậy
CONFIDENCE_THRESHOLD = 0.5

# Độ phân giải camera
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Màu bounding box (BGR)
BOX_COLOR = (0, 255, 0)
```

## 📊 Kết quả output

Chương trình sẽ tạo:
- **Ảnh kết quả** với bounding box: `output/detection_YYYYMMDD_HHMMSS.jpg`
- **Video kết quả**: `output.mp4`
- **Báo cáo batch**: `batch_output_*/report.txt`

## 📝 Ví dụ sử dụng trong code

```python
from traffic_sign_detection.detector import TrafficSignDetector
import cv2

# Khởi tạo detector
detector = TrafficSignDetector(model_path='yolov8m.pt')

# Đọc ảnh
frame = cv2.imread('image.jpg')

# Phát hiện
detections = detector.detect(frame)

# Vẽ kết quả
result = detector.draw_detections(frame, detections)

# Hiển thị
cv2.imshow('Result', result)
cv2.waitKey(0)

# In chi tiết
for det in detections:
    print(f"🔍 Sign: {det['class_name']}")
    print(f"   Confidence: {det['confidence']:.2%}")
    print(f"   Text: {det['text']}")
    print()
```

## 📁 Cấu trúc dự án

```
DACK/
├── README.md
├── requirements.txt
└── traffic_sign_detection/
    ├── main.py              # Script chính
    ├── test.py              # Kiểm tra cài đặt
    ├── batch.py             # Batch processing
    ├── config.py            # Cấu hình
    ├── detector.py          # Lớp phát hiện YOLOv8
    ├── ocr_utils.py         # Utilities OCR
    ├── utils.py             # Utilities khác
    ├── USAGE.md             # Hướng dẫn chi tiết
    └── output/              # Thư mục kết quả
```

## 🎓 Các mô hình YOLOv8

| Model | Tốc độ | Độ chính xác | Dung lượng |
|-------|--------|-------------|-----------|
| yolov8n | ⚡⚡⚡ | ⭐⭐ | ~3MB |
| yolov8s | ⚡⚡ | ⭐⭐⭐ | ~22MB |
| yolov8m | ⚡ | ⭐⭐⭐⭐ | ~49MB |
| yolov8l | ⚠️ | ⭐⭐⭐⭐⭐ | ~99MB |
| yolov8x | ⚠️ | 🌟🌟🌟 | ~168MB |

## 🔧 Troubleshooting

**❌ Camera không mở?**
- Kiểm tra camera kết nối
- Thay đổi `CAMERA_INDEX` từ 0 thành 1, 2, ...

**❌ OCR không hoạt động?**
- Kiểm tra Tesseract đã cài đặt
- Cập nhật `PYTESSERACT_PATH` đúng

**❌ Detection chậm?**
- Dùng model nhỏ hơn (yolov8s)
- Giảm độ phân giải frame

**❌ CUDA out of memory?**
- Chuyển sang CPU bằng config
- Giảm kích thước frame

## 📚 Tài liệu tham khảo

- [YOLOv8 Documentation](https://github.com/ultralytics/ultralytics)
- [OpenCV Documentation](https://opencv.org/)
- [Tesseract-OCR](https://github.com/tesseract-ocr/tesseract)
- [PyTorch](https://pytorch.org/)

## 📝 Log và Debug

Chương trình ghi log chi tiết vào:
- Console (realtime)
- File: `logs/traffic_sign.log`

## 🤝 Đóng góp

Nếu bạn có ý kiến cải thiện, hãy tạo issue hoặc pull request!

## 📄 License

MIT License - Tự do sử dụng cho mục đích học tập và thương mại

## 👨‍💻 Tác giả

Phát triển bởi AI Assistant - May 2026

---

### ⭐ Hãy star dự án nếu bạn thấy hữu ích!

**Happy Coding!** 🎉
=======
# DACK
D23CQPTUD01-N 
>>>>>>> c599e0ec676c45661bb4ae0f5a266bc634d3e6c0
