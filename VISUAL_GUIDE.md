# 🎬 Traffic Sign Detection System - Visual Guide

Hướng dẫn trực quan về cách sử dụng hệ thống

---

## 📁 Project Structure

```
d:\DACK\
│
├── 📄 README.md                    ← Hướng dẫn chính
├── 📄 QUICKSTART.md                ← Bắt đầu nhanh (5 phút)
├── 📄 INSTALLATION.md              ← Hướng dẫn cài đặt
├── 📄 PROJECT_SUMMARY.md           ← Tóm tắt dự án
├── 📋 requirements.txt              ← Python dependencies
│
└── 📁 traffic_sign_detection/
    │
    ├── 🚀 CORE FILES
    │   ├── main.py                 ← Chạy phát hiện
    │   ├── detector.py             ← YOLOv8 detector
    │   ├── ocr_utils.py            ← Tesseract OCR
    │   └── config.py               ← Cấu hình
    │
    ├── 🛠️ UTILITY FILES
    │   ├── utils.py                ← Helper functions
    │   ├── test.py                 ← Kiểm tra cài đặt
    │   ├── setup.py                ← Setup tự động
    │   ├── batch.py                ← Batch processing
    │   └── advanced.py             ← Advanced features
    │
    ├── 📚 DOCUMENTATION
    │   └── USAGE.md                ← Hướng dẫn chi tiết
    │
    └── 📁 DATA FOLDERS
        ├── output/                 ← Ảnh kết quả
        └── logs/                   ← Log files
```

---

## 🎯 Usage Flowchart

```
START
  ↓
[Cài đặt]
  ├─ pip install -r requirements.txt
  ├─ Cài Tesseract-OCR
  └─ python setup.py (optional)
  ↓
[Chọn source]
  ├─ Camera     → python main.py -c
  ├─ Video      → python main.py -v file.mp4
  ├─ Image      → python main.py -i file.jpg
  └─ Batch      → python batch.py --images folder
  ↓
[Phát hiện]
  ├─ YOLOv8 detect signs
  ├─ Tesseract read text
  └─ Draw boxes + labels
  ↓
[Output]
  ├─ Hiển thị real-time
  ├─ Lưu ảnh (press 's')
  └─ Log results
  ↓
END
```

---

## 🎮 Example Workflows

### Workflow 1: Real-time Camera Detection

```
1. Open terminal in traffic_sign_detection folder
   $ cd traffic_sign_detection

2. Start camera detection
   $ python main.py -c
   
3. See live detection with boxes
   ┌──────────────────────────────┐
   │  🚗 Camera Feed              │
   │  ┌─────────────┐             │
   │  │ Speed Limit │ Conf: 0.95  │
   │  │  [50]       │ Text: "50"  │
   │  └─────────────┘             │
   │  FPS: 24.5 | Detections: 1   │
   └──────────────────────────────┘
   
4. Save results (press 's')
   ✓ Saved: output/detection_20260516_103045.jpg
   
5. Exit (press 'q')
```

### Workflow 2: Batch Process Images

```
1. Prepare folder with images
   my_images/
   ├── sign1.jpg
   ├── sign2.jpg
   └── sign3.jpg

2. Run batch processing
   $ python batch.py --images my_images
   
3. See progress
   [1/3] Processing: sign1.jpg
   [1/3] Processing: sign2.jpg
   [1/3] Processing: sign3.jpg
   
4. Get results
   batch_output_20260516_103045/
   ├── result_sign1.jpg
   ├── result_sign2.jpg
   ├── result_sign3.jpg
   └── report.txt
```

### Workflow 3: Process Video

```
1. Have video file ready
   traffic_video.mp4

2. Run video detection
   $ python main.py -v traffic_video.mp4
   
3. Watch processing
   Frame 100: Detections 5
   Frame 200: Detections 3
   Frame 300: Detections 4
   ...
   
4. Results saved with bounding boxes
```

---

## 📊 Output Examples

### Detection Output Format

```python
Detection result for frame:
[
    {
        'class': 1,
        'class_name': 'Speed Limit',
        'confidence': 0.9543,
        'bbox': [320, 150, 420, 250],
        'text': '50'
    },
    {
        'class': 0,
        'class_name': 'Stop Sign',
        'confidence': 0.8721,
        'bbox': [600, 200, 700, 300],
        'text': ''
    }
]
```

### Console Log Output

```
INFO:__main__:Frame 42: Phát hiện 2 biển báo
INFO:__main__:  [1] Speed Limit - Confidence: 0.95 - Text: 50
INFO:__main__:  [2] Stop Sign - Confidence: 0.87 - Text: 
INFO:__main__:Ảnh đã lưu: output/detection_20260516_103045.jpg
```

### Batch Report Output

```
TRAFFIC SIGN DETECTION - BATCH REPORT
=====================================
Timestamp: 2026-05-16 10:30:45
Total Images: 42
Total Detections: 87

DETAIL:
----

File: image_001.jpg
Detections: 2
  [1] Speed Limit
      Confidence: 0.9234
      Text: 60
  [2] No Parking
      Confidence: 0.8765
      Text: 

File: image_002.jpg
Detections: 1
  [1] Stop Sign
      Confidence: 0.9102
      Text:
...
```

---

## 🔑 Keyboard Controls

```
┌──────────────────────────────┐
│  KEYBOARD SHORTCUTS          │
├──────────────────────────────┤
│  q  → Quit program           │
│  s  → Save current frame     │
│  r  → Reset frame counter    │
└──────────────────────────────┘
```

---

## 📈 Performance Examples

### FPS Comparison (1280x720)

```
Model        CPU        GPU
─────────────────────────────
yolov8n      ~15 FPS    ~40 FPS
yolov8s      ~8 FPS     ~25 FPS
yolov8m      ~4 FPS     ~15 FPS
yolov8l      ~2 FPS     ~8 FPS
yolov8x      ~1 FPS     ~4 FPS
```

### Detection Time (per frame)

```
Resolution   yolov8m (GPU)   yolov8s (GPU)
─────────────────────────────────────────
640x480      ~40ms           ~30ms
1280x720     ~65ms           ~45ms
1920x1080    ~95ms           ~65ms
```

---

## 🧠 Detection Example

### Original Image
```
┌─────────────────────────────┐
│  Road with speed limit sign  │
│                             │
│        SPEED   STOP         │
│        LIMIT   SIGN         │
│                             │
└─────────────────────────────┘
```

### Detected Output
```
┌─────────────────────────────┐
│  Road with detected signs    │
│                             │
│    ┌──────────┐  ┌────────┐ │
│    │Speed 60  │  │STOP    │ │
│    │C: 0.94   │  │C: 0.91│ │
│    └──────────┘  └────────┘ │
│                             │
└─────────────────────────────┘

Results:
[1] Speed Limit - 60
[2] Stop Sign
```

---

## 🎛️ Configuration Examples

### Fast Detection (Speed Priority)
```python
# config.py
MODEL_PATH = 'yolov8n.pt'      # Nano model
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CONFIDENCE_THRESHOLD = 0.6
```

### Balanced (Default)
```python
MODEL_PATH = 'yolov8m.pt'      # Medium model
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
CONFIDENCE_THRESHOLD = 0.5
```

### High Accuracy (Quality Priority)
```python
MODEL_PATH = 'yolov8l.pt'      # Large model
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
CONFIDENCE_THRESHOLD = 0.4
```

---

## 📊 Class Distribution (Example)

```
Detected Traffic Signs Distribution:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Speed Limit    ████████████████████ 45%
Stop Sign      ███████████ 25%
Yield          ██████ 14%
No Entry       ███ 8%
Parking        ██ 5%
Other          █ 3%
```

---

## 🔄 Data Flow

```
┌──────────────┐
│   Camera /   │
│  Video /Img  │
└──────────────┘
       ↓
┌──────────────────────────┐
│  Frame Preprocessing     │
│  (Resize, normalize)     │
└──────────────────────────┘
       ↓
┌──────────────────────────┐
│  YOLOv8 Detection        │
│  (Bounding boxes)        │
└──────────────────────────┘
       ↓
┌──────────────────────────┐
│  Region Extraction       │
│  (Crop detection area)   │
└──────────────────────────┘
       ↓
┌──────────────────────────┐
│  Tesseract OCR           │
│  (Read text)             │
└──────────────────────────┘
       ↓
┌──────────────────────────┐
│  Visualization           │
│  (Draw boxes + labels)   │
└──────────────────────────┘
       ↓
┌──────────────────────────┐
│  Output                  │
│  (Display/Save)          │
└──────────────────────────┘
```

---

## 💾 Output Directory Structure

```
output/
├── detection_20260516_103045.jpg
├── detection_20260516_103102.jpg
├── detection_20260516_103215.jpg
└── ...

batch_output_20260516_103300/
├── result_image_001.jpg
├── result_image_002.jpg
├── report.txt
└── ...

logs/
├── traffic_sign.log
└── ...
```

---

## 🎓 Tips & Tricks

### Tip 1: Camera Selection
```python
# Nếu không chắc camera index:
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera found at index {i}")
        cap.release()
```

### Tip 2: Optimize for Speed
```python
# Use smaller model + lower resolution
MODEL_PATH = 'yolov8s.pt'
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CONFIDENCE_THRESHOLD = 0.6
```

### Tip 3: Better OCR Accuracy
```python
# Use preprocessing
from ocr_utils import OCRReader
ocr = OCRReader()
text = ocr.read_text(image, use_preprocessing=True)
```

### Tip 4: Batch Processing Large Datasets
```bash
# Process multiple batches
python batch.py --images folder1
python batch.py --images folder2
python batch.py --images folder3
```

---

## ✅ Checklist Before Using

- [ ] Python 3.8+ installed
- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] Tesseract-OCR installed
- [ ] config.py updated (if needed)
- [ ] test.py passes all checks
- [ ] Camera works (or video/image ready)

---

## 🚀 Ready to Go!

Your Traffic Sign Detection System is ready!

```bash
# Go to main folder
cd traffic_sign_detection

# Start camera detection
python main.py -c

# Or process an image
python main.py -i path/to/image.jpg

# Or batch process
python batch.py --images path/to/images
```

**Happy Detecting!** 🎉

---

*Last Updated: May 16, 2026*  
*Version: 1.0.0 - Complete & Production Ready*
