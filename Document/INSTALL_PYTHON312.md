# 🐍 Hướng Dẫn Cài Đặt Python 3.12

## ✅ Cài Đặt Nhanh

### **1️⃣ Windows - Có GPU NVIDIA**
```bash
# Tạo virtual environment
python -m venv venv
venv\Scripts\activate

# Cài dependencies (CUDA 12.1)
pip install -r requirements.txt

# Test
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

### **2️⃣ Windows - Không Có GPU (CPU Only)**
```bash
# Tạo virtual environment
python -m venv venv
venv\Scripts\activate

# Cài dependencies (CPU)
pip install -r requirements-cpu.txt

# Test
python -c "import torch; print(f'PyTorch {torch.__version__}')"
```

### **3️⃣ Linux/Mac - Có GPU NVIDIA**
```bash
# Tạo virtual environment
python3.12 -m venv venv
source venv/bin/activate

# Cài dependencies (CUDA 12.1)
pip install -r requirements.txt

# Test
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

### **4️⃣ Linux/Mac - CPU Only**
```bash
# Tạo virtual environment
python3.12 -m venv venv
source venv/bin/activate

# Cài dependencies (CPU)
pip install -r requirements-cpu.txt
```

---

## ⚡ Cài Đặt Nhanh (One-Liner)

### **GPU (CUDA 12.1)**
```bash
python -m venv venv && venv\Scripts\activate && pip install -r requirements.txt
```

### **CPU Only**
```bash
python -m venv venv && venv\Scripts\activate && pip install -r requirements-cpu.txt
```

---

## 🧪 Kiểm Tra Cài Đặt

### **Python Version**
```bash
python --version
# Output: Python 3.12.x
```

### **PyTorch Version**
```bash
python -c "import torch; print(torch.__version__)"
# Output: 2.4.0 hoặc cao hơn
```

### **CUDA Support (nếu có GPU)**
```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"CPU\"}')"
```

### **YOLO Model**
```bash
python -c "from ultralytics import YOLO; model = YOLO('model02/best28121.pt'); print('✓ Model loaded')"
```

### **OpenCV**
```bash
python -c "import cv2; print(f'OpenCV {cv2.__version__}')"
```

---

## ❌ Lỗi Phổ Biến & Giải Pháp

### **Error 1: CUDA Error**
```
RuntimeError: CUDA is not available
```
**Giải Pháp:** Dùng `requirements-cpu.txt` hoặc cài CUDA 12.1 driver

### **Error 2: ImportError: torch**
```
ModuleNotFoundError: No module named 'torch'
```
**Giải Pháp:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### **Error 3: Pillow Unicode Error**
```
UnicodeDecodeError with Vietnamese text
```
**Giải Pháp:**
```bash
pip install --upgrade Pillow
```

### **Error 4: OpenCV Codec Error**
```
[ERROR:0] OpenH264 encoder is not available
```
**Giải Pháp:**
```bash
pip install opencv-contrib-python
```

---

## 📊 Phiên Bản Python Khuyên

| Python | Status | Khuyên |
|--------|--------|-------|
| 3.9 | ✅ Cũ | Tốt |
| 3.10 | ✅ Cũ | Rất tốt |
| 3.11 | ✅ Hiện Tại | **Khuyên nhất** |
| 3.12 | ✅ Mới | Tốt (file này) |
| 3.13 | ❌ Quá Mới | Không khuyên |

---

## 🚀 Chạy Dự Án

### **Sau khi cài xong:**

```bash
# 1. Kích hoạt environment
venv\Scripts\activate

# 2. Chạy GUI chính
python main.py

# 3. Hoặc chạy từng module
python Detection/image.py --image_path test.jpg
python Detection/inference.py --source video.mp4
python Detection/webcam.py
```

---

## 💾 Cập Nhật File

- ✅ `requirements.txt` → Python 3.12 + CUDA 12.1
- ✅ `requirements-cpu.txt` → Python 3.12 + CPU only
- ✅ Tất cả dependencies tương thích Python 3.12

---

**Tác Giả:** GitHub Copilot  
**Ngày Cập Nhật:** 01/06/2026
