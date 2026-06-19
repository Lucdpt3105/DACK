"""
🚦 Vietnamese Traffic Sign Detection - Web App (FastAPI)
========================================================
Phiên bản web của ứng dụng nhận diện biển báo giao thông Việt Nam.
Không cần cài đặt phức tạp, chỉ cần pip install và chạy.

Usage:
    python run_webapp.py
    # Sau đó mở trình duyệt http://localhost:8000
"""

import os
import sys
import uuid
import json
from pathlib import Path

# Thêm thư mục gốc vào path để import được module Detection
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import uvicorn

from Detection.image import TrafficSignDetector, VIE_NAMES
from Detection.inference import draw_detections, enhance_image, apply_clahe, sharpen_image

import cv2
import numpy as np
from ultralytics import YOLO

# ============================================================
# Khởi tạo App
# ============================================================
app = FastAPI(
    title="🚦 Nhận Diện Biển Báo Giao Thông",
    description="Web app nhận diện biển báo giao thông Việt Nam sử dụng YOLO",
    version="2.0.0"
)

# ============================================================
# Cấu hình đường dẫn
# ============================================================
BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
UPLOAD_DIR = BASE_DIR / "uploads"
MODEL_PATH = BASE_DIR.parent / "model" / "best.pt"
CLASSES_PATH = BASE_DIR.parent / "model" / "classes.txt"
OUTPUT_DIR = BASE_DIR.parent / "output" / "bienbao"

# Tạo các thư mục cần thiết
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Static files & Templates
# ============================================================
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# ============================================================
# Khởi tạo Model (lazy loading)
# ============================================================
model = None
detector = None

def get_model():
    global model
    if model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(status_code=500, detail=f"Không tìm thấy mô hình tại: {MODEL_PATH}")
        print(f"🔄 Đang tải mô hình YOLO: {MODEL_PATH}...")
        model = YOLO(str(MODEL_PATH))
        print(f"✅ Tải thành công! Mô hình có: {len(model.names)} classes.")
    return model

def get_detector():
    global detector
    if detector is None:
        if not MODEL_PATH.exists():
            raise HTTPException(status_code=500, detail=f"Không tìm thấy mô hình tại: {MODEL_PATH}")
        detector = TrafficSignDetector(model_path=str(MODEL_PATH))
    return detector

def load_classes():
    """Đọc danh sách classes từ file."""
    if CLASSES_PATH.exists():
        with open(CLASSES_PATH, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    return list(VIE_NAMES.keys())

def allowed_file(filename: str) -> bool:
    """Kiểm tra định dạng file cho phép."""
    ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ext = Path(filename).suffix.lower()
    return ext in ALLOWED_EXTENSIONS

def allowed_video(filename: str) -> bool:
    """Kiểm tra định dạng video cho phép."""
    ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    ext = Path(filename).suffix.lower()
    return ext in ALLOWED_EXTENSIONS

# ============================================================
# Trang chủ
# ============================================================
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html", 
        {
            "request": request,
            "classes": VIE_NAMES,
            "total_classes": len(VIE_NAMES)
        }
    )

# ============================================================
# API: Nhận diện ảnh (Upload + Detect)
# ============================================================
@app.post("/api/detect/image")
async def detect_image(
    file: UploadFile = File(...),
    conf: float = Form(0.25),
    enhance: bool = Form(False),
    clahe: bool = Form(False),
    sharpen: bool = Form(False)
):
    """Nhận diện biển báo từ ảnh upload."""
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="Định dạng file không hỗ trợ. Chấp nhận: jpg, jpeg, png, bmp, webp")
    
    # Lưu file upload
    file_ext = Path(file.filename).suffix
    file_id = str(uuid.uuid4())
    upload_path = UPLOAD_DIR / f"{file_id}{file_ext}"
    
    content = await file.read()
    with open(upload_path, "wb") as f:
        f.write(content)
    
    try:
        # Đọc ảnh
        img = cv2.imread(str(upload_path))
        if img is None:
            raise HTTPException(status_code=400, detail="Không thể đọc file ảnh")
        
        # Tiền xử lý (enhancement)
        img_processed = img.copy()
        if enhance:
            img_processed = enhance_image(img_processed)
        if clahe:
            img_processed = apply_clahe(img_processed)
        if sharpen:
            img_processed = sharpen_image(img_processed)
        
        # Chạy inference
        yolo_model = get_model()
        results = yolo_model(img_processed, conf=conf, verbose=False)
        
        # Vẽ kết quả lên ảnh gốc
        result_img = draw_detections(img, results, yolo_model.names)
        
        # Lưu ảnh kết quả
        output_filename = f"detected_{file_id}.jpg"
        output_path = OUTPUT_DIR / output_filename
        cv2.imwrite(str(output_path), result_img)
        
        # Tạo response JSON
        detections = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            class_name = str(yolo_model.names[cls_id]) if cls_id < len(yolo_model.names) else f"class_{cls_id}"
            conf_val = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            detections.append({
                "class_name": class_name,
                "class_name_vi": VIE_NAMES.get(class_name, class_name),
                "confidence": round(conf_val, 4),
                "bbox": {
                    "x1": round(x1, 1),
                    "y1": round(y1, 1),
                    "x2": round(x2, 1),
                    "y2": round(y2, 1)
                }
            })
        
        return {
            "success": True,
            "file_id": file_id,
            "original_filename": file.filename,
            "output_image": f"/api/output/{output_filename}",
            "total_signs": len(detections),
            "detections": detections
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# API: Nhận diện Video (Upload và xử lý)
# ============================================================
@app.post("/api/detect/video")
async def detect_video(
    file: UploadFile = File(...),
    conf: float = Form(0.22),
    imgsz: int = Form(1280),
    enhance: bool = Form(False)
):
    """Nhận diện biển báo từ video upload."""
    if not allowed_video(file.filename):
        raise HTTPException(status_code=400, detail="Định dạng video không hỗ trợ. Chấp nhận: mp4, avi, mov, mkv, webm")
    
    # Lưu file upload
    file_ext = Path(file.filename).suffix
    file_id = str(uuid.uuid4())
    upload_path = UPLOAD_DIR / f"{file_id}{file_ext}"
    
    content = await file.read()
    with open(upload_path, "wb") as f:
        f.write(content)
    
    try:
        yolo_model = get_model()
        
        cap = cv2.VideoCapture(str(upload_path))
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Không thể mở file video")
        
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Tạo video output
        output_video = f"detected_{file_id}.mp4"
        output_path = OUTPUT_DIR / output_video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
        
        frame_count = 0
        processed_count = 0
        all_detections = {}
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Xử lý enhancement
            frame_proc = frame.copy()
            if enhance:
                frame_proc = enhance_image(frame_proc)
            
            # Predict mỗi frame thứ 3 (skip frames để nhanh hơn)
            if frame_count % 3 == 0:
                results = yolo_model(frame_proc, conf=conf, imgsz=imgsz, verbose=False)
                annotated = draw_detections(frame, results, yolo_model.names)
                processed_count += 1
                
                # Thu thập thông tin
                frame_detections = []
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    class_name = str(yolo_model.names[cls_id])
                    frame_detections.append({
                        "class_name": class_name,
                        "class_name_vi": VIE_NAMES.get(class_name, class_name),
                        "confidence": round(float(box.conf[0]), 4)
                    })
                if frame_detections:
                    all_detections[f"frame_{frame_count}"] = frame_detections
            else:
                annotated = frame
            
            writer.write(annotated)
        
        cap.release()
        writer.release()
        
        return {
            "success": True,
            "file_id": file_id,
            "original_filename": file.filename,
            "output_video": f"/api/output/{output_video}",
            "total_frames": frame_count,
            "processed_frames": processed_count,
            "detections_summary": all_detections
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# API: Lấy ảnh/video kết quả
# ============================================================
@app.get("/api/output/{filename}")
async def get_output(filename: str):
    """Lấy file kết quả (ảnh hoặc video)."""
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File không tồn tại")
    return FileResponse(str(file_path))


# ============================================================
# API: Danh sách classes
# ============================================================
@app.get("/api/classes")
async def get_classes():
    """Lấy danh sách các loại biển báo."""
    return {
        "total": len(VIE_NAMES),
        "classes": [{"code": k, "name": v} for k, v in VIE_NAMES.items()]
    }


# ============================================================
# API: Thông tin model
# ============================================================
@app.get("/api/info")
async def get_info():
    """Lấy thông tin về model."""
    yolo_model = get_model()
    return {
        "model": "YOLOv8",
        "model_path": str(MODEL_PATH),
        "num_classes": len(yolo_model.names),
        "classes": list(yolo_model.names.values())
    }


# ============================================================
# Chạy app
# ============================================================
if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════╗
    ║   🚦 Vietnamese Traffic Sign Detection - WEB    ║
    ║   Nhận diện biển báo giao thông Việt Nam       ║
    ║                                                  ║
    ║   📍 http://localhost:8000                       ║
    ║                                                  ║
    ║   📸 Upload ảnh → Nhận diện ngay                ║
    ║   🎥 Upload video → Xử lý tự động               ║
    ╚══════════════════════════════════════════════════╝
    """)
    uvicorn.run("web_app:app", host="0.0.0.0", port=8000, reload=True)