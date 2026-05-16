# Configuration cho Traffic Sign Detection
import os

# Đường dẫn mô hình
MODEL_PATH = 'yolov8m.pt'  # Hoặc yolov8s.pt, yolov8l.pt, yolov8x.pt
DEFAULT_VIDEO_PATH = r'D:\videos\traffic.mp4'
# Tham số detection
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45
MAX_DETECTIONS = 100

# Camera/Video
CAMERA_INDEX = 0  # 0 là camera mặc định
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS = 30

# OCR Settings
PYTESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows path
# Cho Linux: PYTESSERACT_PATH = '/usr/bin/tesseract'

# Màu cho bounding box
BOX_COLOR = (0, 255, 0)  # BGR format (Green)
BOX_THICKNESS = 2
TEXT_COLOR = (255, 255, 255)  # White
TEXT_SCALE = 0.7

# Thư mục lưu trữ
OUTPUT_DIR = 'output'
LOG_DIR = 'logs'

# Tạo thư mục nếu chưa tồn tại
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Các lớp biển báo phổ biến
TRAFFIC_SIGN_CLASSES = {
    0: 'Stop Sign',
    1: 'Speed Limit',
    2: 'Yield',
    3: 'No Entry',
    4: 'Priority Road',
    5: 'No Parking',
    6: 'No Left Turn',
    7: 'One Way',
    8: 'Pedestrian Crossing',
    9: 'School Zone'
}
