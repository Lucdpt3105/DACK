"""
Tiện ích bổ sung
"""
import cv2
import numpy as np
import os
from datetime import datetime


class FrameRecorder:
    """Lưu video từ detections"""
    
    def __init__(self, output_path="output.mp4", fps=30, frame_size=(1280, 720)):
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size
        self.writer = None
        self.initialize_writer()
    
    def initialize_writer(self):
        """Khởi tạo video writer"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, self.frame_size)
    
    def write_frame(self, frame):
        """Ghi frame vào video"""
        if frame.shape[1] != self.frame_size[0] or frame.shape[0] != self.frame_size[1]:
            frame = cv2.resize(frame, self.frame_size)
        self.writer.write(frame)
    
    def release(self):
        """Kết thúc ghi video"""
        if self.writer:
            self.writer.release()


class DetectionLogger:
    """Ghi log detection results"""
    
    def __init__(self, log_file="detections.log"):
        self.log_file = log_file
    
    def log_detections(self, frame_id, detections):
        """
        Ghi detection vào file
        
        Args:
            frame_id: ID của frame
            detections: Danh sách detection
        """
        with open(self.log_file, 'a') as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"\n[{timestamp}] Frame {frame_id}\n")
            
            if not detections:
                f.write("  No detections\n")
            else:
                for i, det in enumerate(detections):
                    f.write(f"  [{i+1}] {det['class_name']}\n")
                    f.write(f"      Confidence: {det['confidence']:.4f}\n")
                    f.write(f"      BBox: {det['bbox']}\n")
                    f.write(f"      Text: {det['text']}\n")


def resize_frame(frame, max_width=1280, max_height=720):
    """
    Resize frame nếu quá lớn
    
    Args:
        frame: OpenCV frame
        max_width: Chiều rộng tối đa
        max_height: Chiều cao tối đa
        
    Returns:
        Frame đã resize
    """
    height, width = frame.shape[:2]
    
    if width <= max_width and height <= max_height:
        return frame
    
    scale = min(max_width / width, max_height / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)


def create_comparison_image(original, processed, title="Comparison"):
    """
    Tạo ảnh so sánh (trước và sau)
    
    Args:
        original: Ảnh gốc
        processed: Ảnh đã xử lý
        title: Tiêu đề
        
    Returns:
        Ảnh so sánh
    """
    # Đảm bảo cùng kích thước
    h1, w1 = original.shape[:2]
    h2, w2 = processed.shape[:2]
    
    if h1 != h2 or w1 != w2:
        processed = cv2.resize(processed, (w1, h1))
    
    # Chuyển processed sang BGR nếu là grayscale
    if len(processed.shape) == 2:
        processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    
    # Ghép ngang
    comparison = np.hstack([original, processed])
    
    # Thêm tiêu đề
    cv2.putText(comparison, "Original", (10, 30),
              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(comparison, "Processed", (w1 + 10, 30),
              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return comparison


def get_system_info():
    """Lấy thông tin hệ thống"""
    import platform
    import torch
    
    info = {
        'platform': platform.system(),
        'python_version': platform.python_version(),
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else 'N/A',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    return info


def print_system_info():
    """In thông tin hệ thống"""
    info = get_system_info()
    
    print("\n" + "=" * 50)
    print("SYSTEM INFORMATION")
    print("=" * 50)
    print(f"Platform: {info['platform']}")
    print(f"Python Version: {info['python_version']}")
    print(f"PyTorch Version: {info['torch_version']}")
    print(f"CUDA Available: {info['cuda_available']}")
    if info['cuda_available']:
        print(f"CUDA Version: {info['cuda_version']}")
    print(f"Device: {info['device'].upper()}")
    print("=" * 50 + "\n")
