"""
Traffic Sign Detector - Sử dụng YOLOv8
"""
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from config import (
    MODEL_PATH, CONFIDENCE_THRESHOLD, IOU_THRESHOLD,
    BOX_COLOR, BOX_THICKNESS, TEXT_COLOR, TEXT_SCALE,
    TRAFFIC_SIGN_CLASSES
)
import logging
from ocr_utils import OCRReader, extract_sign_region

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrafficSignDetector:
    """Lớp phát hiện biển báo giao thông sử dụng YOLOv8"""
    
    def __init__(self, model_path=MODEL_PATH):
        """
        Khởi tạo detector
        
        Args:
            model_path: Đường dẫn đến mô hình YOLOv8
        """
        try:
            # Kiểm tra GPU
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Sử dụng device: {self.device}")
            
            # Load model
            self.model = YOLO(model_path)
            self.model.to(self.device)
            logger.info(f"Mô hình YOLOv8 đã load từ {model_path}")
            
            # Khởi tạo OCR reader
            self.ocr_reader = OCRReader()
            
        except Exception as e:
            logger.error(f"Lỗi load model: {e}")
            raise
    
    def detect(self, frame):
        """
        Phát hiện biển báo trong frame
        
        Args:
            frame: OpenCV frame (BGR)
            
        Returns:
            Danh sách detection với định dạng:
            [{
                'class': class_id,
                'class_name': class_name,
                'confidence': confidence,
                'bbox': [x1, y1, x2, y2],
                'text': text_từ_ocr
            }]
        """
        try:
            # Chạy inference
            results = self.model(frame, 
                               conf=CONFIDENCE_THRESHOLD,
                               iou=IOU_THRESHOLD,
                               verbose=False)
            
            detections = []
            
            # Xử lý kết quả
            if results[0].boxes is not None:
                boxes = results[0].boxes
                
                for box in boxes:
                    # Lấy tọa độ bbox
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Trích xuất vùng biển báo
                    sign_region = extract_sign_region(frame, [x1, y1, x2, y2])
                    
                    # Đọc text bằng OCR
                    ocr_text = ""
                    if sign_region is not None and sign_region.size > 0:
                        ocr_text = self.ocr_reader.read_text(sign_region)
                    
                    # Lấy tên lớp
                    class_name = TRAFFIC_SIGN_CLASSES.get(class_id, f"Class {class_id}")
                    
                    detection = {
                        'class': class_id,
                        'class_name': class_name,
                        'confidence': confidence,
                        'bbox': [x1, y1, x2, y2],
                        'text': ocr_text
                    }
                    
                    detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"Lỗi phát hiện: {e}")
            return []
    
    def draw_detections(self, frame, detections):
        """
        Vẽ bounding box và thông tin lên frame
        
        Args:
            frame: OpenCV frame
            detections: Danh sách detection
            
        Returns:
            Frame đã vẽ
        """
        try:
            for detection in detections:
                x1, y1, x2, y2 = detection['bbox']
                confidence = detection['confidence']
                class_name = detection['class_name']
                text = detection['text']
                
                # Vẽ bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)
                
                # Tạo label
                label = f"{class_name} ({confidence:.2f})"
                if text:
                    label += f" - {text}"
                
                # Vẽ background cho text
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 
                                          TEXT_SCALE, 1)[0]
                cv2.rectangle(frame, (x1, y1 - text_size[1] - 5),
                            (x1 + text_size[0], y1), BOX_COLOR, -1)
                
                # Vẽ text
                cv2.putText(frame, label, (x1, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE,
                          TEXT_COLOR, 1, cv2.LINE_AA)
            
            return frame
            
        except Exception as e:
            logger.error(f"Lỗi vẽ detection: {e}")
            return frame
    
    def process_frame(self, frame):
        """
        Xử lý frame: detect và vẽ
        
        Args:
            frame: OpenCV frame
            
        Returns:
            (frame_đã_vẽ, danh_sách_detection)
        """
        # Phát hiện
        detections = self.detect(frame)
        
        # Vẽ lên frame
        frame_with_boxes = self.draw_detections(frame.copy(), detections)
        
        return frame_with_boxes, detections
