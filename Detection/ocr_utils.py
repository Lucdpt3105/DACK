"""
Tiện ích OCR để đọc chữ và số trên biển báo giao thông
"""
import pytesseract
import cv2
import numpy as np
from PIL import Image
from config import PYTESSERACT_PATH
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Đặt đường dẫn Tesseract
pytesseract.pytesseract.pytesseract_cmd = PYTESSERACT_PATH


class OCRReader:
    """Lớp để đọc text từ ảnh bằng Tesseract OCR"""
    
    def __init__(self):
        self.logger = logger
    
    def preprocess_image(self, image, blur=True):
        """
        Tiền xử lý ảnh để cải thiện OCR
        
        Args:
            image: OpenCV image
            blur: Có áp dụng Gaussian blur không
            
        Returns:
            Ảnh đã xử lý
        """
        try:
            # Chuyển sang grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Áp dụng Gaussian blur để giảm noise
            if blur:
                gray = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Áp dụng CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            
            # Thresholding
            _, gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            
            return gray
        except Exception as e:
            self.logger.error(f"Lỗi tiền xử lý ảnh: {e}")
            return image
    
    def read_text(self, image, use_preprocessing=True):
        """
        Đọc text từ ảnh
        
        Args:
            image: OpenCV image
            use_preprocessing: Có tiền xử lý ảnh không
            
        Returns:
            Text đọc được
        """
        try:
            if use_preprocessing:
                processed = self.preprocess_image(image)
            else:
                processed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Sử dụng Tesseract để nhận dạng
            custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'
            text = pytesseract.image_to_string(processed, config=custom_config)
            
            return text.strip()
        except Exception as e:
            self.logger.error(f"Lỗi đọc text: {e}")
            return ""
    
    def read_speed_limit(self, image):
        """
        Đọc số tốc độ từ biển báo giới hạn tốc độ
        
        Args:
            image: OpenCV image của biển báo
            
        Returns:
            Số tốc độ (string)
        """
        try:
            # Tiền xử lý
            processed = self.preprocess_image(image, blur=True)
            
            # Resize ảnh để cải thiện OCR
            height, width = processed.shape
            if height < 50 or width < 50:
                scale = max(50 / height, 50 / width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                processed = cv2.resize(processed, (new_width, new_height), 
                                      interpolation=cv2.INTER_CUBIC)
            
            # Đọc chỉ số
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'
            text = pytesseract.image_to_string(processed, config=custom_config)
            
            # Lấy chỉ số từ text
            numbers = ''.join(filter(str.isdigit, text))
            return numbers if numbers else "?"
        except Exception as e:
            self.logger.error(f"Lỗi đọc giới hạn tốc độ: {e}")
            return "?"
    
    def read_multi_line(self, image):
        """
        Đọc text nhiều dòng từ biển báo
        
        Args:
            image: OpenCV image
            
        Returns:
            Danh sách text từ mỗi dòng
        """
        try:
            processed = self.preprocess_image(image)
            
            # Sử dụng Tesseract để nhận dạng nhiều dòng
            text = pytesseract.image_to_string(processed)
            
            # Chia thành các dòng
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            return lines
        except Exception as e:
            self.logger.error(f"Lỗi đọc multi-line: {e}")
            return []


def extract_sign_region(frame, bbox, padding=10):
    """
    Trích xuất vùng biển báo từ frame
    
    Args:
        frame: OpenCV frame
        bbox: Bounding box [x1, y1, x2, y2]
        padding: Độ rộng padding xung quanh bbox
        
    Returns:
        Ảnh được cắt ra của biển báo
    """
    try:
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(frame.shape[1], x2 + padding)
        y2 = min(frame.shape[0], y2 + padding)
        
        sign_region = frame[y1:y2, x1:x2]
        return sign_region
    except Exception as e:
        logger.error(f"Lỗi trích xuất vùng: {e}")
        return None
