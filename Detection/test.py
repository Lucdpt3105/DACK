"""
Script kiểm tra khả năng phát hiện
"""
import cv2
import logging
from detector import TrafficSignDetector
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_on_single_image(image_path):
    """
    Kiểm tra phát hiện trên một ảnh
    
    Args:
        image_path: Đường dẫn đến ảnh kiểm tra
    """
    logger.info(f"Đang kiểm tra ảnh: {image_path}")
    
    # Kiểm tra file tồn tại
    if not os.path.exists(image_path):
        logger.error(f"File không tồn tại: {image_path}")
        return
    
    # Load ảnh
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Không thể đọc ảnh: {image_path}")
        return
    
    # Khởi tạo detector
    try:
        detector = TrafficSignDetector(model_path='yolov8m.pt')
    except Exception as e:
        logger.error(f"Lỗi khởi tạo detector: {e}")
        logger.info("Bạn có cần tải mô hình không? Chạy:")
        logger.info("  python -c \"from ultralytics import YOLO; YOLO('yolov8m.pt')\"")
        return
    
    # Phát hiện
    detections = detector.detect(image)
    
    # Vẽ kết quả
    image_result = detector.draw_detections(image.copy(), detections)
    
    # Hiển thị thông tin
    logger.info(f"Phát hiện: {len(detections)} biển báo")
    for i, det in enumerate(detections):
        logger.info(f"  [{i+1}] {det['class_name']}")
        logger.info(f"      Độ tin cậy: {det['confidence']:.2f}")
        logger.info(f"      Tọa độ: {det['bbox']}")
        logger.info(f"      Text OCR: {det['text']}")
    
    # Lưu kết quả
    output_path = "test_result.jpg"
    cv2.imwrite(output_path, image_result)
    logger.info(f"Kết quả lưu tại: {output_path}")
    
    # Hiển thị
    cv2.imshow('Test Result', image_result)
    logger.info("Nhấn bất kỳ phím nào để đóng")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_model_loading():
    """Kiểm tra load mô hình YOLOv8"""
    logger.info("Kiểm tra load mô hình YOLOv8...")
    
    try:
        from ultralytics import YOLO
        import torch
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Device: {device}")
        
        logger.info("Đang tải mô hình yolov8m...")
        model = YOLO('yolov8m.pt')
        logger.info("✓ Mô hình yolov8m đã tải thành công")
        
        return True
    except Exception as e:
        logger.error(f"✗ Lỗi tải mô hình: {e}")
        return False


def test_dependencies():
    """Kiểm tra các thư viện cần thiết"""
    logger.info("Kiểm tra các thư viện...")
    
    dependencies = {
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'torch': 'PyTorch',
        'ultralytics': 'Ultralytics',
        'pytesseract': 'pytesseract',
        'PIL': 'Pillow'
    }
    
    all_ok = True
    
    for module, name in dependencies.items():
        try:
            __import__(module)
            logger.info(f"✓ {name}")
        except ImportError:
            logger.error(f"✗ {name} - Chưa cài đặt")
            all_ok = False
    
    return all_ok


if __name__ == "__main__":
    import sys
    
    logger.info("=" * 50)
    logger.info("KIỂM TRA TRAFFIC SIGN DETECTOR")
    logger.info("=" * 50)
    
    # Kiểm tra thư viện
    logger.info("\n1. Kiểm tra thư viện...")
    if not test_dependencies():
        logger.error("Vui lòng cài đặt các thư viện còn thiếu")
        logger.info("Chạy: pip install -r requirements.txt")
        sys.exit(1)
    
    logger.info("✓ Tất cả thư viện được cài đặt")
    
    # Kiểm tra mô hình
    logger.info("\n2. Kiểm tra mô hình YOLOv8...")
    if not test_model_loading():
        logger.error("Vui lòng cài đặt PyTorch và Ultralytics")
        logger.info("Chạy: pip install -r requirements.txt")
        sys.exit(1)
    
    logger.info("✓ Mô hình YOLOv8 sẵn sàng")
    
    # Kiểm tra OCR
    logger.info("\n3. Kiểm tra OCR...")
    try:
        import pytesseract
        logger.info("✓ pytesseract có sẵn")
        logger.info("  Lưu ý: Hãy cài đặt Tesseract-OCR tại:")
        logger.info("  Windows: https://github.com/UB-Mannheim/tesseract/wiki")
        logger.info("  Linux: apt-get install tesseract-ocr")
        logger.info("  Mac: brew install tesseract")
    except ImportError:
        logger.error("✗ pytesseract chưa cài đặt")
    
    logger.info("\n" + "=" * 50)
    logger.info("Kiểm tra hoàn tất!")
    logger.info("=" * 50)
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        logger.info(f"\nKiểm tra trên ảnh: {image_path}")
        test_on_single_image(image_path)
