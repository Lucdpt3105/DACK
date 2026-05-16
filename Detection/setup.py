#!/usr/bin/env python
"""
Setup script - Cài đặt và kiểm tra dependencies
"""
import subprocess
import sys
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def check_python_version():
    """Kiểm tra phiên bản Python"""
    version = sys.version_info
    
    logger.info(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logger.error("Yêu cầu Python 3.8 hoặc cao hơn")
        return False
    
    logger.info("✓ Python version OK")
    return True


def install_requirements():
    """Cài đặt requirements"""
    logger.info("\nCài đặt dependencies...")
    
    try:
        # Kiểm tra file requirements.txt
        if not os.path.exists('requirements.txt'):
            logger.error("File requirements.txt không tìm thấy")
            return False
        
        # Cài đặt
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        logger.info("✓ Dependencies installed successfully")
        return True
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Lỗi cài đặt: {e}")
        return False


def check_tesseract():
    """Kiểm tra Tesseract-OCR"""
    logger.info("\nKiểm tra Tesseract-OCR...")
    
    try:
        import pytesseract
        
        # Thử detect Tesseract
        try:
            pytesseract.get_tesseract_version()
            logger.info("✓ Tesseract-OCR đã cài đặt")
            return True
        except pytesseract.TesseractNotFoundError:
            logger.warning("⚠ pytesseract đã cài nhưng Tesseract-OCR chưa cài")
            logger.info("\nCài đặt Tesseract-OCR:")
            logger.info("  Windows: https://github.com/UB-Mannheim/tesseract/wiki")
            logger.info("  Linux: sudo apt-get install tesseract-ocr")
            logger.info("  Mac: brew install tesseract")
            return False
    
    except ImportError:
        logger.warning("⚠ pytesseract chưa cài đặt")
        return False


def check_pytorch():
    """Kiểm tra PyTorch"""
    logger.info("\nKiểm tra PyTorch...")
    
    try:
        import torch
        
        logger.info(f"PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            logger.info(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("ℹ CUDA not available (sẽ sử dụng CPU)")
        
        return True
    
    except ImportError:
        logger.error("✗ PyTorch chưa cài đặt")
        return False


def check_ultralytics():
    """Kiểm tra Ultralytics"""
    logger.info("\nKiểm tra Ultralytics...")
    
    try:
        from ultralytics import YOLO
        logger.info("✓ Ultralytics available")
        return True
    except ImportError:
        logger.error("✗ Ultralytics chưa cài đặt")
        return False


def download_model(model_name='yolov8m.pt'):
    """Tải mô hình YOLOv8"""
    logger.info(f"\nTải mô hình {model_name}...")
    
    try:
        from ultralytics import YOLO
        
        # Download model
        logger.info(f"Đang tải {model_name}...")
        model = YOLO(model_name)
        
        logger.info(f"✓ Mô hình {model_name} đã tải")
        return True
    
    except Exception as e:
        logger.error(f"Lỗi tải mô hình: {e}")
        return False


def main():
    """Main setup function"""
    
    logger.info("=" * 60)
    logger.info("TRAFFIC SIGN DETECTION - SETUP")
    logger.info("=" * 60)
    
    # Kiểm tra Python
    if not check_python_version():
        sys.exit(1)
    
    # Cài đặt requirements
    if not install_requirements():
        logger.warning("⚠ Có lỗi cài đặt requirements, nhưng tiếp tục...")
    
    # Kiểm tra các thành phần
    checks = {
        'PyTorch': check_pytorch(),
        'Ultralytics': check_ultralytics(),
        'Tesseract-OCR': check_tesseract()
    }
    
    logger.info("\n" + "=" * 60)
    logger.info("KIỂM TRA KẾT QUẢ")
    logger.info("=" * 60)
    
    for name, result in checks.items():
        status = "✓" if result else "✗"
        logger.info(f"{status} {name}")
    
    # Tải mô hình nếu các kiểm tra thành công
    if checks['PyTorch'] and checks['Ultralytics']:
        logger.info("\nCài đặt mô hình YOLOv8...")
        
        try:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Device: {device}")
            
            if download_model('yolov8m.pt'):
                logger.info("\n✓ Setup hoàn tất!")
                logger.info("\nBạn có thể bắt đầu sử dụng:")
                logger.info("  cd traffic_sign_detection")
                logger.info("  python main.py -c")
            else:
                logger.warning("⚠ Lỗi tải mô hình (có thể thử lại sau)")
        
        except Exception as e:
            logger.error(f"Lỗi: {e}")
    
    else:
        logger.error("\n✗ Setup không hoàn tất. Vui lòng cài đặt các dependencies còn thiếu")
        sys.exit(1)


if __name__ == "__main__":
    main()
