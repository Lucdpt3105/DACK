"""
Advanced batch processing script
"""
import os
import cv2
import logging
from detector import TrafficSignDetector
from ocr_utils import OCRReader
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BatchProcessor:
    """Xử lý batch các ảnh/video"""
    
    def __init__(self, model_path='yolov8m.pt'):
        self.detector = TrafficSignDetector(model_path)
        self.results = []
    
    def process_images_folder(self, folder_path, output_folder=None):
        """
        Xử lý tất cả ảnh trong thư mục
        
        Args:
            folder_path: Thư mục chứa ảnh
            output_folder: Thư mục lưu kết quả (nếu None, tạo mới)
        """
        if output_folder is None:
            output_folder = f"batch_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        os.makedirs(output_folder, exist_ok=True)
        
        # Lấy danh sách ảnh
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        images = [f for f in os.listdir(folder_path) 
                 if f.lower().endswith(image_extensions)]
        
        logger.info(f"Tìm thấy {len(images)} ảnh trong {folder_path}")
        
        results = []
        
        for idx, image_file in enumerate(images, 1):
            image_path = os.path.join(folder_path, image_file)
            logger.info(f"[{idx}/{len(images)}] Xử lý: {image_file}")
            
            try:
                # Đọc ảnh
                frame = cv2.imread(image_path)
                if frame is None:
                    logger.warning(f"  Không thể đọc ảnh")
                    continue
                
                # Detect
                detections = self.detector.detect(frame)
                
                # Vẽ kết quả
                frame_result = self.detector.draw_detections(frame.copy(), detections)
                
                # Lưu ảnh
                output_path = os.path.join(output_folder, f"result_{image_file}")
                cv2.imwrite(output_path, frame_result)
                
                # Ghi kết quả
                result = {
                    'file': image_file,
                    'detections_count': len(detections),
                    'detections': detections,
                    'output_path': output_path
                }
                results.append(result)
                
                logger.info(f"  Phát hiện: {len(detections)} biển báo")
                
            except Exception as e:
                logger.error(f"  Lỗi xử lý {image_file}: {e}")
        
        # Lưu report
        self._save_report(results, output_folder)
        
        logger.info(f"Hoàn tất! Kết quả lưu tại: {output_folder}")
        return results
    
    def _save_report(self, results, output_folder):
        """Lưu báo cáo kết quả"""
        report_path = os.path.join(output_folder, 'report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("TRAFFIC SIGN DETECTION - BATCH REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Images: {len(results)}\n")
            f.write("\n")
            
            total_detections = sum(r['detections_count'] for r in results)
            f.write(f"Total Detections: {total_detections}\n\n")
            
            f.write("DETAIL:\n")
            f.write("-" * 60 + "\n")
            
            for result in results:
                f.write(f"\nFile: {result['file']}\n")
                f.write(f"Detections: {result['detections_count']}\n")
                
                if result['detections']:
                    for i, det in enumerate(result['detections'], 1):
                        f.write(f"  [{i}] {det['class_name']}\n")
                        f.write(f"      Confidence: {det['confidence']:.4f}\n")
                        f.write(f"      Text: {det['text']}\n")


def process_video_to_frames(video_path, output_folder=None, frame_interval=10):
    """
    Chuyển đổi video thành frames và detect
    
    Args:
        video_path: Đường dẫn video
        output_folder: Thư mục lưu kết quả
        frame_interval: Khoảng cách frames (mỗi N frame lưu 1)
    """
    if output_folder is None:
        output_folder = f"frames_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    os.makedirs(output_folder, exist_ok=True)
    
    detector = TrafficSignDetector(model_path='yolov8m.pt')
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        logger.error(f"Không thể mở video: {video_path}")
        return
    
    frame_count = 0
    saved_count = 0
    
    logger.info(f"Đang xử lý video: {video_path}")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_count += 1
        
        if frame_count % frame_interval == 0:
            # Detect
            detections = detector.detect(frame)
            
            if detections:  # Chỉ lưu frame có detection
                frame_result = detector.draw_detections(frame.copy(), detections)
                
                output_path = os.path.join(output_folder, 
                                          f"frame_{frame_count:06d}.jpg")
                cv2.imwrite(output_path, frame_result)
                
                saved_count += 1
                logger.info(f"Frame {frame_count}: Phát hiện {len(detections)} biển báo")
    
    cap.release()
    logger.info(f"Hoàn tất! Lưu {saved_count} frames tại: {output_folder}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Cách sử dụng:")
        print("  python batch.py --images folder_path")
        print("  python batch.py --video video_path")
        print("\nVí dụ:")
        print("  python batch.py --images ./my_images")
        print("  python batch.py --video ./my_video.mp4")
        sys.exit(1)
    
    if sys.argv[1] == '--images' and len(sys.argv) > 2:
        processor = BatchProcessor(model_path='yolov8m.pt')
        processor.process_images_folder(sys.argv[2])
    
    elif sys.argv[1] == '--video' and len(sys.argv) > 2:
        process_video_to_frames(sys.argv[2])
    
    else:
        print("Tham số không hợp lệ")
