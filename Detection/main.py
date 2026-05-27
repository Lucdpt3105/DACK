"""
Main script - Chạy phát hiện biển báo từ camera thực tế
"""
import cv2
import logging
from datetime import datetime
from detector import TrafficSignDetector
from config import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, OUTPUT_DIR
)
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CameraTrafficSignDetector:
    """Lớp chạy phát hiện từ camera"""
    
    def __init__(self, model_path='yolov8m.pt'):
        """Khởi tạo detector"""
        self.detector = TrafficSignDetector(model_path)
        self.output_dir = OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)
    
    def run_from_camera(self):
        """Chạy phát hiện từ camera"""
        try:
            # Mở camera
            cap = cv2.VideoCapture(CAMERA_INDEX)
            
            if not cap.isOpened():
                logger.error("Không thể mở camera")
                return
            
            # Cấu hình camera
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            logger.info("Camera đã mở. Nhấn 'q' để thoát, 's' để lưu ảnh")
            
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    logger.warning("Không thể đọc frame từ camera")
                    break
                
                # Xử lý frame
                frame_processed, detections = self.detector.process_frame(frame)
                
                # Hiển thị FPS và số detection
                frame_count += 1
                text_info = f"Frame: {frame_count} | Detections: {len(detections)}"
                cv2.putText(frame_processed, text_info, (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Hiển thị frame
                cv2.imshow('Traffic Sign Detection', frame_processed)
                
                # In thông tin detection
                if detections:
                    logger.info(f"Frame {frame_count}: Phát hiện {len(detections)} biển báo")
                    for i, det in enumerate(detections):
                        logger.info(f"  [{i+1}] {det['class_name']} - "
                                  f"Confidence: {det['confidence']:.2f} - "
                                  f"Text: {det['text']}")
                
                # Xử lý phím
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    logger.info("Thoát chương trình")
                    break
                elif key == ord('s'):
                    # Lưu ảnh
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join(self.output_dir, f"detection_{timestamp}.jpg")
                    cv2.imwrite(filename, frame_processed)
                    logger.info(f"Ảnh đã lưu: {filename}")
                elif key == ord('r'):
                    # Reset frame count
                    frame_count = 0
            
            cap.release()
            cv2.destroyAllWindows()
            
        except Exception as e:
            logger.error(f"Lỗi: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def run_from_video(self, video_path):
        """
        Chạy phát hiện từ file video
        
        Args:
            video_path: Đường dẫn đến file video
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                logger.error(f"Không thể mở video: {video_path}")
                return
            
            logger.info(f"Video đã mở: {video_path}")
            logger.info("Nhấn 'q' để thoát, 's' để lưu ảnh")
            
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    logger.info("Kết thúc video")
                    break
                
                # Resize frame nếu quá lớn
                height, width = frame.shape[:2]
                if width > FRAME_WIDTH or height > FRAME_HEIGHT:
                    scale = min(FRAME_WIDTH / width, FRAME_HEIGHT / height)
                    frame = cv2.resize(frame, 
                                      (int(width * scale), int(height * scale)))
                
                # Xử lý frame
                frame_processed, detections = self.detector.process_frame(frame)
                
                frame_count += 1
                text_info = f"Frame: {frame_count} | Detections: {len(detections)}"
                cv2.putText(frame_processed, text_info, (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('Traffic Sign Detection - Video', frame_processed)
                
                if detections:
                    logger.info(f"Frame {frame_count}: Phát hiện {len(detections)} biển báo")
                    for i, det in enumerate(detections):
                        logger.info(f"  [{i+1}] {det['class_name']} - "
                                  f"Confidence: {det['confidence']:.2f} - "
                                  f"Text: {det['text']}")
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join(self.output_dir, f"detection_{timestamp}.jpg")
                    cv2.imwrite(filename, frame_processed)
                    logger.info(f"Ảnh đã lưu: {filename}")
            
            cap.release()
            cv2.destroyAllWindows()
            
        except Exception as e:
            logger.error(f"Lỗi: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def run_from_image(self, image_path):
        """
        Chạy phát hiện từ ảnh tĩnh
        
        Args:
            image_path: Đường dẫn đến file ảnh
        """
        try:
            frame = cv2.imread(image_path)
            
            if frame is None:
                logger.error(f"Không thể đọc ảnh: {image_path}")
                return
            
            logger.info(f"Ảnh đã tải: {image_path}")
            
            # Xử lý frame
            frame_processed, detections = self.detector.process_frame(frame)
            
            # Hiển thị
            cv2.imshow('Traffic Sign Detection - Image', frame_processed)
            
            # In thông tin
            logger.info(f"Phát hiện {len(detections)} biển báo")
            for i, det in enumerate(detections):
                logger.info(f"[{i+1}] {det['class_name']} - "
                          f"Confidence: {det['confidence']:.2f} - "
                          f"Text: {det['text']}")
            
            # Lưu kết quả
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_dir, f"detection_{timestamp}.jpg")
            cv2.imwrite(output_path, frame_processed)
            logger.info(f"Kết quả đã lưu: {output_path}")
            
            logger.info("Nhấn bất kỳ phím nào để đóng")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        except Exception as e:
            logger.error(f"Lỗi: {e}")


if __name__ == "__main__":
    import sys
    
    detector = CameraTrafficSignDetector(model_path='yolov8m.pt')
    
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        
        if arg.lower() in ['-c', '--camera']:
            # Từ camera
            detector.run_from_camera()
        elif arg.lower() in ['-v', '--video']:
            if len(sys.argv) > 2:
                video_path = sys.argv[2]
                detector.run_from_video(video_path)
            else:
                print("Vui lòng cung cấp đường dẫn video")
        elif arg.lower() in ['-i', '--image']:
            if len(sys.argv) > 2:
                image_path = sys.argv[2]
                detector.run_from_image(image_path)
            else:
                print("Vui lòng cung cấp đường dẫn ảnh")
        else:
            print("Cách sử dụng:")
            print("  python main.py -c             # Từ camera")
            print("  python main.py -v video.mp4   # Từ file video")
            print("  python main.py -i image.jpg   # Từ file ảnh")
    else:
        # Mặc định: chạy từ camera
        detector.run_from_camera()
