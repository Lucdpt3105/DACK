"""
Advanced example với metrics và performance monitoring
"""
import cv2
import time
import numpy as np
import logging
from detector import TrafficSignDetector
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Theo dõi hiệu suất detection"""
    
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.fps_history = []
        self.detection_times = []
        self.detections_per_frame = []
    
    def add_fps(self, fps):
        """Thêm FPS vào history"""
        self.fps_history.append(fps)
        if len(self.fps_history) > self.window_size:
            self.fps_history.pop(0)
    
    def add_detection_time(self, time_ms):
        """Thêm thời gian detect"""
        self.detection_times.append(time_ms)
        if len(self.detection_times) > self.window_size:
            self.detection_times.pop(0)
    
    def add_detections(self, count):
        """Thêm số detection"""
        self.detections_per_frame.append(count)
        if len(self.detections_per_frame) > self.window_size:
            self.detections_per_frame.pop(0)
    
    def get_avg_fps(self):
        """Lấy FPS trung bình"""
        return np.mean(self.fps_history) if self.fps_history else 0
    
    def get_avg_detection_time(self):
        """Lấy thời gian detect trung bình"""
        return np.mean(self.detection_times) if self.detection_times else 0
    
    def get_avg_detections(self):
        """Lấy số detection trung bình"""
        return np.mean(self.detections_per_frame) if self.detections_per_frame else 0


class AdvancedDetector:
    """Detector nâng cao với metrics"""
    
    def __init__(self, model_path='yolov8m.pt'):
        self.detector = TrafficSignDetector(model_path)
        self.monitor = PerformanceMonitor()
        self.total_detections = 0
        self.start_time = time.time()
    
    def process_frame_with_metrics(self, frame):
        """
        Xử lý frame với metrics
        
        Returns:
            (frame_result, detections, metrics)
        """
        # Detect
        detect_start = time.time()
        detections = self.detector.detect(frame)
        detect_time = (time.time() - detect_start) * 1000  # ms
        
        # Vẽ
        frame_result = self.detector.draw_detections(frame.copy(), detections)
        
        # Cập nhật metrics
        self.monitor.add_detection_time(detect_time)
        self.monitor.add_detections(len(detections))
        self.total_detections += len(detections)
        
        metrics = {
            'detect_time_ms': detect_time,
            'num_detections': len(detections),
            'avg_fps': self.monitor.get_avg_fps(),
            'avg_detect_time': self.monitor.get_avg_detection_time(),
            'avg_detections_per_frame': self.monitor.get_avg_detections(),
            'total_detections': self.total_detections
        }
        
        return frame_result, detections, metrics
    
    def draw_metrics_on_frame(self, frame, metrics):
        """Vẽ metrics lên frame"""
        y_offset = 30
        
        info_texts = [
            f"FPS: {metrics['avg_fps']:.1f}",
            f"Detect: {metrics['detect_time_ms']:.1f}ms",
            f"Detections: {metrics['num_detections']}",
            f"Avg/Frame: {metrics['avg_detections_per_frame']:.1f}",
            f"Total: {metrics['total_detections']}"
        ]
        
        for text in info_texts:
            cv2.putText(frame, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                       (0, 255, 0), 2)
            y_offset += 25
        
        return frame


def run_advanced_camera_demo(model_path='yolov8m.pt'):
    """
    Chạy demo nâng cao với metrics từ camera
    """
    advanced = AdvancedDetector(model_path)
    
    cap = cv2.VideoCapture(0)
    
    logger.info("Camera demo nâng cao - Nhấn 'q' để thoát")
    
    prev_time = time.time()
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Tính FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        
        advanced.monitor.add_fps(fps)
        
        # Xử lý frame
        frame_result, detections, metrics = advanced.process_frame_with_metrics(frame)
        
        # Vẽ metrics
        frame_result = advanced.draw_metrics_on_frame(frame_result, metrics)
        
        # Hiển thị
        cv2.imshow('Advanced Traffic Sign Detection', frame_result)
        
        # In log
        if advanced.monitor.detections_per_frame[-1] > 0:
            logger.info(f"Frame: {metrics['total_detections']} | "
                       f"FPS: {metrics['avg_fps']:.1f} | "
                       f"Detections: {metrics['num_detections']}")
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Báo cáo cuối
    logger.info("\n" + "=" * 50)
    logger.info("BÁO CÁO HOÀN TẤT")
    logger.info("=" * 50)
    logger.info(f"Thời gian chạy: {time.time() - advanced.start_time:.1f}s")
    logger.info(f"FPS trung bình: {advanced.monitor.get_avg_fps():.1f}")
    logger.info(f"Thời gian detect trung bình: {advanced.monitor.get_avg_detection_time():.1f}ms")
    logger.info(f"Detection trung bình/frame: {advanced.monitor.get_avg_detections():.1f}")
    logger.info(f"Tổng detections: {advanced.total_detections}")
    logger.info("=" * 50)


def benchmark_model(image_path, num_runs=10):
    """
    Benchmark mô hình
    
    Args:
        image_path: Ảnh để benchmark
        num_runs: Số lần chạy
    """
    detector = TrafficSignDetector(model_path='yolov8m.pt')
    
    image = cv2.imread(image_path)
    
    logger.info(f"Benchmark với {num_runs} lần chạy...")
    
    times = []
    
    # Warm up
    detector.detect(image)
    
    # Benchmark
    for i in range(num_runs):
        start = time.time()
        detector.detect(image)
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)
    
    # Kết quả
    logger.info("\n" + "=" * 50)
    logger.info("BENCHMARK RESULTS")
    logger.info("=" * 50)
    logger.info(f"Số lần chạy: {num_runs}")
    logger.info(f"Thời gian min: {min(times):.2f}ms")
    logger.info(f"Thời gian max: {max(times):.2f}ms")
    logger.info(f"Thời gian trung bình: {np.mean(times):.2f}ms")
    logger.info(f"Std deviation: {np.std(times):.2f}ms")
    logger.info(f"FPS: {1000 / np.mean(times):.1f}")
    logger.info("=" * 50)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--benchmark':
        if len(sys.argv) > 2:
            benchmark_model(sys.argv[2])
        else:
            logger.error("Vui lòng cung cấp ảnh để benchmark")
    else:
        run_advanced_camera_demo()
