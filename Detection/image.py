"""
🚦 Vietnamese Traffic Sign Detection - Image Recognition
========================================================
Chương trình nhận diện biển báo giao thông từ ảnh tĩnh.
Sử dụng mô hình YOLOv8 đã huấn luyện trên tập dữ liệu biển báo Việt Nam.

Usage:
    python image.py --image_path path/to/image.jpg
    python image.py --image_path path/to/image.jpg --conf 0.25
    python image.py --image_path path/to/image.jpg --conf 0.25 --save
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# ============================================================
# Danh mục 29 nhóm biển báo giao thông - Model02
# ============================================================
VIE_NAMES = {
    "one way prohibition":             "Cấm đi ngược chiều",
    "no parking":                      "Cấm đỗ xe",
    "no stopping and parking":         "Cấm dừng và đỗ xe",
    "no turn left":                    "Cấm rẽ trái",
    "no turn right":                   "Cấm rẽ phải",
    "no u turn":                       "Cấm quay đầu",
    "no u and left turn":              "Cấm quay đầu và rẽ trái",
    "no u and right turn":             "Cấm quay đầu và rẽ phải",
    "no motorbike entry/turning":      "Cấm xe máy",
    "no car entry/turning":            "Cấm ô tô",
    "no truck entry/turning":          "Cấm xe tải",
    "other prohibition":               "Biển cấm khác",
    "indication":                      "Biển chỉ dẫn",
    "direction":                       "Biển phương hướng",
    "speed limit":                     "Giới hạn tốc độ",
    "weight limit":                    "Giới hạn trọng tải",
    "height limit":                    "Giới hạn chiều cao",
    "pedestrian crossing":             "Đường người đi bộ cắt ngang",
    "intersection danger":             "Nguy hiểm giao lộ",
    "road danger":                     "Nguy hiểm đường bộ",
    "pedestrian danger":               "Nguy hiểm người đi bộ",
    "construction danger":             "Công trình xây dựng",
    "slow warning":                    "Đi chậm",
    "other warning":                   "Cảnh báo khác",
    "vehicle permission lane":         "Làn xe được phép",
    "vehicle and speed permission lane": "Làn xe và tốc độ cho phép",
    "overpass route":                  "Tuyến đường vượt",
    "no more prohibition":             "Hết hiệu lực cấm",
    "other":                           "Biển báo khác",
}

# ============================================================
# Hàm xác định màu BGR theo nhóm biển báo
# ============================================================
def get_class_color_bgr(class_name: str) -> tuple:
    """Trả về màu BGR dựa theo nhóm biển báo."""
    name = class_name.lower()
    if "no more prohibition" in name:
        return (0, 180, 0)       # Xanh lá - Hết cấm
    if any(k in name for k in ["prohibition", "no parking", "no stopping",
                                "no turn", "no u", "no motorbike",
                                "no car", "no truck", "one way"]):
        return (0, 0, 220)       # Đỏ - Biển cấm
    if any(k in name for k in ["limit"]):
        return (0, 100, 255)     # Cam đậm - Giới hạn
    if any(k in name for k in ["danger", "warning", "slow", "crossing"]):
        return (0, 165, 255)     # Cam - Cảnh báo
    if any(k in name for k in ["indication", "direction", "permission", "overpass"]):
        return (200, 100, 0)     # Xanh dương - Chỉ dẫn/hiệu lệnh
    return (128, 128, 128)       # Xám - Khác


# ============================================================
# Lớp nhận diện biển báo giao thông
# ============================================================
class TrafficSignDetector:
    """Nhận diện biển báo giao thông từ ảnh."""
    
    def __init__(self, model_path: str = "model02/best28121.pt", conf: float = 0.25):
        """
        Khởi tạo detector.
        
        Args:
            model_path: Đường dẫn tới file mô hình (.pt)
            conf: Ngưỡng tự tin tối thiểu (0-1)
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Không tìm thấy mô hình: {model_path}")
        
        self.model = YOLO(model_path)
        self.conf = conf
        print(f"✓ Tải mô hình thành công: {model_path}")
        print(f"✓ Ngưỡng tự tin: {conf}")
    
    def detect(self, image_path: str) -> dict:
        """
        Nhận diện biển báo trong ảnh.
        
        Args:
            image_path: Đường dẫn tới ảnh
            
        Returns:
            dict: Thông tin về các biển báo được phát hiện
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")
        
        # Chạy inference
        results = self.model(image_path, conf=self.conf, verbose=False)
        result = results[0]
        
        # Xử lý kết quả
        detections = []
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            confidence = float(box.conf[0])
            
            # Lấy tọa độ bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            bbox = {
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
            }
            
            detection = {
                "class_id": class_id,
                "class_name": class_name,
                "class_name_vi": VIE_NAMES.get(class_name, class_name),
                "confidence": confidence,
                "bbox": bbox
            }
            detections.append(detection)
        
        return {
            "image_path": image_path,
            "total_signs": len(detections),
            "detections": detections
        }
    
    def draw_results(self, image_path: str, output_path: str | None = None) -> str:
        """
        Vẽ bounding box và label lên ảnh.
        
        Args:
            image_path: Đường dẫn ảnh đầu vào
            output_path: Đường dẫn lưu ảnh kết quả (nếu None, sẽ lưu trong thư mục output/)
            
        Returns:
            str: Đường dẫn ảnh đã lưu
        """
        # Nhận diện
        detection_result = self.detect(image_path)
        detections = detection_result["detections"]
        
        # Tải ảnh
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Không thể đọc ảnh: {image_path}")
        
        h, w = image.shape[:2]
        
        # Vẽ từng phát hiện
        for det in detections:
            x1 = int(det["bbox"]["x1"])
            y1 = int(det["bbox"]["y1"])
            x2 = int(det["bbox"]["x2"])
            y2 = int(det["bbox"]["y2"])
            
            class_name = det["class_name"]
            class_name_vi = det["class_name_vi"]
            confidence = det["confidence"]
            
            # Lấy màu
            color = get_class_color_bgr(class_name)
            
            # Vẽ bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Tạo label
            label = f"{class_name_vi} ({confidence:.2f})"
            
            # Lấy kích thước text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Vẽ nền cho text
            cv2.rectangle(image, 
                        (x1, y1 - text_h - baseline - 4),
                        (x1 + text_w + 4, y1),
                        color, -1)
            
            # Vẽ text
            cv2.putText(image, label, 
                       (x1 + 2, y1 - baseline - 2),
                       font, font_scale, (255, 255, 255), thickness)
        
        # Lưu ảnh
        if output_path is None:
            os.makedirs("output", exist_ok=True)
            filename = Path(image_path).stem + "_detected.jpg"
            output_path = os.path.join("output", filename)
        
        cv2.imwrite(output_path, image)
        print(f"\n✓ Ảnh kết quả đã lưu: {output_path}")
        
        return output_path
    
    def print_results(self, detection_result: dict):
        """In kết quả phát hiện ra console."""
        print("\n" + "="*60)
        print("📊 KẾT QUẢ NHẬN DIỆN BIỂN BÁO")
        print("="*60)
        print(f"Ảnh: {detection_result['image_path']}")
        print(f"Số lượng biển báo phát hiện: {detection_result['total_signs']}")
        print("-"*60)
        
        if detection_result['total_signs'] > 0:
            for i, det in enumerate(detection_result["detections"], 1):
                print(f"\n[{i}] {det['class_name_vi']}")
                print(f"    - Tên gốc: {det['class_name']}")
                print(f"    - Độ tự tin: {det['confidence']:.2%}")
                bbox = det["bbox"]
                print(f"    - Vị trí: ({bbox['x1']:.0f}, {bbox['y1']:.0f}) -> ({bbox['x2']:.0f}, {bbox['y2']:.0f})")
        else:
            print("\n⚠ Không phát hiện biển báo nào trong ảnh")
        
        print("\n" + "="*60)


# ============================================================
# Hàm chính
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Nhận diện biển báo giao thông từ ảnh",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ sử dụng:
  python image.py --image_path test.jpg
  python image.py --image_path test.jpg --conf 0.3
  python image.py --image_path test.jpg --conf 0.3 --save result.jpg
        """
    )
    
    parser.add_argument("--image_path", "-i", 
                       required=True,
                       help="Đường dẫn tới ảnh cần nhận diện")
    parser.add_argument("--conf", type=float, default=0.25,
                       help="Ngưỡng tự tin (0-1, mặc định: 0.25)")
    parser.add_argument("--model", "-m", default="model02/best28121.pt",
                       help="Đường dẫn tới mô hình (mặc định: model02/best28121.pt)")
    parser.add_argument("--save", "-s", 
                       help="Lưu ảnh kết quả vào đường dẫn này (tùy chọn)")
    
    args = parser.parse_args()
    
    try:
        # Tạo detector
        detector = TrafficSignDetector(model_path=args.model, conf=args.conf)
        
        # Nhận diện
        print(f"\n🔍 Đang nhận diện ảnh: {args.image_path}")
        result = detector.detect(args.image_path)
        
        # In kết quả
        detector.print_results(result)
        
        # Vẽ và lưu ảnh
        output_path: str | None = args.save if args.save else None
        detector.draw_results(args.image_path, output_path=output_path)
        
    except FileNotFoundError as e:
        print(f"❌ Lỗi: {e}")
        return 1
    except Exception as e:
        print(f"❌ Lỗi không xác định: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
