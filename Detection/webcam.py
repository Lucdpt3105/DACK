"""
🚦 Vietnamese Traffic Sign Detection - Webcam
============================================
Chương trình nhận diện biển báo giao thông trực tiếp từ webcam.

Usage:
    python Detection/webcam.py
    python Detection/webcam.py --camera 1 --conf 0.25 --imgsz 640
    python Detection/webcam.py --save
"""

import argparse
import os
import sys
import time
from collections import deque
from typing import cast

import cv2
import numpy as np
from ultralytics import YOLO

from inference import (
    apply_clahe,
    draw_detections,
    enhance_image,
    sharpen_image,
)


def open_camera(camera_index: int, width: int | None, height: int | None):
    """Mở webcam và cấu hình kích thước nếu có truyền tham số."""
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(f"Không thể mở webcam index {camera_index}")

    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    return cap


def main():
    parser = argparse.ArgumentParser(
        description="🚦 Nhận diện biển báo giao thông trực tiếp bằng webcam"
    )
    parser.add_argument("--model", "-m", default="model/best.pt",
                        help="Đường dẫn file mô hình YOLO (.pt)")
    parser.add_argument("--camera", type=int, default=0,
                        help="Index webcam (mặc định: 0)")
    parser.add_argument("--conf", "-c", type=float, default=0.25,
                        help="Ngưỡng tự tin (mặc định: 0.25)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Kích thước ảnh đưa vào mô hình (mặc định: 640)")
    parser.add_argument("--skip", type=int, default=0,
                        help="Số frame bỏ qua giữa mỗi lần predict (mặc định: 0)")
    parser.add_argument("--width", type=int, default=1280,
                        help="Chiều rộng webcam mong muốn (mặc định: 1280)")
    parser.add_argument("--height", type=int, default=720,
                        help="Chiều cao webcam mong muốn (mặc định: 720)")
    parser.add_argument("--display-width", type=int, default=1024,
                        help="Chiều rộng cửa sổ hiển thị tối đa (mặc định: 1024)")
    parser.add_argument("--save", action="store_true",
                        help="Lưu video kết quả vào thư mục output/")
    parser.add_argument("--output", "-o", default="output",
                        help="Thư mục lưu video khi dùng --save")

    parser.add_argument("--enhance", action="store_true",
                        help="Bật tăng cường chất lượng ảnh LAB")
    parser.add_argument("--clahe", action="store_true",
                        help="Bật CLAHE - nâng cao contrast địa phương")
    parser.add_argument("--sharpen", action="store_true",
                        help="Bật làm sắc nét ảnh")
    parser.add_argument("--brightness", type=float, default=1.0,
                        help="Độ sáng cho LAB enhancement")
    parser.add_argument("--contrast", type=float, default=1.2,
                        help="Độ tương phản cho LAB enhancement")

    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"❌ Không tìm thấy file mô hình tại: {args.model}")
        return 1

    print(f"🔄 Đang tải mô hình YOLO: {args.model}...")
    model = YOLO(args.model)
    model_classes = model.names
    print(f"✅ Tải thành công! Mô hình có: {len(model_classes)} classes.")

    try:
        cap = open_camera(args.camera, args.width, args.height)
    except RuntimeError as exc:
        print(f"❌ {exc}")
        return 1

    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or args.width
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or args.height
    cam_fps = cap.get(cv2.CAP_PROP_FPS) or 30

    scale = min(args.display_width / cam_w, 1.0)
    disp_w, disp_h = int(cam_w * scale), int(cam_h * scale)

    writer = None
    out_path = None
    if args.save:
        os.makedirs(args.output, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(args.output, f"webcam_detected_{timestamp}.mp4")
        fourcc = getattr(cv2, "VideoWriter_fourcc")(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, cam_fps, (disp_w, disp_h))
        print(f"💾 Video kết quả sẽ được lưu tại: {out_path}")

    enhancements = []
    if args.enhance:
        enhancements.append(f"LAB(brightness={args.brightness}, contrast={args.contrast})")
    if args.clahe:
        enhancements.append("CLAHE")
    if args.sharpen:
        enhancements.append("Sharpen")

    print(
        f"🚀 Webcam {args.camera}: {cam_w}x{cam_h} | "
        f"imgsz={args.imgsz} | conf={args.conf} | skip={args.skip}"
    )
    print(f"🎨 Enhancement: {' | '.join(enhancements) if enhancements else 'Không có'}")
    print("Nhấn 'q' để thoát, 's' để chụp ảnh màn hình.")

    frame_count = 0
    fps_history = deque(maxlen=30)
    prev_time = time.time()
    last_results = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("\n❌ Không đọc được frame từ webcam.")
            break

        frame_count += 1

        frame_for_predict = cast(np.ndarray, frame.copy())
        if args.enhance:
            frame_for_predict = cast(np.ndarray, enhance_image(
                frame_for_predict,
                brightness=args.brightness,
                contrast=args.contrast,
            ))
        if args.clahe:
            frame_for_predict = cast(np.ndarray, apply_clahe(frame_for_predict))
        if args.sharpen:
            frame_for_predict = cast(np.ndarray, sharpen_image(frame_for_predict))

        if frame_count % (args.skip + 1) == 0 or last_results is None:
            last_results = model.predict(
                source=frame_for_predict,
                conf=args.conf,
                iou=0.4,
                imgsz=args.imgsz,
                verbose=False,
            )

        if scale < 1.0:
            display_frame = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        else:
            display_frame = frame.copy()

        if last_results is not None:
            annotated = draw_detections(
                display_frame,
                last_results,
                model_classes,
                scale_x=scale,
                scale_y=scale,
            )
            num_det = len(last_results[0].boxes) if last_results[0].boxes is not None else 0
        else:
            annotated = display_frame
            num_det = 0

        curr_time = time.time()
        dt = curr_time - prev_time
        if dt > 0:
            fps_history.append(1.0 / dt)
        prev_time = curr_time
        avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0

        cv2.putText(
            annotated,
            f"FPS: {avg_fps:.1f}",
            (10, disp_h - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.rectangle(annotated, (0, disp_h - 35), (disp_w, disp_h), (0, 0, 0), -1)
        info_str = f"Camera: {args.camera} | Tim thay: {num_det} bien bao | q: thoat | s: chup anh"
        cv2.putText(
            annotated,
            info_str,
            (10, disp_h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow("Traffic Sign Webcam Detection", annotated)
        if writer:
            writer.write(annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("\n👋 Đã thoát chương trình.")
            break
        if key == ord("s"):
            os.makedirs(args.output, exist_ok=True)
            screenshot_path = os.path.join(
                args.output,
                f"webcam_screenshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg",
            )
            cv2.imwrite(screenshot_path, annotated)
            print(f"\n📸 Đã lưu ảnh: {screenshot_path}")

        sys.stdout.write(f"\rFPS: {avg_fps:.1f} | Detections: {num_det} | Frames: {frame_count}")
        sys.stdout.flush()

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    if out_path:
        print(f"\n✅ Video đã lưu: {out_path}")
    print(f"✅ Đã xử lý {frame_count} frame.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
