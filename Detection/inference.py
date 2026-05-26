"""
🚦 Vietnamese Traffic Sign Detection - Inference Script
=======================================================
Detect traffic signs using YOLO model.
Supports: Webcam, Video file, Image file, Folder.

Usage:
    python inference.py --model Models/1.pt --source video.mp4           # Video
    python inference.py --model Models/1.pt --source 0                   # Webcam
    python inference.py --model Models/1.pt --source image.jpg           # Image
    python inference.py --model Models/1.pt --source ./images/           # Folder

Requirements:
    pip install ultralytics opencv-python numpy Pillow
"""

import argparse
import os
import sys
import time
from collections import deque

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


# ============================================================
# Class Info - Vietnamese translations for all supported models
# ============================================================

# Vietnamese translations for traffic signs
VIE_NAMES = {
    "W.224": "Đường trơn trượt",
    "W.205c": "Giao nhau ngã ba",
    "P.102": "Cấm đi ngược chiều",
    "R.302a": "Hướng phải đi theo (phải)",
    "W.205a": "Giao nhau đường cùng cấp",
    "W.207": "Giao nhau đường sắt có rào chắn",
    "W.201a": "Chỗ ngoặt nguy hiểm (trái)",
    "P.123a": "Cấm rẽ trái",
    "I.434a": " Biển báo xe buýt",
    "R.303": "Nơi giao nhau chạy vòng xuyến",
    "P.130": "Cấm dừng và đỗ xe",
    "I.409": "Nơi đỗ xe",
    "R.415a": "Đường một chiều",
    "W.245a": "Gờ giảm tốc",
    "P.106a*Xe tải": "Cấm xe tải",
    "W.203c": "Đường bị hẹp (phải)",
    "P.117*": "Hạn chế trọng lượng",
    "P.124a*": "Cấm quay đầu",
    "P.107": "Cấm ô tô",
    "P.124d": "Cấm quay đầu và rẽ phải",
    "P.103a": "Cấm xe đi vào",
    "W.203b": "Đường bị hẹp (trái)",
    "W.221b": "Đường không bằng phẳng",
    "P.111": "Cấm xe ba gác",
    "P.129": "Cấm đỗ xe",
    "S.505a*Xe máy": "Biển phụ: xe máy",
    "W.246a": "Nguy hiểm khác",
    "W.225": "Trẻ em",
    "S.505a*Xe tải và công": "Biển phụ: xe tải và công nông",
    "P.104": "Cấm mô tô",
    "S.505a*Xe tải": "Biển phụ: xe tải",
    "Camera": "Camera giám sát",
    "P.123b": "Cấm rẽ phải",
    "W.202b": "Nhiều chỗ ngoặt liên tiếp (phải)",
    "B.8a": "Hết cấm",
    "P.137": "Cấm rẽ trái và phải",
    "P.139": "Cấm vượt",
    "W.205b": "Giao nhau đường không ưu tiên",
    "P.127*50": "Giới hạn tốc độ 50km/h",
    "P.127*60": "Giới hạn tốc độ 60km/h",
    "P.127*80": "Giới hạn tốc độ 80km/h",
    "P.127*40": "Giới hạn tốc độ 40km/h",
    "R.301e": "Hướng đi vòng xuyến",
    "W.239b*": "Đường đang sửa chữa",
    "W.233": "Người đi bộ cắt ngang",
    "I.407a": "Đường cho ô tô",
    "P.131a": "Hạn chế chiều cao",
    "P.124b1": "Cấm quay đầu",
    "W.210": "Giao nhau đường sắt không rào chắn",
    "P.124c": "Cấm quay đầu và rẽ trái",
    "W.201b": "Chỗ ngoặt nguy hiểm (phải)",
}


def get_class_info_from_model(model):
    """Auto-detect class codes and Vietnamese names from a loaded YOLO model."""
    model_names = model.names  # dict {0: 'class_name', 1: ...}
    class_codes = [model_names[i] for i in range(len(model_names))]
    class_names_vie = [
        VIE_NAMES.get(code, code)  # fallback to code if no translation
        for code in class_codes
    ]
    return class_codes, class_names_vie

# Color scheme by sign type (RGB for PIL)
SIGN_COLORS_RGB = {
    'P': (220, 0, 0),     # Red - Prohibition (Biển cấm)
    'W': (255, 165, 0),   # Orange - Warning (Biển cảnh báo)
    'R': (0, 120, 220),   # Blue - Regulatory (Biển quy định)
    'I': (0, 180, 0),     # Green - Information (Biển thông tin)
    'S': (180, 0, 180),   # Purple - Supplementary
    'B': (180, 0, 0),     # Dark Red
    'C': (128, 128, 128), # Gray - Camera
}

# Color scheme by sign type (BGR for OpenCV)
SIGN_COLORS_BGR = {
    'P': (0, 0, 220),     # Red
    'W': (0, 165, 255),   # Orange
    'R': (220, 120, 0),   # Blue
    'I': (0, 180, 0),     # Green
    'S': (180, 0, 180),   # Purple
    'B': (0, 0, 180),     # Dark Red
    'C': (128, 128, 128), # Gray
}


# ============================================================
# Font Loading - Vietnamese Unicode Support via PIL
# ============================================================
FONT_SEARCH_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
    "/usr/share/fonts/truetype/lato/Lato-Regular.ttf",
]

_font_cache = {}


def get_pil_font(size=16):
    """Get a PIL font that supports Vietnamese characters, with caching."""
    if size in _font_cache:
        return _font_cache[size]

    font = None
    for path in FONT_SEARCH_PATHS:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, size)
                break
            except Exception:
                continue

    if font is None:
        font = ImageFont.load_default()

    _font_cache[size] = font
    return font


def get_color_rgb(class_code):
    """Get RGB color based on sign type prefix (for PIL)."""
    prefix = class_code[0] if class_code else '?'
    return SIGN_COLORS_RGB.get(prefix, (255, 255, 255))


def get_color_bgr(class_code):
    """Get BGR color based on sign type prefix (for OpenCV)."""
    prefix = class_code[0] if class_code else '?'
    return SIGN_COLORS_BGR.get(prefix, (255, 255, 255))


def draw_detections(frame, results, class_codes, class_names_vie, scale_x=1.0, scale_y=1.0, show_vie=True):
    """
    Draw detections on frame with scaled coordinates.
    Single unified function for drawing all detections.
    """
    if results[0].boxes is None or len(results[0].boxes) == 0:
        return frame

    annotated = frame.copy()
    boxes_info = []
    detection_summary = []  # Store detection info for summary

    # --- Phase 1: Draw bounding boxes with OpenCV ---
    for box in results[0].boxes:
        ox1, oy1, ox2, oy2 = box.xyxy[0].tolist()
        x1 = int(ox1 * scale_x)
        y1 = int(oy1 * scale_y)
        x2 = int(ox2 * scale_x)
        y2 = int(oy2 * scale_y)

        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        code = class_codes[cls_id] if cls_id < len(class_codes) else f"C{cls_id}"
        color_bgr = get_color_bgr(code)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color_bgr, 2)
        boxes_info.append((x1, y1, x2, y2, conf, cls_id, code))
        
        # Collect detection info
        vie = class_names_vie[cls_id] if cls_id < len(class_names_vie) else ""
        detection_summary.append((code, vie, conf))

    # --- Phase 2: Render text labels with PIL (supports Vietnamese) ---
    pil_img = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font_label = get_pil_font(16)
    font_vie = get_pil_font(14)

    for x1, y1, x2, y2, conf, cls_id, code in boxes_info:
        color_rgb = get_color_rgb(code)
        label = f"{code} {conf:.0%}"

        bbox_label = draw.textbbox((0, 0), label, font=font_label)
        tw = bbox_label[2] - bbox_label[0]
        th = bbox_label[3] - bbox_label[1]

        label_y = max(y1 - th - 14, 2)
        padding = 6

        # Draw label background
        bg_box = [x1, label_y, x1 + tw + padding * 2, label_y + th + padding]
        draw.rectangle(bg_box, fill=color_rgb, outline=(0, 0, 0), width=2)
        
        # Draw text with outline
        text_x = x1 + padding
        text_y = label_y + padding // 2
        for adj_x in [-1, 0, 1]:
            for adj_y in [-1, 0, 1]:
                if adj_x != 0 or adj_y != 0:
                    draw.text((text_x + adj_x, text_y + adj_y), label, 
                             fill=(0, 0, 0), font=font_label)
        draw.text((text_x, text_y), label, fill=(255, 255, 255), font=font_label)

        # Vietnamese name below box
        if show_vie and cls_id < len(class_names_vie):
            label_vie = class_names_vie[cls_id]
            bbox_vie = draw.textbbox((0, 0), label_vie, font=font_vie)
            vtw = bbox_vie[2] - bbox_vie[0]
            vth = bbox_vie[3] - bbox_vie[1]

            vie_y = y2 + 4
            draw.rectangle([x1, vie_y, x1 + vtw + 8, vie_y + vth + 6], fill=(0, 0, 0))
            draw.text((x1 + 4, vie_y + 2), label_vie, fill=(255, 255, 255), font=font_vie)

    # --- Add summary banner at top ---
    if detection_summary:
        summary_text = "DETECTED: " + ", ".join([f"{code}({conf:.0%})" for code, vie, conf in detection_summary])
        pil_img = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        
        # Add dark background for summary
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 35), (0, 0, 0), -1)
        
        # Re-create PIL image with modified background
        pil_img = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        draw.text((5, 8), summary_text, fill=(0, 255, 0), font=get_pil_font(12))
        annotated = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def draw_fps(frame, fps):
    """Draw FPS counter on frame."""
    text = f"FPS: {fps:.1f}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2, cv2.LINE_AA)
    return frame


def draw_info_panel(frame, num_detections):
    """Draw info panel on frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 40), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    info = f"Detected: {num_detections} sign(s) | Press 'q' to quit | 's' to save screenshot"
    cv2.putText(frame, info, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def process_image(model, image_path, class_codes, class_names_vie, conf, output_dir):
    """Process a single image."""
    print(f"📷 Processing: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Cannot read image: {image_path}")
        return

    results = model.predict(
        source=img, conf=conf, iou=0.5, imgsz=800,
        verbose=False,
    )

    annotated = draw_detections(img, results, class_codes, class_names_vie)
    num_det = len(results[0].boxes) if results[0].boxes is not None else 0

    print(f"   Found {num_det} sign(s)")

    if results[0].boxes is not None:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf_val = float(box.conf[0])
            code = class_codes[cls_id] if cls_id < len(class_codes) else f"C{cls_id}"
            vie = class_names_vie[cls_id] if cls_id < len(class_names_vie) else ""
            print(f"   🔹 {code} ({vie}) - Conf: {conf_val:.1%}")

    # Save output
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.basename(image_path)
        base_name = os.path.splitext(base)[0]
        
        # Save image
        out_path = os.path.join(output_dir, f"det_{base}")
        cv2.imwrite(out_path, annotated)
        print(f"   💾 Saved image: {out_path}")
        
        # Save text report
        txt_path = os.path.join(output_dir, f"det_{base_name}.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"Traffic Sign Detection Report\n")
            f.write(f"{'='*50}\n")
            f.write(f"Image: {image_path}\n")
            f.write(f"Total signs detected: {num_det}\n\n")
            if num_det > 0:
                f.write("Detections:\n")
                f.write("-" * 50 + "\n")
                for i, box in enumerate(results[0].boxes, 1):
                    cls_id = int(box.cls[0])
                    conf_val = float(box.conf[0])
                    code = class_codes[cls_id] if cls_id < len(class_codes) else f"C{cls_id}"
                    vie = class_names_vie[cls_id] if cls_id < len(class_names_vie) else ""
                    f.write(f"{i}. {code}\n")
                    f.write(f"   Vietnamese: {vie}\n")
                    f.write(f"   Confidence: {conf_val:.1%}\n\n")
        print(f"   📄 Saved report: {txt_path}")

    # Display
    cv2.imshow("Traffic Sign Detection", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_video(model, source, class_codes, class_names_vie, conf, output_dir,
                  skip_frames=2, imgsz=800, display_width=960):
    """Process video file or webcam stream."""
    is_webcam = str(source) == '0' or str(source).startswith('/dev/')

    if is_webcam:
        cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
        print("📹 Starting webcam... Press 'q' to quit.")
    else:
        cap = cv2.VideoCapture(source)
        print(f"🎬 Processing video: {source}")

    if not cap.isOpened():
        print(f"❌ Cannot open source: {source}")
        return

    # Get video properties
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate display scale
    scale = min(display_width / w, 1.0)
    disp_w = int(w * scale)
    disp_h = int(h * scale)

    print(f"📐 Video: {w}x{h} @ {fps_video:.0f}fps | Display: {disp_w}x{disp_h}")
    print(f"⚡ Skip frames: {skip_frames} | Inference size: {imgsz}")

    # Video writer for output
    writer = None
    if output_dir and not is_webcam:
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(source))[0]
        out_path = os.path.join(output_dir, f"det_{base}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, fps_video, (disp_w, disp_h))
        print(f"💾 Output: {out_path}")

    frame_count = 0
    fps_history = deque(maxlen=30)
    prev_time = time.time()
    screenshot_count = 0
    last_results = None
    last_annotated = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Run detection only every N frames
        if frame_count % (skip_frames + 1) == 0 or last_results is None:
            last_results = model.predict(
                source=frame, conf=conf, iou=0.5, imgsz=imgsz,
                verbose=False,
            )

        # Resize frame for display AFTER inference
        if scale < 1.0:
            display_frame = cv2.resize(frame, (disp_w, disp_h),
                                       interpolation=cv2.INTER_AREA)
        else:
            display_frame = frame.copy()

        # Draw detections
        if last_results is not None:
            last_annotated = draw_detections(
                display_frame, last_results, class_codes, class_names_vie,
                scale_x=scale, scale_y=scale,
            )
            num_det = len(last_results[0].boxes) if last_results[0].boxes is not None else 0
        else:
            last_annotated = display_frame
            num_det = 0

        # Calculate FPS
        curr_time = time.time()
        dt = curr_time - prev_time
        if dt > 0:
            fps_history.append(1.0 / dt)
        prev_time = curr_time
        avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0

        annotated = draw_fps(last_annotated, avg_fps)
        annotated = draw_info_panel(annotated, num_det)

        # Progress info for video files
        if not is_webcam and total_frames > 0:
            progress = frame_count / total_frames * 100
            sys.stdout.write(
                f"\r⏳ Frame {frame_count}/{total_frames} ({progress:.1f}%) "
                f"| FPS: {avg_fps:.1f} | Detected: {num_det}"
            )
            sys.stdout.flush()

        # Display
        cv2.imshow("Traffic Sign Detection", annotated)

        # Write to output
        if writer:
            writer.write(annotated)

        # Handle key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n\n👋 Quitting...")
            break
        elif key == ord('s'):
            screenshot_count += 1
            ss_dir = output_dir or '.'
            os.makedirs(ss_dir, exist_ok=True)
            ss_path = os.path.join(ss_dir, f"screenshot_{screenshot_count}.jpg")
            cv2.imwrite(ss_path, annotated)
            print(f"\n📸 Screenshot saved: {ss_path}")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    print(f"\n\n✅ Processed {frame_count} frames (avg FPS: {avg_fps:.1f})")


def process_folder(model, folder_path, class_codes, class_names_vie, conf, output_dir):
    """Process all images in a folder."""
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    images = [
        os.path.join(folder_path, f)
        for f in sorted(os.listdir(folder_path))
        if os.path.splitext(f)[1].lower() in exts
    ]

    if not images:
        print(f"❌ No images found in: {folder_path}")
        return

    print(f"📁 Found {len(images)} images in {folder_path}")
    for img_path in images:
        process_image(model, img_path, class_codes, class_names_vie, conf, output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="🚦 Vietnamese Traffic Sign Detection using YOLO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process video
  python inference.py --model Models/1.pt --source video.mp4

  # Process webcam
  python inference.py --model Models/1.pt --source 0

  # Process image
  python inference.py --model Models/1.pt --source image.jpg

  # Process folder
  python inference.py --model Models/1.pt --source ./images/
        """,
    )
    parser.add_argument('--model', '-m', default='Models/1.pt',
                        help='Path to YOLO model (.pt file) - default: Models/1.pt')
    parser.add_argument('--source', '-s', required=True,
                        help='Input source: 0 (webcam), video.mp4, image.jpg, or folder/')
    parser.add_argument('--conf', '-c', type=float, default=0.5,
                        help='Confidence threshold (default: 0.5)')
    parser.add_argument('--output', '-o', default='output',
                        help='Output directory for saving results (default: output)')
    parser.add_argument('--no-vie', action='store_true',
                        help='Disable Vietnamese labels')
    parser.add_argument('--skip', type=int, default=2,
                        help='Skip N frames between inferences for speed (default: 2)')
    parser.add_argument('--imgsz', type=int, default=800,
                        help='Inference input size (default: 800)')
    parser.add_argument('--display-width', type=int, default=960,
                        help='Display window width in pixels (default: 960)')

    args = parser.parse_args()

    # Load model
    print(f"🔄 Loading model: {args.model}")
    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        sys.exit(1)

    model = YOLO(args.model)
    print("✅ Model loaded!")

    # Load class info
    class_codes, class_names_vie = get_class_info_from_model(model)
    print(f"📊 Classes: {len(class_codes)}")
    print(f"   Examples: {class_codes[:5]}...")

    show_vie = not args.no_vie

    # Determine source type
    source = args.source

    if source.isdigit() or source.startswith('/dev/'):
        # Webcam
        process_video(model, source, class_codes, class_names_vie, args.conf,
                      args.output, skip_frames=args.skip, imgsz=args.imgsz,
                      display_width=args.display_width)
    elif os.path.isdir(source):
        # Folder
        process_folder(model, source, class_codes, class_names_vie, args.conf, args.output)
    elif os.path.isfile(source):
        ext = os.path.splitext(source)[1].lower()
        if ext in {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv'}:
            # Video
            process_video(model, source, class_codes, class_names_vie, args.conf,
                          args.output, skip_frames=args.skip, imgsz=args.imgsz,
                          display_width=args.display_width)
        else:
            # Image
            process_image(model, source, class_codes, class_names_vie, args.conf, args.output)
    else:
        print(f"❌ Source not found: {source}")
        sys.exit(1)


if __name__ == '__main__':
    main()
