# ============================================================
# 🚦 YOLO V8m Training - Vietnamese Traffic Sign Detection
# ============================================================
# Hướng dẫn: Copy từng phần (đánh dấu #%%) vào từng cell Colab
# Dataset: Kaggle maitam/vietnamese-traffic-signs (52 classes)
# Target: mAP@0.5 > 0.8
# ============================================================

#%% ====== CELL 1: SETUP & INSTALL ======
# !pip install -q ultralytics kagglehub albumentations
# from google.colab import drive
# drive.mount('/content/drive')

import os
import shutil
import random
import yaml
import glob
import numpy as np
from pathlib import Path

print("✅ Setup complete")

#%% ====== CELL 2: DOWNLOAD DATASET ======
import kagglehub

dataset_base = "/content/drive/My Drive/vietnamese-traffic-signs"

if not os.path.exists(dataset_base):
    from getpass import getpass
    os.environ['KAGGLE_USERNAME'] = input("Kaggle username: ")
    os.environ['KAGGLE_KEY'] = getpass("Kaggle token: ")

    print("⏳ Downloading dataset...")
    path = kagglehub.dataset_download("maitam/vietnamese-traffic-signs")
    shutil.copytree(path, dataset_base)
    print(f"✅ Dataset saved to: {dataset_base}")
else:
    print(f"✅ Dataset already exists at: {dataset_base}")

# Detect archive subfolder
archive_path = os.path.join(dataset_base, "archive")
if not os.path.exists(archive_path):
    archive_path = dataset_base

images_src = os.path.join(archive_path, "images")
labels_src = os.path.join(archive_path, "labels")
split_dir = os.path.join(archive_path, "split_dataset")

print(f"📁 Images: {len(os.listdir(images_src))} files")
print(f"📁 Labels: {len(os.listdir(labels_src))} files")

#%% ====== CELL 3: LOAD CLASS NAMES ======
classes_path = os.path.join(archive_path, "classes.txt")
classes_en_path = os.path.join(archive_path, "classes_en.txt")
classes_vie_path = os.path.join(archive_path, "classes_vie.txt")

with open(classes_path, 'r', encoding='utf-8') as f:
    class_codes = [l.strip() for l in f.readlines()]

class_names_en, class_names_vie = [], []
if os.path.exists(classes_en_path):
    with open(classes_en_path, 'r', encoding='utf-8') as f:
        class_names_en = [l.strip() for l in f.readlines()]
if os.path.exists(classes_vie_path):
    with open(classes_vie_path, 'r', encoding='utf-8') as f:
        class_names_vie = [l.strip() for l in f.readlines()]

print(f"📊 Total classes: {len(class_codes)}")
for i, code in enumerate(class_codes):
    en = class_names_en[i] if i < len(class_names_en) else ""
    vie = class_names_vie[i] if i < len(class_names_vie) else ""
    print(f"  [{i:2d}] {code:<20s} {en:<35s} {vie}")

#%% ====== CELL 4: PREPARE YOLO DIRECTORY STRUCTURE ======
DATASET_DIR = "/content/dataset"

# Clean previous
if os.path.exists(DATASET_DIR):
    shutil.rmtree(DATASET_DIR)

for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(DATASET_DIR, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, split, 'labels'), exist_ok=True)

# Load split files
train_files_path = os.path.join(split_dir, "train_files.txt")
test_files_path = os.path.join(split_dir, "test_files.txt")

with open(train_files_path, 'r') as f:
    train_files = [l.strip() for l in f.readlines() if l.strip()]
with open(test_files_path, 'r') as f:
    test_files = [l.strip() for l in f.readlines() if l.strip()]

# Split train → train (85%) + val (15%)
random.seed(42)
random.shuffle(train_files)
val_count = max(1, int(len(train_files) * 0.15))
val_files = train_files[:val_count]
actual_train_files = train_files[val_count:]

print(f"📊 Split: Train={len(actual_train_files)}, Val={len(val_files)}, Test={len(test_files)}")

def copy_files(file_list, split_name):
    """Copy image and label files to YOLO directory structure."""
    copied = 0
    skipped = 0
    for fname in file_list:
        # fname could be just number like "0001" or "0001.jpg"
        base = os.path.splitext(fname)[0] if '.' in fname else fname

        img_file = None
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            candidate = os.path.join(images_src, base + ext)
            if os.path.exists(candidate):
                img_file = candidate
                break

        label_file = os.path.join(labels_src, base + '.txt')

        if img_file and os.path.exists(label_file):
            shutil.copy2(img_file, os.path.join(DATASET_DIR, split_name, 'images', os.path.basename(img_file)))
            shutil.copy2(label_file, os.path.join(DATASET_DIR, split_name, 'labels', base + '.txt'))
            copied += 1
        else:
            skipped += 1

    print(f"  {split_name}: copied={copied}, skipped={skipped}")
    return copied

copy_files(actual_train_files, 'train')
copy_files(val_files, 'val')
copy_files(test_files, 'test')

print("✅ YOLO directory structure ready!")

#%% ====== CELL 5: CREATE data.yaml ======
data_yaml = {
    'path': DATASET_DIR,
    'train': 'train/images',
    'val': 'val/images',
    'test': 'test/images',
    'nc': len(class_codes),
    'names': class_codes,
}

yaml_path = os.path.join(DATASET_DIR, 'data.yaml')
with open(yaml_path, 'w', encoding='utf-8') as f:
    yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)

print(f"✅ data.yaml created at: {yaml_path}")
print(f"   Classes: {len(class_codes)}")

# Verify
with open(yaml_path, 'r') as f:
    print(f.read())

#%% ====== CELL 6: ANALYZE DATASET DISTRIBUTION ======
import matplotlib.pyplot as plt
from collections import Counter

def count_classes(split_name):
    """Count instances per class in a split."""
    counter = Counter()
    labels_dir = os.path.join(DATASET_DIR, split_name, 'labels')
    for label_file in glob.glob(os.path.join(labels_dir, '*.txt')):
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    counter[int(parts[0])] += 1
    return counter

train_dist = count_classes('train')
val_dist = count_classes('val')
test_dist = count_classes('test')

total_instances = sum(train_dist.values()) + sum(val_dist.values()) + sum(test_dist.values())
print(f"📊 Total object instances: {total_instances}")
print(f"   Train: {sum(train_dist.values())}, Val: {sum(val_dist.values())}, Test: {sum(test_dist.values())}")

# Plot distribution
fig, ax = plt.subplots(figsize=(16, 6))
all_classes = range(len(class_codes))
train_counts = [train_dist.get(c, 0) for c in all_classes]
val_counts = [val_dist.get(c, 0) for c in all_classes]

x = np.arange(len(class_codes))
width = 0.4
ax.bar(x - width/2, train_counts, width, label='Train', color='#4CAF50')
ax.bar(x + width/2, val_counts, width, label='Val', color='#FF9800')
ax.set_xlabel('Class ID')
ax.set_ylabel('Instance Count')
ax.set_title('Dataset Distribution by Class')
ax.set_xticks(x)
ax.set_xticklabels(class_codes, rotation=90, fontsize=6)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(DATASET_DIR, 'distribution.png'), dpi=150)
plt.show()

# Warn about imbalanced classes
min_class = min(train_dist, key=train_dist.get) if train_dist else 0
max_class = max(train_dist, key=train_dist.get) if train_dist else 0
print(f"\n⚠️  Least samples: Class {min_class} ({class_codes[min_class]}) = {train_dist.get(min_class, 0)}")
print(f"✅ Most samples:  Class {max_class} ({class_codes[max_class]}) = {train_dist.get(max_class, 0)}")

#%% ====== CELL 7: VISUALIZE SAMPLE DATA ======
import cv2
from matplotlib import patches

def visualize_sample(split='train', num_samples=6):
    """Show sample images with bounding boxes."""
    img_dir = os.path.join(DATASET_DIR, split, 'images')
    lbl_dir = os.path.join(DATASET_DIR, split, 'labels')

    img_files = sorted(glob.glob(os.path.join(img_dir, '*')))
    samples = random.sample(img_files, min(num_samples, len(img_files)))

    cols = 3
    rows = (len(samples) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, 6 * rows))
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    axes = np.array(axes).flatten()

    colors_map = {
        'P': '#FF4444',  # Red - Prohibition
        'W': '#FF9900',  # Orange - Warning
        'R': '#4444FF',  # Blue - Regulatory
        'I': '#44CC44',  # Green - Information
        'S': '#AA44CC',  # Purple - Supplementary
        'B': '#CC4444',  # Dark Red
        'C': '#888888',  # Gray - Camera
    }

    for idx, img_path in enumerate(samples):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        axes[idx].imshow(img)

        base = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(lbl_dir, base + '.txt')

        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        xc, yc, bw, bh = map(float, parts[1:5])
                        x1 = (xc - bw / 2) * w
                        y1 = (yc - bh / 2) * h
                        box_w = bw * w
                        box_h = bh * h

                        code = class_codes[cls_id] if cls_id < len(class_codes) else '?'
                        color = colors_map.get(code[0], '#FFFFFF')

                        rect = patches.Rectangle(
                            (x1, y1), box_w, box_h,
                            linewidth=2, edgecolor=color, facecolor='none'
                        )
                        axes[idx].add_patch(rect)
                        axes[idx].text(x1, y1 - 4, f"{code}", fontsize=7,
                                       color='white', backgroundcolor=color,
                                       fontweight='bold')

        axes[idx].set_title(os.path.basename(img_path), fontsize=8)
        axes[idx].axis('off')

    for idx in range(len(samples), len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f'Sample {split} images with annotations', fontsize=14)
    plt.tight_layout()
    plt.show()

visualize_sample('train', 6)

#%% ====== CELL 8: TRAIN YOLOv8m ======
# 🎯 Chiến lược đạt mAP > 0.8:
# 1. YOLOv8m - model trung bình, cân bằng tốc độ/chính xác
# 2. imgsz=800 - lớn hơn mặc định, tốt cho biển báo nhỏ
# 3. cos_lr - cosine learning rate scheduler
# 4. label_smoothing - giảm overfitting
# 5. fliplr=0.0 - TẮT flip vì biển rẽ trái/phải sẽ nhầm
# 6. patience=30 - early stopping đủ kiên nhẫn
# 7. Augmentation mạnh nhưng hợp lý

from ultralytics import YOLO

# Load pretrained YOLOv8m
model = YOLO('yolov8m.pt')

# Đường dẫn lưu kết quả trên Drive (không mất khi Colab ngắt)
SAVE_DIR = '/content/drive/My Drive/DACK_training'

results = model.train(
    data=os.path.join(DATASET_DIR, 'data.yaml'),

    # === Core ===
    epochs=150,
    imgsz=800,          # Lớn hơn 640 → detect biển báo nhỏ tốt hơn
    batch=8,            # Giảm batch vì imgsz lớn (T4 16GB VRAM)
    device=0,

    # === Optimizer ===
    optimizer='SGD',
    lr0=0.01,
    lrf=0.01,           # Final LR = lr0 * lrf
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=5,
    warmup_momentum=0.8,
    cos_lr=True,         # Cosine LR scheduler → smooth convergence

    # === Augmentation ===
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=15.0,        # Xoay nhẹ
    translate=0.15,
    scale=0.5,
    shear=2.0,
    perspective=0.0001,
    fliplr=0.0,          # ⚠️ TẮT flip ngang (rẽ trái ≠ rẽ phải)
    flipud=0.0,
    mosaic=1.0,
    mixup=0.15,
    copy_paste=0.1,

    # === Regularization ===
    label_smoothing=0.05,  # Giảm overfitting
    dropout=0.1,           # Thêm dropout nhẹ

    # === Early Stopping ===
    patience=30,

    # === Save ===
    project=SAVE_DIR,
    name='vn_traffic_signs_v1',
    save=True,
    save_period=20,       # Save checkpoint mỗi 20 epochs
    plots=True,

    # === Other ===
    workers=2,
    seed=42,
    verbose=True,
)

print("✅ Training complete!")
print(f"📁 Results saved to: {SAVE_DIR}/vn_traffic_signs_v1")

#%% ====== CELL 9: EVALUATE ON TEST SET ======
from ultralytics import YOLO

# Load best model
SAVE_DIR = '/content/drive/My Drive/DACK_training'
best_model_path = os.path.join(SAVE_DIR, 'vn_traffic_signs_v1', 'weights', 'best.pt')
model = YOLO(best_model_path)

# Validate on test set
metrics = model.val(
    data=os.path.join(DATASET_DIR, 'data.yaml'),
    split='test',
    imgsz=800,
    batch=8,
    conf=0.25,
    iou=0.6,
    device=0,
    plots=True,
    save_json=True,
)

print("\n" + "=" * 60)
print("📊 TEST SET RESULTS")
print("=" * 60)
print(f"  mAP@0.5:      {metrics.box.map50:.4f}")
print(f"  mAP@0.5:0.95:  {metrics.box.map:.4f}")
print(f"  Precision:      {metrics.box.mp:.4f}")
print(f"  Recall:         {metrics.box.mr:.4f}")

if metrics.box.map50 >= 0.8:
    print("\n🎉 TARGET ACHIEVED! mAP@0.5 >= 0.8")
else:
    print(f"\n⚠️  mAP@0.5 = {metrics.box.map50:.4f} < 0.8")
    print("   Tips: Thử tăng epochs, dùng imgsz=1024, hoặc thêm data")

# Per-class AP
print("\n📋 Per-class AP@0.5:")
if hasattr(metrics.box, 'ap50') and metrics.box.ap50 is not None:
    ap50 = metrics.box.ap50
    for i, ap in enumerate(ap50):
        code = class_codes[i] if i < len(class_codes) else f"Class_{i}"
        status = "✅" if ap >= 0.8 else "⚠️" if ap >= 0.5 else "❌"
        print(f"  {status} [{i:2d}] {code:<20s} AP={ap:.4f}")

#%% ====== CELL 10: VISUALIZE TRAINING RESULTS ======
from IPython.display import Image, display

results_dir = os.path.join(SAVE_DIR, 'vn_traffic_signs_v1')

# Show training curves
plots = [
    'results.png',
    'confusion_matrix.png',
    'confusion_matrix_normalized.png',
    'P_curve.png',
    'R_curve.png',
    'F1_curve.png',
    'PR_curve.png',
]

for plot_name in plots:
    plot_path = os.path.join(results_dir, plot_name)
    if os.path.exists(plot_path):
        print(f"\n📈 {plot_name}:")
        display(Image(filename=plot_path, width=800))
    else:
        print(f"⚠️ {plot_name} not found")

#%% ====== CELL 11: TEST INFERENCE ON SAMPLE IMAGES ======
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

SAVE_DIR = '/content/drive/My Drive/DACK_training'
best_model_path = os.path.join(SAVE_DIR, 'vn_traffic_signs_v1', 'weights', 'best.pt')
model = YOLO(best_model_path)

# Test on random images from test set
test_img_dir = os.path.join(DATASET_DIR, 'test', 'images')
test_images = sorted(glob.glob(os.path.join(test_img_dir, '*')))
samples = random.sample(test_images, min(8, len(test_images)))

fig, axes = plt.subplots(2, 4, figsize=(24, 12))
axes = axes.flatten()

for idx, img_path in enumerate(samples):
    # Predict with TTA (Test Time Augmentation) for best accuracy
    results = model.predict(
        source=img_path,
        conf=0.5,
        iou=0.5,
        imgsz=800,
        augment=True,  # TTA: multi-scale inference
        verbose=False,
    )

    # Draw results
    annotated = results[0].plot(
        line_width=2,
        font_size=10,
    )
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    axes[idx].imshow(annotated)
    axes[idx].set_title(os.path.basename(img_path), fontsize=9)
    axes[idx].axis('off')

for idx in range(len(samples), len(axes)):
    axes[idx].axis('off')

plt.suptitle('🚦 YOLOv8m Predictions on Test Images (TTA enabled)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'vn_traffic_signs_v1', 'test_predictions.png'), dpi=150)
plt.show()

#%% ====== CELL 12: SAVE FINAL MODEL TO DRIVE ======
import shutil

SAVE_DIR = '/content/drive/My Drive/DACK_training'
weights_dir = os.path.join(SAVE_DIR, 'vn_traffic_signs_v1', 'weights')

# Copy model file to an easy-to-find location
final_model_dir = '/content/drive/My Drive/DACK_model'
os.makedirs(final_model_dir, exist_ok=True)

best_src = os.path.join(weights_dir, 'best.pt')
best_dst = os.path.join(final_model_dir, 'vn_traffic_signs_yolov8m_best.pt')

if os.path.exists(best_src):
    shutil.copy2(best_src, best_dst)
    file_size = os.path.getsize(best_dst) / (1024 * 1024)
    print(f"✅ Best model saved to: {best_dst}")
    print(f"   Size: {file_size:.1f} MB")
else:
    print("❌ best.pt not found!")

# Also save class names for inference
class_info = {
    'class_codes': class_codes,
    'class_names_en': class_names_en,
    'class_names_vie': class_names_vie,
}

import json
with open(os.path.join(final_model_dir, 'class_info.json'), 'w', encoding='utf-8') as f:
    json.dump(class_info, f, ensure_ascii=False, indent=2)

print(f"✅ Class info saved to: {final_model_dir}/class_info.json")
print(f"\n📥 Tải về máy local:")
print(f"   1. Vào Google Drive → DACK_model")
print(f"   2. Download file: vn_traffic_signs_yolov8m_best.pt")
print(f"   3. Download file: class_info.json")
print(f"   4. Chạy: python inference.py --model best.pt --source 0")
