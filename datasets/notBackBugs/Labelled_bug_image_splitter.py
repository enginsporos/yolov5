import os
from pathlib import Path
import cv2
from matplotlib import pyplot as plt
import numpy as np

# CONFIG
INPUT_IMAGE_DIR = "images/"
INPUT_LABEL_DIR = "labels/"
OUTPUT_IMAGE_DIR = "dataset_tiled/images"
OUTPUT_LABEL_DIR = "dataset_tiled/labels"

# Grid config based on your image size
ORIG_WIDTH = 3024
ORIG_HEIGHT = 4032
TILE_GRID_X = 4  # Number of tiles across width
TILE_GRID_Y = 6  # Number of tiles across height

TILE_W = ORIG_WIDTH // TILE_GRID_X   # 756px
TILE_H = ORIG_HEIGHT // TILE_GRID_Y  # 672px

os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

def denormalize(bbox, img_w, img_h):
    xc, yc, w, h = bbox
    x1 = (xc - w / 2) * img_w
    y1 = (yc - h / 2) * img_h
    x2 = (xc + w / 2) * img_w
    y2 = (yc + h / 2) * img_h
    return x1, y1, x2, y2

def normalize(x1, y1, x2, y2, tile_w, tile_h):
    xc = (x1 + x2) / 2 / tile_w
    yc = (y1 + y2) / 2 / tile_h
    w = (x2 - x1) / tile_w
    h = (y2 - y1) / tile_h
    return xc, yc, w, h

for img_file in os.listdir(INPUT_IMAGE_DIR):
    if not img_file.lower().endswith(('.jpg', '.png')):
        continue

    base_name = Path(img_file).stem
    image_path = os.path.join(INPUT_IMAGE_DIR, img_file)
    label_path = os.path.join(INPUT_LABEL_DIR, base_name + ".txt")

    if not os.path.exists(label_path):
        print(f"⚠️ No label file for {img_file}")
        continue

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    __,thresh1 = cv2.threshold(gray,110,255,cv2.THRESH_BINARY) 
    contours, _ = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_contour = contours[0]
    for contour in contours:
        if cv2.contourArea(contour)>cv2.contourArea(max_contour):
            max_contour = contour

    mask = np.zeros((gray.shape), np.uint8)
    cv2.drawContours(mask, [max_contour], 0, 255, -1)
    cv2.drawContours(mask, [max_contour], 0, 255, 20)
    img = cv2.bitwise_and(img, img, mask = mask)

    img_h, img_w = img.shape[:2]

    with open(label_path, 'r') as f:
        lines = f.read().strip().splitlines()

    boxes = []
    for line in lines:
        parts = line.split()
        cls = int(parts[0])
        bbox = list(map(float, parts[1:]))
        x1, y1, x2, y2 = denormalize(bbox, img_w, img_h)
        boxes.append((cls, x1, y1, x2, y2))

    for row in range(TILE_GRID_Y):
        for col in range(TILE_GRID_X):
            x_offset = col * TILE_W
            y_offset = row * TILE_H
            tile_img = img[y_offset:y_offset+TILE_H, x_offset:x_offset+TILE_W]

            tile_name = f"{base_name}_{row}_{col}"
            out_img_path = os.path.join(OUTPUT_IMAGE_DIR, tile_name + ".jpg")
            out_label_path = os.path.join(OUTPUT_LABEL_DIR, tile_name + ".txt")

            tile_labels = []
            for cls, x1, y1, x2, y2 in boxes:
                if x2 < x_offset or x1 > x_offset + TILE_W:
                    continue
                if y2 < y_offset or y1 > y_offset + TILE_H:
                    continue

                x1_tile = max(x1, x_offset) - x_offset
                y1_tile = max(y1, y_offset) - y_offset
                x2_tile = min(x2, x_offset + TILE_W) - x_offset
                y2_tile = min(y2, y_offset + TILE_H) - y_offset

                xc, yc, w, h = normalize(x1_tile, y1_tile, x2_tile, y2_tile, TILE_W, TILE_H)
                if w <= 0 or h <= 0:
                    continue

                tile_labels.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                
            if tile_labels:
                cv2.imwrite(out_img_path, tile_img)
                with open(out_label_path, 'w') as f:
                    f.write("\n".join(tile_labels))
