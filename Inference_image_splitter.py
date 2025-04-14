import os
from pathlib import Path
import cv2
import torch

model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/blackBugOnly/weights/best.pt')


# CONFIG
INPUT_IMAGE_DIR = "C:/Users/dell/datasets/bugs/tests"

# Grid config based on your image size
ORIG_WIDTH = 3024
ORIG_HEIGHT = 4032
TILE_GRID_X = 4  # Number of tiles across width
TILE_GRID_Y = 6  # Number of tiles across height

TILE_W = ORIG_WIDTH // TILE_GRID_X   # 756px
TILE_H = ORIG_HEIGHT // TILE_GRID_Y  # 672px

for img_file in os.listdir(INPUT_IMAGE_DIR):
    if not img_file.lower().endswith(('.jpg', '.png')):
        continue

    base_name = Path(img_file).stem
    image_path = os.path.join(INPUT_IMAGE_DIR, img_file)

    img = cv2.imread(image_path)
    img_h, img_w = img.shape[:2]

    tiled_images = []

    for row in range(TILE_GRID_Y):
        for col in range(TILE_GRID_X):
            x_offset = col * TILE_W
            y_offset = row * TILE_H
            tile_img = img[y_offset:y_offset+TILE_H, x_offset:x_offset+TILE_W]

            tiled_images.append(tile_img)
            tile_name = f"{base_name}_{row}_{col}"

    results = model(tiled_images)
    results.save()
    for result in results:
        print(len(result.summary()))