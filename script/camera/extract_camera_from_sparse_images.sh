#!/usr/bin/env bash

# 用法: ./process_txt_and_run.sh your_file.txt new_camera_name.json

SRC_TXT="$1"
NEW_CAM_NAME="$2"
TARGET_DIR="/home/fzhi/fzt/3dgs_pipeline/ml-hug/ml-hugs/data/neuman/dataset/lab/sparse"
MLHUG_DIR="/home/fzhi/fzt/3dgs_pipeline/ml-hug/ml-hugs"
CAM_DIR="$MLHUG_DIR/output/lab_train_camera_params"

if [ -z "$SRC_TXT" ] || [ -z "$NEW_CAM_NAME" ]; then
    echo "Usage: $0 your_file.txt new_camera_name.json"
    exit 1
fi

if [ ! -f "$SRC_TXT" ]; then
    echo "Error: file not found: $SRC_TXT"
    exit 1
fi

# 1. 进入 sparse 目录
cd "$TARGET_DIR" || exit 1

# 2. 删除旧的 images.txt
if [ -f "images.txt" ]; then
    rm images.txt
    echo "Removed old images.txt"
fi

# 3. 移动并重命名为 images
mv "$SRC_TXT" .
BASENAME=$(basename "$SRC_TXT")
mv "$BASENAME" images
echo "Renamed $BASENAME -> images"

# 4. 运行相机提取
cd "$MLHUG_DIR" || exit 1
python run_extract_camera.py --seq lab --split train --output output

# 5. 进入输出目录并重命名 camera_params.json
cd "$CAM_DIR" || exit 1

if [ ! -f "camera_params.json" ]; then
    echo "Error: camera_params.json not found!"
    exit 1
fi

mv camera_params.json "$NEW_CAM_NAME"
echo "Renamed camera_params.json -> $NEW_CAM_NAME"
