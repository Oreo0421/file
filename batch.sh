#!/usr/bin/env bash
set -euo pipefail

# 用法：
#   bash batch_align_to_pose00.sh /path/to/pose_00.npz /path/to/actions_dir [out_dir]
#
# 示例：
#   bash batch_align_to_pose00.sh pose_00.npz ./actions ./trans_out
#   bash batch_align_to_pose00.sh pose_00.npz ./actions   # 输出到 ./actions/trans_out

SRC_NPZ="${1:?Please provide src pose_00.npz}"
IN_DIR="${2:?Please provide input directory containing *.npz}"
OUT_DIR="${3:-${IN_DIR%/}/trans_out}"

MODE="align_first_frame"
INPLACE_MODE="xy"
PREFIX="trans_"

# 这里填你的 trans_npz_all_fix.py 路径（若就在当前目录可保持不变）
TRANS_SCRIPT="trans_npz_all_fix.py"

mkdir -p "$OUT_DIR"

shopt -s nullglob
files=("$IN_DIR"/*.npz)
shopt -u nullglob

if [[ ${#files[@]} -eq 0 ]]; then
  echo "[ERROR] No .npz files found in: $IN_DIR"
  exit 1
fi

echo "[INFO] SRC:     $SRC_NPZ"
echo "[INFO] IN_DIR:  $IN_DIR"
echo "[INFO] OUT_DIR: $OUT_DIR"
echo "[INFO] MODE=$MODE  INPLACE_MODE=$INPLACE_MODE"
echo

for tgt in "${files[@]}"; do
  base="$(basename "$tgt")"
  # 如果你不希望把 pose_00.npz 自己也处理掉，就跳过
  if [[ "$(realpath "$tgt")" == "$(realpath "$SRC_NPZ")" ]]; then
    echo "[SKIP] $base (is SRC)"
    continue
  fi

  out="$OUT_DIR/${PREFIX}${base}"

  echo "[RUN] $base -> $(basename "$out")"
  python "$TRANS_SCRIPT" \
    --src "$SRC_NPZ" \
    --tgt_npz "$tgt" \
    --mode "$MODE" \
    --inplace_mode "$INPLACE_MODE" \
    --out_npz "$out"
done

echo
echo "[DONE] Outputs in: $OUT_DIR"
