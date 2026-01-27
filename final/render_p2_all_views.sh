#!/usr/bin/env bash

SCENE_PT="/home/fzhi/fzt/3dgs_pipeline/animatable_dataset/scenefixed/djr/djr.pt"
CAMERA_ROOT="/home/fzhi/fzt/3dgs_pipeline/ml_hug/ml-hugs/camera/djr"
HUMAN_ROOT="/mnt/data_hdd/fzhi/mid/101010/djr"
OUT_ROOT="/mnt/data_hdd/fzhi/output/101010/djr"

START=00000000
END=00000001

for pdir in ${CAMERA_ROOT}/p*; do
    P=$(basename "$pdir")   # p1, p2, p3 ...

    HUMAN_PT_DIR="${HUMAN_ROOT}/${P}/pt"
    OUT_P_DIR="${OUT_ROOT}/${P}"

    echo "=============================="
    echo "Rendering position: ${P}"
    echo "=============================="

    for cam_json in ${pdir}/*.json; do
        name=$(basename "$cam_json" .json)   # top / forward / backward / left / right
        out_dir="${OUT_P_DIR}/${name}"

        echo "  View: ${name}"
        mkdir -p "${out_dir}"

        python hugs/renderer/render_sequence_from1camera.py \
          --human_pt_dir "${HUMAN_PT_DIR}" \
          --scene_pt "${SCENE_PT}" \
          --output_dir "${out_dir}" \
          --start_frame ${START} \
          --end_frame ${END} \
          --camera_json "${cam_json}" \
          --render_mode human_scene
    done
done
