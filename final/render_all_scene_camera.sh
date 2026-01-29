#!/usr/bin/env bash

CAMERA_ROOT="/home/fzhi/fzt/3dgs_pipeline/ml_hug/ml-hugs/camera"
SCENE_ROOT="/home/fzhi/fzt/3dgs_pipeline/animatable_dataset/scenefixed"
HUMAN_ROOT="/mnt/data_hdd/fzhi/mid/101010/pose01"
OUT_ROOT="/mnt/data_hdd/fzhi/output/101010/pose01"

START=00000000
END=00000001

for SCENE in ${CAMERA_ROOT}/*; do
    SCENE_NAME=$(basename "$SCENE")   # counter / playroom / djr ...

    SCENE_PT="${SCENE_ROOT}/${SCENE_NAME}/${SCENE_NAME}.pt"
    SCENE_CAMERA_DIR="${CAMERA_ROOT}/${SCENE_NAME}"
    SCENE_HUMAN_DIR="${HUMAN_ROOT}/${SCENE_NAME}"
    SCENE_OUT_DIR="${OUT_ROOT}/${SCENE_NAME}"

    echo "########################################"
    echo "Scene: ${SCENE_NAME}"
    echo "########################################"

    for pdir in ${SCENE_CAMERA_DIR}/p*; do
        P=$(basename "$pdir")   # p1, p2, p3 ...

        HUMAN_PT_DIR="${SCENE_HUMAN_DIR}/${P}/pt"
        OUT_P_DIR="${SCENE_OUT_DIR}/${P}"

        echo "=============================="
        echo " Position: ${P}"
        echo "=============================="

        for cam_json in ${pdir}/*.json; do
            name=$(basename "$cam_json" .json)   # top / forward / backward / left / right
            out_dir="${OUT_P_DIR}/${name}"

            echo "   View: ${name}"
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
done
