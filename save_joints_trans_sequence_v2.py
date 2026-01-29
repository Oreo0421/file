#!/usr/bin/env python3
"""
Process per-frame 8-digit .npy joint files from a folder and (optionally) apply a 4x4 transform
loaded from scene_transforms.py.

NEW output structure (everything under a single 'joint' folder):
  output_dir/
    joint/
      json/original/00000000.json
      json/transformed/00000000.json
      npy/original/00000000.npy
      npy/transformed/00000000.npy
      npz/{subject}_{pose}_{scene}_all_original_joints.npz
      npz/{subject}_{pose}_{scene}_all_transformed_joints.npz

Filename prefix rule for NPZ:
  {subject}_{pose_name}_{scene_name}

pose_name priority:
  1) --pose_name (explicit)
  2) --motion basename (without .npz)
  3) auto infer from input_dir by going up N parents (default N=4) -> folder name

subject priority:
  1) --subject_name (explicit)
  2) auto infer from output_dir path by finding the first 6+ digit numeric folder name (e.g. 101010)

Usage examples:
  # minimal (auto subject + auto pose_name)
  python save_joints_trans_sequence.py \
    --input_dir /path/to/.../joints \
    --output_dir /mnt/data_hdd/fzhi/output/101010/06_13_poses_inplace_stride5/djr/p1 \
    --scene djr_p1

  # explicit pose_name + subject_name
  python save_joints_trans_sequence.py \
    --input_dir /path/to/.../joints \
    --output_dir /path/to/out \
    --scene djr_p1 \
    --pose_name 06_13_poses_inplace_stride5 \
    --subject_name 101010

  # use motion basename as pose_name
  python save_joints_trans_sequence.py \
    --input_dir /path/to/.../joints \
    --output_dir /path/to/out \
    --scene djr_p1 \
    --motion 06_13_poses.npz
"""

import os
import re
import json
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm

from scene_transforms import SCENE_TRANSFORMS


def infer_pose_name_from_input_dir(input_dir: str, up: int = 4) -> str:
    """Auto infer pose_name from input_dir by going up N parents and using that folder name."""
    p = os.path.abspath(input_dir)
    for _ in range(max(0, up)):
        p = os.path.dirname(p)
    name = os.path.basename(p)
    return name if name else "unknown_pose"


def infer_subject_name_from_output_dir(output_dir: str) -> str:
    """
    Try to infer subject id (e.g., 101010) from output_dir path components.
    Picks the first folder name that looks like a numeric id with 6+ digits.
    """
    parts = [p for p in os.path.abspath(output_dir).split(os.sep) if p]
    for part in parts:
        if re.fullmatch(r"\d{6,}", part):
            return part
    return "unknown_subject"


def get_pose_name(args) -> str:
    # Priority 1: explicit pose_name
    if getattr(args, "pose_name", None):
        return args.pose_name

    # Priority 2: motion basename
    if getattr(args, "motion", None):
        base = os.path.basename(args.motion)
        return os.path.splitext(base)[0]

    # Priority 3: auto infer from input_dir parents
    return infer_pose_name_from_input_dir(args.input_dir, up=args.auto_up)


def get_subject_name(args) -> str:
    if getattr(args, "subject_name", None):
        return str(args.subject_name)
    return infer_subject_name_from_output_dir(args.output_dir)


def process_joints_folder(
    input_dir: str,
    output_dir: str,
    scene_name: str,
    pose_name: str,
    subject_name: str,
    apply_transform: bool = True,
):
    # ---- Unified base folder ----
    joint_dir = os.path.join(output_dir, "joint")

    json_orig_dir = os.path.join(joint_dir, "json", "original")
    json_trans_dir = os.path.join(joint_dir, "json", "transformed")

    npy_orig_dir = os.path.join(joint_dir, "npy", "original")
    npy_trans_dir = os.path.join(joint_dir, "npy", "transformed")

    npz_dir = os.path.join(joint_dir, "npz")

    os.makedirs(json_orig_dir, exist_ok=True)
    os.makedirs(npy_orig_dir, exist_ok=True)
    os.makedirs(npz_dir, exist_ok=True)

    if apply_transform:
        os.makedirs(json_trans_dir, exist_ok=True)
        os.makedirs(npy_trans_dir, exist_ok=True)

    # Load transform from registry
    if scene_name not in SCENE_TRANSFORMS:
        raise KeyError(
            f"Scene '{scene_name}' not found in scene_transforms.py. "
            f"Available keys example: {list(SCENE_TRANSFORMS.keys())[:10]}"
        )
    trans_matrix = np.array(SCENE_TRANSFORMS[scene_name], dtype=np.float64)
    # ---- Print transform used (only once) ----
    if not hasattr(process_joints_folder, "_printed_T"):
        process_joints_folder._printed_T = set()

    key = (subject_name, pose_name, scene_name)
    if key not in process_joints_folder._printed_T:
        process_joints_folder._printed_T.add(key)
        print("\n[Scene Transform Used]")
        print(f"  subject   : {subject_name}")
        print(f"  pose_name : {pose_name}")
        print(f"  scene     : {scene_name}")
        np.set_printoptions(precision=12, suppress=True)
        print("  T_align:\n", trans_matrix)
        print("")


    # Collect 8-digit .npy files (e.g., 00000000.npy)
    pattern = re.compile(r"^\d{8}\.npy$")
    files = sorted([f for f in os.listdir(input_dir) if pattern.match(f)])

    if not files:
        print(f"No 8-digit .npy files found in {input_dir}")
        return

    all_original = []
    all_transformed = []

    for fname in tqdm(files, desc="Processing joints"):
        frame_idx = int(os.path.splitext(fname)[0])
        src_path = os.path.join(input_dir, fname)

        joints_3d = np.load(src_path)  # expected shape (J, 3)
        joints_3d = np.asarray(joints_3d)

        if joints_3d.ndim != 2 or joints_3d.shape[1] != 3:
            print(f"Skipping {fname}: expected shape (J,3), got {joints_3d.shape}")
            continue

        # ---- Save original ----
        np.save(os.path.join(npy_orig_dir, f"{frame_idx:08d}.npy"), joints_3d.astype(np.float32))
        with open(os.path.join(json_orig_dir, f"{frame_idx:08d}.json"), "w") as f:
            json.dump(
                {"frame_idx": frame_idx, "joints_3d": joints_3d.tolist()},
                f,
                indent=2,
            )

        all_original.append(joints_3d)

        # ---- Save transformed (optional) ----
        if apply_transform:
            joints_h = np.hstack(
                [joints_3d, np.ones((joints_3d.shape[0], 1), dtype=joints_3d.dtype)]
            )  # (J,4)

            transformed = (trans_matrix @ joints_h.T).T[:, :3]

            np.save(os.path.join(npy_trans_dir, f"{frame_idx:08d}.npy"), transformed.astype(np.float32))
            with open(os.path.join(json_trans_dir, f"{frame_idx:08d}.json"), "w") as f:
                json.dump(
                    {
                        "frame_idx": frame_idx,
                        "joints_3d": transformed.tolist(),
                        "transformation_matrix": trans_matrix.tolist(),
                    },
                    f,
                    indent=2,
                )

            all_transformed.append(transformed)

    # ---- Combined arrays (NPZ) ----
    prefix = f"{subject_name}_{pose_name}_{scene_name}"

    if len(all_original) > 0:
        np.savez(
            os.path.join(npz_dir, f"{prefix}_all_original_joints.npz"),
            joints_3d=np.array(all_original, dtype=np.float32),
        )

    if apply_transform and len(all_transformed) > 0:
        np.savez(
            os.path.join(npz_dir, f"{prefix}_all_transformed_joints.npz"),
            joints_3d=np.array(all_transformed, dtype=np.float32),
            transformation_matrix=trans_matrix,
        )

    print(f"Processed {len(files)} frames (saved {len(all_original)} valid).")
    print("Outputs:")
    print(f"  JSON: {os.path.join(joint_dir, 'json')}")
    print(f"  NPY : {os.path.join(joint_dir, 'npy')}")
    print(f"  NPZ : {os.path.join(joint_dir, 'npz', f'{prefix}_all_*.npz')}")
    if apply_transform:
        print(f"  using scene transform: {scene_name}")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Transform per-frame .npy joints in a folder (outputs under output_dir/joint/...) "
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Folder containing 8-digit joint .npy files (00000000.npy...)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Folder to save processed joints (will create output_dir/joint/...)",
    )
    parser.add_argument(
        "--scene",
        type=str,
        required=True,
        help="Scene key in scene_transforms.py (e.g., djr_p1)",
    )

    parser.add_argument(
        "--subject_name",
        type=str,
        default=None,
        help="Subject id to prefix NPZ name (e.g., 101010). If omitted, inferred from output_dir path.",
    )

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--motion",
        type=str,
        default=None,
        help="Pose/motion npz path (pose_name uses its basename without extension)",
    )
    group.add_argument(
        "--pose_name",
        type=str,
        default=None,
        help="Pose name string to use in NPZ prefix",
    )

    parser.add_argument(
        "--auto_up",
        type=int,
        default=4,
        help="Auto-infer pose_name from input_dir: go up N parents and use folder name (default: 4)",
    )

    parser.add_argument(
        "--no-transform",
        action="store_true",
        help="Skip transformation (only save original)",
    )

    args = parser.parse_args()

    pose_name = get_pose_name(args)
    subject_name = get_subject_name(args)

    process_joints_folder(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        scene_name=args.scene,
        pose_name=pose_name,
        subject_name=subject_name,
        apply_transform=not args.no_transform,
    )
