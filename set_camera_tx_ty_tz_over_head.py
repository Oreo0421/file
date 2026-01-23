#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
from plyfile import PlyData

def quat_to_Rwc(qw, qx, qy, qz):
    """
    COLMAP convention: quaternion (qw,qx,qy,qz) represents rotation from world to camera: R_wc
    """
    w, x, y, z = float(qw), float(qx), float(qy), float(qz)
    # Normalize just in case
    n = np.sqrt(w*w + x*x + y*y + z*z) + 1e-12
    w, x, y, z = w/n, x/n, y/n, z/n

    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),         2*(x*z + y*w)],
        [2*(x*y + z*w),         1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [2*(x*z - y*w),         2*(y*z + x*w),         1 - 2*(x*x + y*y)]
    ], dtype=np.float64)
    return R

def load_xyz_from_ply(ply_path: str) -> np.ndarray:
    ply = PlyData.read(ply_path)
    v = ply["vertex"]
    xyz = np.stack(
        [np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])],
        axis=1
    ).astype(np.float64)
    return xyz

def estimate_point(xyz: np.ndarray, mode: str, up_axis: int, top_percent: float) -> np.ndarray:
    """
    mode:
      - head: use the top `top_percent` points along `up_axis` and average them
      - center: mean of all points
    """
    if mode == "center":
        return xyz.mean(axis=0)

    if mode != "head":
        raise ValueError(f"Unknown mode: {mode}")

    k = max(1, int(len(xyz) * (top_percent / 100.0)))
    idx = np.argpartition(xyz[:, up_axis], -k)[-k:]
    return xyz[idx].mean(axis=0)

def parse_images_header_line(line: str):
    """
    COLMAP images.txt header line:
    IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
    """
    parts = line.strip().split()
    if len(parts) < 10:
        return None
    image_id = int(parts[0])
    qw, qx, qy, qz = map(float, parts[1:5])
    tx, ty, tz = map(float, parts[5:8])
    return image_id, qw, qx, qy, qz, tx, ty, tz, parts

def main():
    ap = argparse.ArgumentParser(
        description="Modify COLMAP images.txt TX TY TZ to place camera 'over head' of a person pointcloud, keeping quaternion unchanged."
    )
    ap.add_argument("--images_txt", required=True, help="Input COLMAP images.txt")
    ap.add_argument("--human_ply", required=True, help="Transformed human ply (same world you want camera to align to)")
    ap.add_argument("--out_images_txt", required=True, help="Output images.txt path")
    ap.add_argument("--image_id", type=int, default=0, help="Which image line to modify (default: 0)")
    ap.add_argument("--distance", type=float, default=2.5, help="Distance from target point along camera forward direction (default: 2.5)")
    ap.add_argument("--mode", choices=["head", "center"], default="head",
                    help="Target point: 'head' (top points) or 'center' (mean of all points)")
    ap.add_argument("--up_axis", type=int, choices=[0,1,2], default=2,
                    help="Axis used as 'up' when mode=head: 0=x,1=y,2=z (default: 2)")
    ap.add_argument("--top_percent", type=float, default=0.5,
                    help="For mode=head: use top X percent points (default: 0.5)")
    ap.add_argument("--forward_axis", type=str, default="z",
                    choices=["z", "-z"],
                    help="Camera forward axis in camera coords. Usually +z in COLMAP; use -z if your convention differs.")
    args = ap.parse_args()

    xyz = load_xyz_from_ply(args.human_ply)
    P = estimate_point(xyz, mode=args.mode, up_axis=args.up_axis, top_percent=args.top_percent)

    with open(args.images_txt, "r", encoding="utf-8") as f:
        lines = f.readlines()

    changed = False
    out_lines = []

    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            out_lines.append(line)
            continue

        parsed = parse_images_header_line(line)
        if parsed is None:
            out_lines.append(line)
            continue

        image_id, qw, qx, qy, qz, tx, ty, tz, parts = parsed

        if image_id != args.image_id:
            out_lines.append(line)
            continue

        Rwc = quat_to_Rwc(qw, qx, qy, qz)

        # camera forward direction in world:
        cam_forward_cam = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if args.forward_axis == "-z":
            cam_forward_cam *= -1.0

        v_world = Rwc.T @ cam_forward_cam
        v_world = v_world / (np.linalg.norm(v_world) + 1e-12)

        # desired camera center in world
        C_star = P - args.distance * v_world

        # desired t_wc (TX TY TZ)
        t_star = -Rwc @ C_star

        # for logging / sanity
        old_t = np.array([tx, ty, tz], dtype=np.float64)
        old_C = -Rwc.T @ old_t

        print("=== Target setup ===")
        print(f"image_id = {args.image_id}")
        print(f"mode     = {args.mode}")
        print(f"P_target = {P}")
        print(f"distance = {args.distance}")
        print(f"v_world  = {v_world}")
        print("=== Old pose ===")
        print(f"old_t    = {old_t}")
        print(f"old_C    = {old_C}")
        print("=== New pose ===")
        print(f"new_t    = {t_star}")
        print(f"new_C    = {C_star}")

        # write back into the same line (keep everything else unchanged)
        parts[5] = f"{t_star[0]:.6f}"
        parts[6] = f"{t_star[1]:.6f}"
        parts[7] = f"{t_star[2]:.6f}"
        out_lines.append(" ".join(parts) + "\n")
        changed = True

    if not changed:
        raise RuntimeError(f"Did not find image_id={args.image_id} header line to modify in {args.images_txt}")

    with open(args.out_images_txt, "w", encoding="utf-8") as f:
        f.writelines(out_lines)

    print(f"\nWrote modified images.txt to: {args.out_images_txt}")
    print("Next: re-run your extract-camera step to regenerate camera json, then render.")

if __name__ == "__main__":
    main()
