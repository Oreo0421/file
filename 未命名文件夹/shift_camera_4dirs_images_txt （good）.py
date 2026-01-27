#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
from plyfile import PlyData

def quat_to_Rwc(qw, qx, qy, qz):
    w, x, y, z = map(float, [qw, qx, qy, qz])
    n = (w*w + x*x + y*y + z*z) ** 0.5 + 1e-12
    w, x, y, z = w/n, x/n, y/n, z/n
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),         2*(x*z + y*w)],
        [2*(x*y + z*w),         1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [2*(x*z - y*w),         2*(y*z + x*w),         1 - 2*(x*x + y*y)]
    ], dtype=np.float64)

def parse_images_header_line(line: str):
    parts = line.strip().split()
    if len(parts) < 10:
        return None
    image_id = int(parts[0])
    qw, qx, qy, qz = map(float, parts[1:5])
    tx, ty, tz = map(float, parts[5:8])
    return image_id, qw, qx, qy, qz, tx, ty, tz, parts

def normalize(v):
    n = np.linalg.norm(v)
    return v / (n + 1e-12)

def project_to_horizontal(v, up_world):
    return v - np.dot(v, up_world) * up_world

def load_xyz_from_ply(ply_path: str) -> np.ndarray:
    ply = PlyData.read(ply_path)
    v = ply["vertex"]
    xyz = np.stack(
        [np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])],
        axis=1
    ).astype(np.float64)
    return xyz

def auto_choose_up_axis_from_ply(human_ply: str) -> int:
    xyz = load_xyz_from_ply(human_ply)
    mn = xyz.min(axis=0)
    mx = xyz.max(axis=0)
    ranges = mx - mn
    up_axis = int(np.argmax(ranges))
    print(f"[world_up_axis=auto] ranges={ranges}, choose axis {up_axis} ({['X','Y','Z'][up_axis]}) as UP")
    return up_axis

def write_modified_images_txt(in_path, out_path, image_id_to_change, new_t):
    with open(in_path, "r", encoding="utf-8") as f:
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
        if image_id != image_id_to_change:
            out_lines.append(line)
            continue

        parts[5] = f"{new_t[0]:.6f}"
        parts[6] = f"{new_t[1]:.6f}"
        parts[7] = f"{new_t[2]:.6f}"
        out_lines.append(" ".join(parts) + "\n")
        changed = True

    if not changed:
        raise RuntimeError(f"Did not find image_id={image_id_to_change} in {in_path}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(out_lines)

def main():
    ap = argparse.ArgumentParser(
        description="Generate 4 images.txt variants by shifting camera left/right/forward/back on horizontal plane (height unchanged)."
    )
    ap.add_argument("--images_txt", required=True, help="Input COLMAP images.txt (use your overhead-aligned one)")
    ap.add_argument("--out_dir", required=True, help="Output directory for 4 txt files")
    ap.add_argument("--image_id", type=int, default=0, help="Which image line to modify (default: 0)")
    ap.add_argument("--shift", type=float, default=0.5, help="Shift distance in meters/units (default: 0.5)")

    ap.add_argument("--world_up_axis", type=str, default="auto",
                    help="World up axis: 0=x,1=y,2=z, or 'auto' (default: auto)")
    ap.add_argument("--human_ply", default=None,
                    help="If world_up_axis=auto, provide a human ply to infer up axis by largest range.")

    # NEW: auto right/forward axes based on up-axis
    ap.add_argument("--auto_axes", action="store_true",
                    help="Auto choose right_axis/forward_world_axis as the two axes different from up axis.")

    ap.add_argument("--right_axis", type=int, choices=[0,1,2], default=0,
                    help="Define 'right' direction in WORLD as +axis (default: +X => 0)")
    ap.add_argument("--forward_world_axis", type=int, choices=[0,1,2], default=1,
                    help="Define 'forward' direction in WORLD as +axis (default: +Y => 1)")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Resolve world_up_axis
    if args.world_up_axis.lower() == "auto":
        if args.human_ply is None:
            raise RuntimeError("world_up_axis=auto requires --human_ply to infer up axis.")
        up_axis = auto_choose_up_axis_from_ply(args.human_ply)
    else:
        up_axis = int(args.world_up_axis)
        if up_axis not in (0,1,2):
            raise ValueError("--world_up_axis must be 0/1/2 or 'auto'")
        print(f"[world_up_axis=fixed] using axis {up_axis} ({['X','Y','Z'][up_axis]}) as UP")

    # Auto choose right/forward axes if requested
    right_axis = args.right_axis
    forward_axis = args.forward_world_axis
    if args.auto_axes:
        other = [a for a in (0,1,2) if a != up_axis]
        # deterministic order: smaller index -> right, larger -> forward
        right_axis, forward_axis = other[0], other[1]
        print(f"[auto_axes] up={['X','Y','Z'][up_axis]} -> right={['X','Y','Z'][right_axis]}, forward={['X','Y','Z'][forward_axis]}")

    up_world = np.zeros(3, dtype=np.float64)
    up_world[up_axis] = 1.0

    # Read target q,t
    with open(args.images_txt, "r", encoding="utf-8") as f:
        lines = f.readlines()

    target = None
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parsed = parse_images_header_line(line)
        if parsed is None:
            continue
        image_id, qw, qx, qy, qz, tx, ty, tz, parts = parsed
        if image_id == args.image_id:
            target = (qw, qx, qy, qz, np.array([tx,ty,tz], dtype=np.float64))
            break

    if target is None:
        raise RuntimeError(f"Could not find image_id={args.image_id} in {args.images_txt}")

    qw, qx, qy, qz, t = target
    Rwc = quat_to_Rwc(qw, qx, qy, qz)

    # camera center in world
    C = - Rwc.T @ t

    # movement directions in WORLD
    right_w = np.zeros(3, dtype=np.float64);  right_w[right_axis] = 1.0
    fwd_w   = np.zeros(3, dtype=np.float64);  fwd_w[forward_axis] = 1.0

    # project to horizontal plane
    right_h = normalize(project_to_horizontal(right_w, up_world))
    fwd_h   = normalize(project_to_horizontal(fwd_w,   up_world))

    if np.linalg.norm(right_h) < 1e-9 or np.linalg.norm(fwd_h) < 1e-9:
        raise RuntimeError(
            "Right/Forward direction becomes zero after removing UP component. "
            "Use --auto_axes or pick different --right_axis/--forward_world_axis."
        )

    s = args.shift
    centers = {
        "forward":  C + s * fwd_h,
        "backward": C - s * fwd_h,
        "right":    C + s * right_h,
        "left":     C - s * right_h,
    }

    print("Base camera center C_world:", C)
    print("UP axis:", up_axis, ['X','Y','Z'][up_axis], " up_world=", up_world)
    print("right_h:", right_h, "  (world axis =", ['X','Y','Z'][right_axis], ")")
    print("fwd_h  :", fwd_h,   "  (world axis =", ['X','Y','Z'][forward_axis], ")")
    print("Shift  :", s)

    base_name = os.path.splitext(os.path.basename(args.images_txt))[0]

    for k, Ck in centers.items():
        tk = - Rwc @ Ck
        out_path = os.path.join(args.out_dir, f"{base_name}_{k}_{s:.2f}m.txt")
        write_modified_images_txt(args.images_txt, out_path, args.image_id, tk)
        print(f"[{k}] new TX TY TZ =", tk, "->", out_path)

if __name__ == "__main__":
    main()
