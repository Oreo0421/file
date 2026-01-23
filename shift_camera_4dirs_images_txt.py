#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import numpy as np

def quat_to_Rwc(qw, qx, qy, qz):
    """
    COLMAP convention: quaternion (qw,qx,qy,qz) -> rotation from world to camera (R_wc)
    """
    w, x, y, z = float(qw), float(qx), float(qy), float(qz)
    n = np.sqrt(w*w + x*x + y*y + z*z) + 1e-12
    w, x, y, z = w/n, x/n, y/n, z/n
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),         2*(x*z + y*w)],
        [2*(x*y + z*w),         1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [2*(x*z - y*w),         2*(y*z + x*w),         1 - 2*(x*x + y*y)]
    ], dtype=np.float64)
    return R

def parse_images_header_line(line: str):
    # IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
    parts = line.strip().split()
    if len(parts) < 10:
        return None
    image_id = int(parts[0])
    qw, qx, qy, qz = map(float, parts[1:5])
    tx, ty, tz = map(float, parts[5:8])
    return image_id, qw, qx, qy, qz, tx, ty, tz, parts

def normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n

def project_to_horizontal(v, up_world):
    # remove vertical component so height doesn't change
    return v - np.dot(v, up_world) * up_world

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
        description="Generate 4 images.txt variants by shifting camera left/right/forward/backward on horizontal plane (height unchanged)."
    )
    ap.add_argument("--images_txt", required=True, help="Input COLMAP images.txt (e.g. your overhead-aligned one)")
    ap.add_argument("--out_dir", required=True, help="Output directory for 4 txt files")
    ap.add_argument("--image_id", type=int, default=0, help="Which image line to modify (default: 0)")
    ap.add_argument("--shift", type=float, default=0.5, help="Shift distance in meters/units (default: 0.5)")
    ap.add_argument("--world_up_axis", type=int, choices=[0,1,2], default=2,
                    help="World up axis: 0=x, 1=y, 2=z (default: 2)")
    ap.add_argument("--forward_axis", choices=["z","-z"], default="z",
                    help="Camera forward axis in camera coords for COLMAP (+z usually). Use -z if reversed.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # read the target line once to get q,t
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

    # define world up
    up_world = np.zeros(3, dtype=np.float64)
    up_world[args.world_up_axis] = 1.0

    # camera basis in world
    f_cam = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if args.forward_axis == "-z":
        f_cam *= -1.0
    forward_world = Rwc.T @ f_cam        # camera forward axis in world
    right_world   = Rwc.T @ np.array([1.0, 0.0, 0.0], dtype=np.float64)  # camera right axis in world

    # project to horizontal plane (no height change)
    f_h = normalize(project_to_horizontal(forward_world, up_world))
    r_h = normalize(project_to_horizontal(right_world,   up_world))

    if np.linalg.norm(f_h) < 1e-9 or np.linalg.norm(r_h) < 1e-9:
        raise RuntimeError("Horizontal forward/right vector is near zero. Check world_up_axis or forward_axis.")

    s = args.shift
    centers = {
        "forward":  C + s * f_h,
        "backward": C - s * f_h,
        "right":    C + s * r_h,
        "left":     C - s * r_h,
    }

    print("Base camera center C_world:", C)
    print("Horizontal forward f_h:", f_h)
    print("Horizontal right   r_h:", r_h)
    print("Shift:", s)

    base_name = os.path.splitext(os.path.basename(args.images_txt))[0]

    for k, Ck in centers.items():
        tk = - Rwc @ Ck
        out_path = os.path.join(args.out_dir, f"{base_name}_{k}_{s:.2f}m.txt")
        write_modified_images_txt(args.images_txt, out_path, args.image_id, tk)
        print(f"[{k}] new TX TY TZ =", tk, "->", out_path)

if __name__ == "__main__":
    main()
