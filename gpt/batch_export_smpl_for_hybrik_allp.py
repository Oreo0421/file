#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import glob
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from importlib.machinery import SourceFileLoader


def normalize_T4(T):
    """
    Normalize a 4x4 matrix to standard:
      [ R t ]
      [ 0 1 ]
    Some pipelines store translation in last row (tx,ty,tz,1) and last col nearly zeros.
    If detected, transpose it.
    """
    T = np.array(T, dtype=np.float32)
    if T.shape != (4, 4):
        raise ValueError(f"Expect 4x4, got {T.shape}")
    if abs(T[3, 3] - 1.0) < 1e-4 and np.linalg.norm(T[:3, 3]) < 1e-4 and np.linalg.norm(T[3, :3]) > 1e-4:
        T = T.T
    return T.astype(np.float32)


def load_camera_w2c(camera_json_path, frame_idx=0):
    """
    Support common camera json keys:
      world_view_transform (list or single)
      w2c / extrinsic / world_to_camera (list or single)
      c2w (invert)
    If list and length < needed, default uses frame0 (frame_idx=0).
    """
    cam = json.load(open(camera_json_path, "r"))

    if "world_view_transform" in cam:
        wvt = cam["world_view_transform"]
        M = wvt[min(frame_idx, len(wvt) - 1)] if isinstance(wvt, list) else wvt
        return normalize_T4(M)

    for k in ["w2c", "extrinsic", "world_to_camera"]:
        if k in cam:
            M = cam[k]
            M = M[min(frame_idx, len(M) - 1)] if isinstance(M, list) else M
            return normalize_T4(M)

    if "c2w" in cam:
        C = cam["c2w"]
        C = C[min(frame_idx, len(C) - 1)] if isinstance(C, list) else C
        C = normalize_T4(C)
        return np.linalg.inv(C).astype(np.float32)

    raise KeyError(f"Cannot find w2c/world_view_transform/c2w in {camera_json_path}. keys={list(cam.keys())}")


def decompose_A_to_sR(A):
    """A ~ s*R (uniform scale). Project to nearest rotation with SVD."""
    A = np.array(A, dtype=np.float32)
    col_norms = np.linalg.norm(A, axis=0)
    s = float(np.mean(col_norms))
    Rm = A / (s + 1e-8)

    U, _, Vt = np.linalg.svd(Rm)
    Rproj = U @ Vt
    if np.linalg.det(Rproj) < 0:
        U[:, -1] *= -1
        Rproj = U @ Vt
    return s, Rproj.astype(np.float32)


def apply_root_RT(global_orient_aa, trans, R, t):
    """
    Apply root transform:
      R_new = R * R_old
      trans_new = R @ trans + t
    (Row-vector implementation: x' = x R^T + t)
    """
    T = global_orient_aa.shape[0]
    Rg = Rot.from_rotvec(global_orient_aa)
    Rs = Rot.from_matrix(np.repeat(R[None, :, :], T, axis=0))
    Rnew = Rs * Rg
    global_new = Rnew.as_rotvec().astype(np.float32)
    trans_new = (trans @ R.T) + t[None, :]
    return global_new, trans_new.astype(np.float32)


def slice_frames_0_99(poses, trans, start=0, end=99):
    poses = poses[start:end+1]
    trans = trans[start:end+1]
    if poses.shape[0] != (end - start + 1):
        raise ValueError(f"Frame slicing mismatch: got T={poses.shape[0]}, expect {end-start+1}")
    return poses, trans


def save_npz(path, poses_full, trans_xyz, betas, meta: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if betas is None:
        np.savez(path, poses=poses_full.astype(np.float32), trans=trans_xyz.astype(np.float32), **meta)
    else:
        np.savez(path, poses=poses_full.astype(np.float32), trans=trans_xyz.astype(np.float32), betas=betas.astype(np.float32), **meta)


def export_one(motion_npz, scene_transforms_py, scene_key, camera_json, out_dir,
               start=0, end=99, cam_frame_idx=0):
    d = np.load(motion_npz, allow_pickle=True)
    if "poses" not in d.files or "trans" not in d.files:
        raise KeyError(f"{motion_npz} must contain poses/trans. keys={d.files}")

    poses = d["poses"].astype(np.float32)  # (T,156) or similar
    trans = d["trans"].astype(np.float32)  # (T,3)
    betas = d["betas"].astype(np.float32) if "betas" in d.files else None

    poses, trans = slice_frames_0_99(poses, trans, start=start, end=end)
    T = poses.shape[0]
    if T != (end - start + 1):
        raise ValueError(f"Expected {end-start+1} frames, got {T}")

    global_orient = poses[:, 0:3]
    rest_pose = poses[:, 3:]

    # scene transform
    st = SourceFileLoader("st_dyn", scene_transforms_py).load_module()
    if scene_key not in st.SCENE_TRANSFORMS:
        raise KeyError(f"scene_key '{scene_key}' not in SCENE_TRANSFORMS")

    T_scene = np.array(st.SCENE_TRANSFORMS[scene_key], dtype=np.float32)
    A = T_scene[:3, :3]
    t_scene = T_scene[:3, 3]
    scale_s, R_scene = decompose_A_to_sR(A)

    # world_Rt
    glob_w_Rt, trans_w_Rt = apply_root_RT(global_orient, trans, R_scene, t_scene)

    # world_As (A affects translation)
    glob_w_As = glob_w_Rt.copy()
    trans_w_As = (trans @ A.T) + t_scene[None, :]
    trans_w_As = trans_w_As.astype(np.float32)

    # camera w2c (default uses frame0 if list is short)
    W2C = load_camera_w2c(camera_json, frame_idx=cam_frame_idx)
    R_w2c = W2C[:3, :3]
    t_w2c = W2C[:3, 3]

    # cam_Rt from world_Rt
    glob_c_Rt, trans_c_Rt = apply_root_RT(glob_w_Rt, trans_w_Rt, R_w2c, t_w2c)
    # cam_As from world_As
    glob_c_As, trans_c_As = apply_root_RT(glob_w_As, trans_w_As, R_w2c, t_w2c)

    # pack full poses back
    poses_w_Rt = np.concatenate([glob_w_Rt, rest_pose], axis=1)
    poses_w_As = np.concatenate([glob_w_As, rest_pose], axis=1)
    poses_c_Rt = np.concatenate([glob_c_Rt, rest_pose], axis=1)
    poses_c_As = np.concatenate([glob_c_As, rest_pose], axis=1)

    meta_common = dict(
        scene_key=scene_key,
        camera_json=camera_json,
        frame_start=int(start),
        frame_end=int(end),
        scene_scale=np.array([scale_s], dtype=np.float32),
        cam_frame_idx=int(cam_frame_idx),
    )

    save_npz(os.path.join(out_dir, f"smpl_{scene_key}_world_Rt_0_99.npz"), poses_w_Rt, trans_w_Rt, betas,
             dict(**meta_common, tag="world_Rt"))
    save_npz(os.path.join(out_dir, f"smpl_{scene_key}_world_As_0_99.npz"), poses_w_As, trans_w_As, betas,
             dict(**meta_common, tag="world_As"))
    save_npz(os.path.join(out_dir, f"smpl_{scene_key}_cam_Rt_0_99.npz"), poses_c_Rt, trans_c_Rt, betas,
             dict(**meta_common, tag="cam_Rt"))
    save_npz(os.path.join(out_dir, f"smpl_{scene_key}_cam_As_0_99.npz"), poses_c_As, trans_c_As, betas,
             dict(**meta_common, tag="cam_As"))


def iter_motion_npz(motion_root, recursive=True):
    pat = "**/*.npz" if recursive else "*.npz"
    files = glob.glob(os.path.join(motion_root, pat), recursive=recursive)
    files = [f for f in files if os.path.isfile(f)]
    files.sort()
    return files


def iter_camera_json(camera_root, cam_filter=None):
    """
    Expect camera_root structure:
      camera_root/<scene>/<pX>/*.json
    Return list of (scene, p, cam_name, json_path)
    """
    out = []
    for scene_dir in sorted(glob.glob(os.path.join(camera_root, "*"))):
        if not os.path.isdir(scene_dir):
            continue
        scene = os.path.basename(scene_dir)

        for p_dir in sorted(glob.glob(os.path.join(scene_dir, "p*"))):
            if not os.path.isdir(p_dir):
                continue
            p = os.path.basename(p_dir)

            for j in sorted(glob.glob(os.path.join(p_dir, "*.json"))):
                cam_name = os.path.splitext(os.path.basename(j))[0]
                if cam_filter and cam_name != cam_filter:
                    continue
                out.append((scene, p, cam_name, j))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion_root", required=True, help="folder containing motion npz")
    ap.add_argument("--camera_root", required=True, help="camera json root: <scene>/<pX>/*.json")
    ap.add_argument("--scene_transforms_py", required=True, help="path to scene_transforms.py")
    ap.add_argument("--out_root", default="/mnt/data_hdd/fzhi/smpl_for_hybrik", help="output root dir")

    ap.add_argument("--recursive", action="store_true", help="search motions recursively")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=99)

    ap.add_argument("--cam_filter", default=None,
                    help="only export this camera name (e.g. top). default: export ALL json cameras")
    ap.add_argument("--cam_frame_idx", type=int, default=0,
                    help="if camera json stores a list of matrices, pick which index (default 0)")

    ap.add_argument("--dry_run", action="store_true", help="print plan only, do not write files")
    args = ap.parse_args()

    motions = iter_motion_npz(args.motion_root, recursive=args.recursive)
    cams = iter_camera_json(args.camera_root, cam_filter=args.cam_filter)

    if not motions:
        raise RuntimeError(f"No motion npz found under {args.motion_root}")
    if not cams:
        raise RuntimeError(f"No camera json found under {args.camera_root} (cam_filter={args.cam_filter})")

    print(f"[INFO] motions: {len(motions)}")
    print(f"[INFO] cameras: {len(cams)} (cam_filter={args.cam_filter})")
    print(f"[INFO] frames: {args.start}..{args.end} (T={args.end-args.start+1})")
    print(f"[INFO] out_root: {args.out_root}")

    fail = 0
    for m in motions:
        motion_name = os.path.splitext(os.path.basename(m))[0]
        for scene, p, cam_name, cam_json in cams:
            # scene_key naming convention: <scene>_<pX> e.g. room_p1, djr_p2
            scene_key = f"{scene}_{p}"

            out_dir = os.path.join(args.out_root, motion_name, scene, p, cam_name)

            if args.dry_run:
                print(f"[PLAN] {motion_name} | {scene_key} | {cam_name} -> {out_dir}")
                continue

            try:
                export_one(
                    motion_npz=m,
                    scene_transforms_py=args.scene_transforms_py,
                    scene_key=scene_key,
                    camera_json=cam_json,
                    out_dir=out_dir,
                    start=args.start,
                    end=args.end,
                    cam_frame_idx=args.cam_frame_idx,
                )
                print(f"[OK] {motion_name} | {scene_key} | {cam_name}")
            except Exception as e:
                fail += 1
                print(f"[FAIL] {motion_name} | {scene_key} | {cam_name}: {e}")

    print(f"[DONE] fails={fail}")


if __name__ == "__main__":
    main()
