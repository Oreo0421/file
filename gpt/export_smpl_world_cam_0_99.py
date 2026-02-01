#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, argparse
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from importlib.machinery import SourceFileLoader


def normalize_T4(T):
    """
    Make 4x4 matrix "standard":
      [ R t ]
      [ 0 1 ]
    Some pipelines store translation in last ROW (tx,ty,tz,1) and last column zeros.
    If so, transpose it.
    """
    T = np.array(T, dtype=np.float32)
    if T.shape != (4, 4):
        raise ValueError(f"Expect 4x4, got {T.shape}")

    # Heuristic: translation seems in last row, last column near 0 in first 3 rows
    if abs(T[3, 3] - 1.0) < 1e-4 and np.linalg.norm(T[:3, 3]) < 1e-4 and np.linalg.norm(T[3, :3]) > 1e-4:
        T = T.T
    return T.astype(np.float32)


def decompose_A_to_sR(A):
    """A ~ s * R (uniform scale), project to nearest rotation."""
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


def load_camera_w2c(camera_json_path, frame_idx=0):
    cam = json.load(open(camera_json_path, "r"))

    # Prefer world_view_transform (common in your file)
    if "world_view_transform" in cam:
        wvt = cam["world_view_transform"]
        if isinstance(wvt, list):
            M = wvt[min(frame_idx, len(wvt) - 1)]
        else:
            M = wvt
        return normalize_T4(M)

    # Fallback: if only c2w is provided, invert it
    if "c2w" in cam:
        c2w = cam["c2w"]
        if isinstance(c2w, list):
            C = c2w[min(frame_idx, len(c2w) - 1)]
        else:
            C = c2w
        C = normalize_T4(C)
        return np.linalg.inv(C).astype(np.float32)

    # Optional common names
    for k in ["w2c", "extrinsic", "world_to_camera"]:
        if k in cam:
            M = cam[k]
            if isinstance(M, list):
                M = M[min(frame_idx, len(M) - 1)]
            return normalize_T4(M)

    raise KeyError(f"Cannot find w2c/world_view_transform/c2w in {camera_json_path}. keys={list(cam.keys())}")


def apply_root_RT(global_orient_aa, trans, R, t):
    """
    Apply root transform:
      R_new = R * R_old
      trans_new = R @ trans + t
    axis-angle in/out.
    """
    T = global_orient_aa.shape[0]
    Rg = Rot.from_rotvec(global_orient_aa)                 # (T,)
    Rs = Rot.from_matrix(np.repeat(R[None, :, :], T, 0))   # (T,)
    Rnew = Rs * Rg
    global_new = Rnew.as_rotvec().astype(np.float32)

    # row-vector form: x' = x R^T + t
    trans_new = (trans @ R.T) + t[None, :]
    return global_new, trans_new.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_npz", required=True)
    ap.add_argument("--scene_transforms_py", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--camera_json", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=99)  # inclusive
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- load motion ----
    d = np.load(args.src_npz, allow_pickle=True)
    poses = d["poses"].astype(np.float32)  # (T,156)
    trans = d["trans"].astype(np.float32)  # (T,3)
    betas = d["betas"].astype(np.float32) if "betas" in d.files else None

    # slice frames [start, end] inclusive
    s, e = args.start, args.end
    poses = poses[s:e+1]
    trans = trans[s:e+1]
    T = poses.shape[0]
    assert T == (e - s + 1), f"Frame slicing mismatch: got T={T}, expect {e-s+1}"
    assert T == 100, f"You requested 0-99 (100 frames). Now got T={T}."

    global_orient = poses[:, 0:3]
    rest_pose = poses[:, 3:]  # keep as-is (body/hands/etc)

    # ---- scene transform ----
    st = SourceFileLoader("st_dyn", args.scene_transforms_py).load_module()
    T_scene = np.array(st.SCENE_TRANSFORMS[args.scene], dtype=np.float32)
    A = T_scene[:3, :3]
    t_scene = T_scene[:3, 3]

    scale_s, R_scene = decompose_A_to_sR(A)

    # world_Rt: use R_scene, t_scene
    glob_w_Rt, trans_w_Rt = apply_root_RT(global_orient, trans, R_scene, t_scene)

    # world_As: use A on translation (A@trans + t), but rotation still uses R_scene
    glob_w_As = glob_w_Rt.copy()
    trans_w_As = (trans @ A.T) + t_scene[None, :]   # row-vector form of A@x + t
    trans_w_As = trans_w_As.astype(np.float32)

    # ---- camera w2c (per-frame if camera has list) ----
    # camera json has list length 32 in your file; we just take frame0 for all frames by default
    # If you later have per-frame w2c length>=100, change frame_idx=i in loop.
    W2C0 = load_camera_w2c(args.camera_json, frame_idx=0)
    R_w2c = W2C0[:3, :3]
    t_w2c = W2C0[:3, 3]

    # cam_Rt from world_Rt
    glob_c_Rt, trans_c_Rt = apply_root_RT(glob_w_Rt, trans_w_Rt, R_w2c, t_w2c)
    # cam_As from world_As
    glob_c_As, trans_c_As = apply_root_RT(glob_w_As, trans_w_As, R_w2c, t_w2c)

    # ---- pack & save ----
    def save_npz(path, glob_aa, trans_xyz, tag):
        out_poses = np.concatenate([glob_aa, rest_pose], axis=1).astype(np.float32)
        if betas is None:
            np.savez(path,
                     poses=out_poses, trans=trans_xyz,
                     scene=args.scene, camera_json=args.camera_json,
                     frame_start=s, frame_end=e,
                     scene_scale=np.array([scale_s], np.float32),
                     tag=tag)
        else:
            np.savez(path,
                     poses=out_poses, trans=trans_xyz, betas=betas,
                     scene=args.scene, camera_json=args.camera_json,
                     frame_start=s, frame_end=e,
                     scene_scale=np.array([scale_s], np.float32),
                     tag=tag)

    p_world_Rt = os.path.join(args.out_dir, f"smpl_{args.scene}_world_Rt_0_99.npz")
    p_world_As = os.path.join(args.out_dir, f"smpl_{args.scene}_world_As_0_99.npz")
    p_cam_Rt   = os.path.join(args.out_dir, f"smpl_{args.scene}_cam_Rt_0_99.npz")
    p_cam_As   = os.path.join(args.out_dir, f"smpl_{args.scene}_cam_As_0_99.npz")

    save_npz(p_world_Rt, glob_w_Rt, trans_w_Rt, "world_Rt")
    save_npz(p_world_As, glob_w_As, trans_w_As, "world_As")
    save_npz(p_cam_Rt,   glob_c_Rt, trans_c_Rt, "cam_Rt")
    save_npz(p_cam_As,   glob_c_As, trans_c_As, "cam_As")

    print("[OK] saved 4 files:")
    print(" ", p_world_Rt)
    print(" ", p_world_As)
    print(" ", p_cam_Rt)
    print(" ", p_cam_As)
    print(f"[INFO] frames: {s}..{e} (T={T})")
    print(f"[INFO] scene_scale(s)={scale_s:.6f} (stored as metadata; As uses A on trans, Rt uses R only)")
    print(f"[INFO] camera uses world_view_transform frame0 by default (your json list len is limited).")

if __name__ == "__main__":
    main()
