#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
transfer_root_00_to_newformat.py

Goal:
  Take root motion from 00.npz (global_orient + transl/trans)
  and apply it to a "new format" motion:
    - folder containing poses.npy, trans.npy, etc.  (like your screenshot)
    - OR a .npz containing keys 'poses' and 'trans'

New format convention:
  poses: (N, D) where first 3 dims are global_orient axis-angle
  trans: (N, 3) translation

Modes:
  - align_first_frame (recommended): align tgt frame0 to src frame0, then apply to all frames
  - copy: directly replace tgt root with src root (length adjusted)

Usage:
  # target is a folder (contains poses.npy + trans.npy)
  python transfer_root_00_to_newformat.py --src 00.npz --tgt_dir /path/to/new_motion_dir \
      --out_dir /path/to/new_motion_dir_rooted --mode align_first_frame

  # target is a .npz (contains poses/trans)
  python transfer_root_00_to_newformat.py --src 00.npz --tgt_npz 01_newformat.npz \
      --out_npz 01_newformat_rooted.npz --mode align_first_frame
"""

import argparse
import os
import shutil
from pathlib import Path
import numpy as np


# -----------------------------
# Math utils: axis-angle <-> R
# -----------------------------
def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0, -z, y],
                     [z, 0, -x],
                     [-y, x, 0]], dtype=np.float64)


def axis_angle_to_R(aa: np.ndarray) -> np.ndarray:
    aa = np.asarray(aa, dtype=np.float64).reshape(3)
    theta = np.linalg.norm(aa)
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    k = aa / theta
    K = _skew(k)
    R = np.eye(3, dtype=np.float64) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return R


def R_to_axis_angle(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    trace = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(trace)
    if theta < 1e-12:
        return np.zeros(3, dtype=np.float64)

    w = np.array([R[2, 1] - R[1, 2],
                  R[0, 2] - R[2, 0],
                  R[1, 0] - R[0, 1]], dtype=np.float64)
    axis = w / (2.0 * np.sin(theta) + 1e-12)
    return axis * theta


# -----------------------------
# Generic loaders (src 00.npz)
# -----------------------------
def _get_first_existing(npz, keys):
    for k in keys:
        if k in npz.files:
            return k, npz[k]
    return None, None


def load_src_root(src_npz_path: str):
    """
    Load source root from 00.npz. Supports:
      - global_orient + transl
      - poses (first 3 dims) + trans
    Returns:
      src_go: (Ns,3) float64
      src_tr: (Ns,3) float64
    """
    src = np.load(src_npz_path, allow_pickle=True)

    # orientation: prefer explicit global_orient, else from poses[:, :3]
    k_go, go = _get_first_existing(src, ["global_orient", "root_orient"])
    if go is None:
        k_pose, poses = _get_first_existing(src, ["poses"])
        if poses is None:
            raise KeyError("Source 00.npz missing global_orient/root_orient and poses.")
        poses = np.asarray(poses)
        if poses.ndim != 2 or poses.shape[1] < 3:
            raise ValueError(f"Source poses has bad shape {poses.shape}, expected (N,D>=3).")
        go = poses[:, :3]
        k_go = "poses[:,:3]"

    # translation: prefer transl, else trans
    k_tr, tr = _get_first_existing(src, ["transl", "trans", "translation"])
    if tr is None:
        raise KeyError("Source 00.npz missing transl/trans/translation.")

    go = np.asarray(go)
    tr = np.asarray(tr)

    # Normalize shapes to (Ns,3)
    if go.shape == (3,):
        go = go[None, :]
    if tr.shape == (3,):
        tr = tr[None, :]

    if go.ndim != 2 or go.shape[1] != 3:
        raise ValueError(f"Source global_orient has bad shape {go.shape}, expected (N,3) or (3,).")
    if tr.ndim != 2 or tr.shape[1] != 3:
        raise ValueError(f"Source transl/trans has bad shape {tr.shape}, expected (N,3) or (3,).")

    return go.astype(np.float64), tr.astype(np.float64), k_go, k_tr


def adjust_len(arr: np.ndarray, N: int, mode: str):
    """
    arr: (K,3)
    mode:
      - repeat_first: use arr[0] repeat N
      - tile_or_crop: if K>=N crop, else tile then crop
    """
    K = arr.shape[0]
    if K == N:
        return arr
    if mode == "repeat_first":
        return np.tile(arr[0:1], (N, 1))
    if mode == "tile_or_crop":
        if K > N:
            return arr[:N]
        reps = int(np.ceil(N / K))
        tiled = np.tile(arr, (reps, 1))
        return tiled[:N]
    raise ValueError(f"Unknown length mode: {mode}")


# -----------------------------
# Target loaders: folder or npz
# -----------------------------
def load_tgt_from_dir(tgt_dir: str):
    """
    Expect files:
      poses.npy (N,D>=3), trans.npy (N,3)
    Return dict of arrays for saving, plus N, D.
    """
    tgt_dir = Path(tgt_dir)
    poses_path = tgt_dir / "poses.npy"
    trans_path = tgt_dir / "trans.npy"
    if not poses_path.exists():
        raise FileNotFoundError(f"Target dir missing poses.npy: {poses_path}")
    if not trans_path.exists():
        raise FileNotFoundError(f"Target dir missing trans.npy: {trans_path}")

    poses = np.load(str(poses_path))
    trans = np.load(str(trans_path))

    if poses.ndim != 2 or poses.shape[1] < 3:
        raise ValueError(f"Target poses.npy has bad shape {poses.shape}, expected (N,D>=3).")
    if trans.ndim != 2 or trans.shape[1] != 3:
        raise ValueError(f"Target trans.npy has bad shape {trans.shape}, expected (N,3).")
    if poses.shape[0] != trans.shape[0]:
        raise ValueError(f"Target poses/trans length mismatch: {poses.shape[0]} vs {trans.shape[0]}")

    # load all npy files in dir so we can copy them to out_dir
    arrays = {}
    for p in tgt_dir.glob("*.npy"):
        arrays[p.name] = np.load(str(p))

    N, D = poses.shape
    return arrays, N, D


def save_to_dir(arrays: dict, out_dir: str):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in arrays.items():
        np.save(str(out_dir / name), arr)


def load_tgt_from_npz(tgt_npz: str):
    """
    Expect keys:
      poses: (N,D>=3)
      trans: (N,3)  (or transl)
    """
    tgt = np.load(tgt_npz, allow_pickle=True)
    if "poses" not in tgt.files:
        raise KeyError("Target npz missing key 'poses' (new format expects poses).")

    poses = np.asarray(tgt["poses"])
    if poses.ndim != 2 or poses.shape[1] < 3:
        raise ValueError(f"Target poses has bad shape {poses.shape}, expected (N,D>=3).")

    # translation can be 'trans' or 'transl'
    k_tr, trans = _get_first_existing(tgt, ["trans", "transl", "translation"])
    if trans is None:
        raise KeyError("Target npz missing translation key (trans/transl/translation).")

    trans = np.asarray(trans)
    if trans.ndim != 2 or trans.shape[1] != 3:
        raise ValueError(f"Target trans has bad shape {trans.shape}, expected (N,3).")

    if poses.shape[0] != trans.shape[0]:
        raise ValueError(f"Target poses/trans length mismatch: {poses.shape[0]} vs {trans.shape[0]}")

    data = {k: tgt[k] for k in tgt.files}
    return data, poses.shape[0], poses.shape[1], k_tr


# -----------------------------
# Apply modes
# -----------------------------
def apply_copy(src_go, src_tr, tgt_poses, tgt_trans, src_len_mode="tile_or_crop"):
    N = tgt_poses.shape[0]
    go = adjust_len(src_go, N, src_len_mode)
    tr = adjust_len(src_tr, N, src_len_mode)
    out_poses = np.array(tgt_poses, copy=True)
    out_trans = np.array(tgt_trans, copy=True)
    out_poses[:, :3] = go
    out_trans[:, :] = tr
    return out_poses, out_trans


def apply_align_first_frame(src_go, src_tr, tgt_poses, tgt_trans):
    """
    Align target root to source using frame0:
      R_align = R_src0 @ inv(R_tgt0)
      t_align = t_src0 - R_align @ t_tgt0
    Apply to all frames.
    """
    R_src0 = axis_angle_to_R(src_go[0])
    R_tgt0 = axis_angle_to_R(tgt_poses[0, :3])
    R_align = R_src0 @ np.linalg.inv(R_tgt0)
    t_align = src_tr[0] - (R_align @ tgt_trans[0])

    out_poses = np.array(tgt_poses, copy=True).astype(np.float64)
    out_trans = np.array(tgt_trans, copy=True).astype(np.float64)

    for i in range(tgt_poses.shape[0]):
        R_i = axis_angle_to_R(tgt_poses[i, :3])
        R_new = R_align @ R_i
        out_poses[i, :3] = R_to_axis_angle(R_new)
        out_trans[i, :] = (R_align @ tgt_trans[i]) + t_align

    return out_poses.astype(np.float32), out_trans.astype(np.float32), R_align, t_align


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source 00.npz (old format)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--tgt_dir", help="target new-format folder containing poses.npy/trans.npy")
    g.add_argument("--tgt_npz", help="target new-format .npz containing poses + trans")
    ap.add_argument("--out_dir", default=None, help="output folder (for tgt_dir mode)")
    ap.add_argument("--out_npz", default=None, help="output npz path (for tgt_npz mode)")
    ap.add_argument("--mode", choices=["align_first_frame", "copy"], default="align_first_frame")
    ap.add_argument("--src_len_mode", choices=["repeat_first", "tile_or_crop"], default="tile_or_crop",
                    help="only for mode=copy when src length != tgt length")
    args = ap.parse_args()

    src_go, src_tr, src_go_key, src_tr_key = load_src_root(args.src)

    if args.tgt_dir:
        arrays, N, D = load_tgt_from_dir(args.tgt_dir)
        poses = arrays["poses.npy"]
        trans = arrays["trans.npy"]

        if args.mode == "copy":
            out_poses, out_trans = apply_copy(src_go, src_tr, poses, trans, src_len_mode=args.src_len_mode)
            debug = f"[copy] src({src_go_key},{src_tr_key}) -> tgt_dir N={N}"
        else:
            out_poses, out_trans, R_align, t_align = apply_align_first_frame(src_go, src_tr, poses, trans)
            debug = f"[align_first_frame] N={N}\nR_align=\n{R_align}\nt_align={t_align}"

        arrays["poses.npy"] = out_poses
        arrays["trans.npy"] = out_trans

        out_dir = args.out_dir
        if out_dir is None:
            out_dir = str(Path(args.tgt_dir).with_name(Path(args.tgt_dir).name + "_rooted"))
        # copy other non-npy files if exist
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        for p in Path(args.tgt_dir).iterdir():
            if p.is_file() and p.suffix != ".npy":
                shutil.copy2(str(p), str(Path(out_dir) / p.name))
        save_to_dir(arrays, out_dir)

        print("Saved folder:", out_dir)
        print("poses.npy:", arrays["poses.npy"].shape, "trans.npy:", arrays["trans.npy"].shape)
        print(debug)

    else:
        data, N, D, tgt_tr_key = load_tgt_from_npz(args.tgt_npz)
        poses = data["poses"]
        trans = data[tgt_tr_key]

        if args.mode == "copy":
            out_poses, out_trans = apply_copy(src_go, src_tr, poses, trans, src_len_mode=args.src_len_mode)
            debug = f"[copy] src({src_go_key},{src_tr_key}) -> tgt_npz N={N} (tgt_tr_key={tgt_tr_key})"
        else:
            out_poses, out_trans, R_align, t_align = apply_align_first_frame(src_go, src_tr, poses, trans)
            debug = f"[align_first_frame] N={N} (tgt_tr_key={tgt_tr_key})\nR_align=\n{R_align}\nt_align={t_align}"

        data["poses"] = out_poses
        data[tgt_tr_key] = out_trans

        out_npz = args.out_npz
        if out_npz is None:
            p = Path(args.tgt_npz)
            out_npz = str(p.with_name(p.stem + "_rooted.npz"))

        np.savez(out_npz, **data)
        print("Saved npz:", out_npz)
        print("poses:", data["poses"].shape, f"{tgt_tr_key}:", data[tgt_tr_key].shape)
        print(debug)


if __name__ == "__main__":
    main()
