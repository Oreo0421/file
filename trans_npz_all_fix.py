#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
trans_npz_all_fix.py

Align root motion of SMPL / SMPL-X motion data using a reference 00.npz,
with optional in-place (freeze) control.

Features:
- align_first_frame (recommended)
- optional inplace control: none | xy | xyz
- safe handling of .npz (no in-place modification)
"""

import argparse
import shutil
from pathlib import Path
import numpy as np


# ============================================================
# Math utils: axis-angle <-> rotation matrix
# ============================================================
def _skew(v):
    x, y, z = v
    return np.array([[0, -z, y],
                     [z, 0, -x],
                     [-y, x, 0]], dtype=np.float64)


def axis_angle_to_R(aa):
    aa = np.asarray(aa, dtype=np.float64).reshape(3)
    theta = np.linalg.norm(aa)
    if theta < 1e-12:
        return np.eye(3)
    k = aa / theta
    K = _skew(k)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def R_to_axis_angle(R):
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    trace = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(trace)
    if theta < 1e-12:
        return np.zeros(3)

    w = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1]
    ])
    axis = w / (2.0 * np.sin(theta) + 1e-12)
    return axis * theta


# ============================================================
# IO helpers
# ============================================================
def _get_first_existing(npz, keys):
    for k in keys:
        if k in npz.files:
            return k, npz[k]
    return None, None


def load_src_root(src_npz):
    src = np.load(src_npz, allow_pickle=True)

    k_go, go = _get_first_existing(src, ["global_orient", "root_orient"])
    if go is None:
        go = src["poses"][:, :3]

    k_tr, tr = _get_first_existing(src, ["transl", "trans", "translation"])
    if tr is None:
        raise KeyError("Source npz missing translation")

    if go.ndim == 1:
        go = go[None]
    if tr.ndim == 1:
        tr = tr[None]

    return go.astype(np.float64), tr.astype(np.float64)


def load_tgt_npz(path):
    npz = np.load(path, allow_pickle=True)
    data = {k: npz[k] for k in npz.files}

    poses = data["poses"]
    k_tr, trans = _get_first_existing(npz, ["trans", "transl", "translation"])
    if trans is None:
        raise KeyError("Target npz missing translation")

    return data, poses, trans, k_tr


# ============================================================
# Core logic
# ============================================================
def apply_align_first_frame(src_go, src_tr, poses, trans):
    R_src0 = axis_angle_to_R(src_go[0])
    R_tgt0 = axis_angle_to_R(poses[0, :3])

    R_align = R_src0 @ np.linalg.inv(R_tgt0)
    t_align = src_tr[0] - R_align @ trans[0]

    out_poses = poses.copy().astype(np.float64)
    out_trans = trans.copy().astype(np.float64)

    for i in range(len(out_poses)):
        R_i = axis_angle_to_R(out_poses[i, :3])
        out_poses[i, :3] = R_to_axis_angle(R_align @ R_i)
        out_trans[i] = R_align @ out_trans[i] + t_align

    return out_poses.astype(np.float32), out_trans.astype(np.float32)


def apply_inplace(trans, mode):
    if mode == "none":
        return trans

    trans_new = trans.copy()

    if mode == "xy":
        trans_new[:, 0] = trans_new[0, 0]
        trans_new[:, 1] = trans_new[0, 1]
    elif mode == "xyz":
        trans_new[:] = trans_new[0]

    return trans_new


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="reference 00.npz")
    ap.add_argument("--tgt_npz", required=True, help="target motion npz")
    ap.add_argument("--out_npz", required=True, help="output npz")
    ap.add_argument("--mode", default="align_first_frame",
                    choices=["align_first_frame"])
    ap.add_argument("--inplace_mode", default="none",
                    choices=["none", "xy", "xyz"],
                    help="lock translation after alignment")
    args = ap.parse_args()

    # load
    src_go, src_tr = load_src_root(args.src)
    data, poses, trans, k_tr = load_tgt_npz(args.tgt_npz)

    # align
    out_poses, out_trans = apply_align_first_frame(
        src_go, src_tr, poses, trans
    )

    # optional inplace
    out_trans = apply_inplace(out_trans, args.inplace_mode)

    # write back
    data["poses"] = out_poses
    data[k_tr] = out_trans

    np.savez(args.out_npz, **data)
    print("Saved:", args.out_npz)


if __name__ == "__main__":
    main()
