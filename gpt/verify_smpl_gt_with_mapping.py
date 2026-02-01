#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, re, glob, argparse
import numpy as np
import torch
import smplx
from scipy.optimize import linear_sum_assignment
from importlib.machinery import SourceFileLoader


# ------------------ IO: per-frame joints dir ------------------
def load_joints_dir(joints_dir: str, pattern="*.npy") -> np.ndarray:
    files = glob.glob(os.path.join(joints_dir, pattern))
    if not files:
        raise FileNotFoundError(f"No {pattern} in {joints_dir}")

    def frame_key(p):
        name = os.path.basename(p)
        nums = re.findall(r"\d+", name)
        return int(nums[-1]) if nums else name

    files = sorted(files, key=frame_key)
    arrs = []
    for f in files:
        a = np.load(f).astype(np.float32)
        if a.ndim == 3 and a.shape[0] == 1:
            a = a[0]
        if a.ndim != 2 or a.shape[1] != 3:
            raise ValueError(f"Bad shape {a.shape} in {f}, expect (J,3)")
        arrs.append(a)
    return np.stack(arrs, axis=0)  # (T,J,3)


def crop_to_common_T(*arrays):
    Ts = [a.shape[0] for a in arrays if a is not None]
    Tmin = min(Ts)
    out = []
    for a in arrays:
        if a is None:
            out.append(None)
            continue
        if a.shape[0] != Tmin:
            print(f"[WARN] T mismatch {a.shape[0]} -> crop to {Tmin}")
        out.append(a[:Tmin])
    return out


# ------------------ SMPL npz -> SMPLX forward joints ------------------
def find_key(npz, candidates):
    for k in candidates:
        if k in npz.files:
            return k
    return None


def make_betas_batch(betas_np, T, num_betas):
    if betas_np is None:
        betas_np = np.zeros((num_betas,), dtype=np.float32)
    betas_np = betas_np.astype(np.float32)
    if betas_np.ndim == 1:
        b = betas_np[:num_betas]
        return np.repeat(b[None, :], T, axis=0)
    b = betas_np[:T, :num_betas]
    if b.shape[0] != T:
        b = np.repeat(b[:1], T, axis=0)
    return b


def split_pose_to_smplx_kwargs(poses_t: torch.Tensor):
    """
    Your poses are (T,156) (SMPL-H style).
    For SMPL-X forward, pad jaw/eyes = 0 => (T,165).
    """
    T, D = poses_t.shape
    if D == 156:
        pad = torch.zeros((T, 9), device=poses_t.device, dtype=poses_t.dtype)
        poses_t = torch.cat([poses_t, pad], dim=1)
        D = 165
    if D != 165:
        raise ValueError(f"SMPLX expects 165 (or 156 to pad), got {D}")

    return dict(
        global_orient=poses_t[:, 0:3],
        body_pose=poses_t[:, 3:66],
        left_hand_pose=poses_t[:, 66:111],
        right_hand_pose=poses_t[:, 111:156],
        jaw_pose=poses_t[:, 156:159],
        leye_pose=poses_t[:, 159:162],
        reye_pose=poses_t[:, 162:165],
    )


def forward_smplx_joints_from_npz(npz_path, model_path, gender="neutral", num_betas=10, device="cuda"):
    d = np.load(npz_path, allow_pickle=True)
    poses_key = find_key(d, ["poses", "fullpose"])
    trans_key = find_key(d, ["trans", "transl"])
    if poses_key is None:
        raise KeyError(f"{npz_path} missing poses/fullpose. keys={d.files}")

    poses = d[poses_key].astype(np.float32)
    T = poses.shape[0]
    trans = d[trans_key].astype(np.float32) if trans_key is not None else None
    betas = d["betas"].astype(np.float32) if "betas" in d.files else None

    poses_t = torch.from_numpy(poses).to(device=device, dtype=torch.float32)
    betas_b = make_betas_batch(betas, T=T, num_betas=num_betas)
    betas_t = torch.from_numpy(betas_b).to(device=device, dtype=torch.float32)

    kwargs = split_pose_to_smplx_kwargs(poses_t)
    kwargs["betas"] = betas_t
    if trans is not None:
        kwargs["transl"] = torch.from_numpy(trans).to(device=device, dtype=torch.float32)

    model = smplx.create(
        model_path=model_path,
        model_type="smplx",
        gender=gender,
        num_betas=num_betas,
        use_pca=False,
        batch_size=T,
    ).to(device)
    model.eval()

    with torch.no_grad():
        out = model(**kwargs, return_verts=False)
        joints = out.joints.detach().cpu().numpy().astype(np.float32)  # (T,Jm,3)

    meta = {"T": T, "Jm": joints.shape[1], "poses_dim": poses.shape[1], "poses_key": poses_key, "trans_key": trans_key}
    return joints, meta


# ------------------ Mapping 127 -> 22 (Hungarian) ------------------
def compute_cost_matrix(pred, gt, frame_sample=20):
    """
    pred: (T,Jm,3), gt: (T,22,3)
    cost C[j,k] = mean over sampled frames of ||pred[:,j]-gt[:,k]||
    """
    T, Jm, _ = pred.shape
    _, Jg, _ = gt.shape
    assert Jg == 22, f"GT joints must be 22, got {Jg}"

    if frame_sample > 0 and frame_sample < T:
        idx = np.linspace(0, T - 1, frame_sample).astype(np.int64)
        pred_s = pred[idx]
        gt_s = gt[idx]
    else:
        pred_s = pred
        gt_s = gt

    C = np.zeros((Jm, Jg), dtype=np.float32)
    for j in range(Jm):
        diff = pred_s[:, j:j+1, :] - gt_s  # (Ts,22,3)
        C[j] = np.linalg.norm(diff, axis=-1).mean(axis=0)
    return C


def hungarian_mapping(pred, gt, frame_sample=20):
    C = compute_cost_matrix(pred, gt, frame_sample=frame_sample)  # (Jm,22)
    row_ind, col_ind = linear_sum_assignment(C)
    mapping = np.zeros((gt.shape[1],), dtype=np.int64)  # (22,)
    for r, c in zip(row_ind, col_ind):
        mapping[c] = r
    return mapping, C


# ------------------ Scene transform apply ------------------
def apply_T_to_joints(joints, T4):
    # joints: (T,J,3)
    Tn, J, _ = joints.shape
    ones = np.ones((Tn, J, 1), dtype=np.float32)
    jh = np.concatenate([joints, ones], axis=-1)  # (T,J,4)
    out = (T4[None, None, :, :] @ jh[..., None]).squeeze(-1)
    return out[..., :3].astype(np.float32)


# ------------------ Metrics ------------------
def stats_l2(pred, gt):
    l2 = np.linalg.norm(pred - gt, axis=-1)  # (T,J)
    per_frame = l2.mean(axis=1)
    per_joint = l2.mean(axis=0)
    return {
        "mean": float(l2.mean()),
        "median": float(np.median(l2)),
        "p90": float(np.percentile(l2, 90)),
        "p99": float(np.percentile(l2, 99)),
        "max": float(l2.max()),
        "worst_frame": int(np.argmax(per_frame)),
        "worst_joint": int(np.argmax(per_joint)),
        "worst_frame_mean": float(per_frame.max()),
        "worst_joint_mean": float(per_joint.max()),
    }


def bone_lengths(joints, kin_parent):
    # joints: (T,22,3), kin_parent: (22,)
    lens = []
    for j in range(22):
        p = int(kin_parent[j])
        if p < 0:
            continue
        d = np.linalg.norm(joints[:, j] - joints[:, p], axis=-1)  # (T,)
        lens.append(d.mean())
    return np.array(lens, dtype=np.float32)


# ------------------ Main ------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smpl_npz_original", required=True,
                    help="Original motion NPZ BEFORE scene placement. (e.g. /mnt/data/brooming.npz)")
    ap.add_argument("--joints_dir_original", required=True,
                    help="Dir of per-frame GT original joints (22,3)")
    ap.add_argument("--joints_dir_transformed", default=None,
                    help="Dir of per-frame GT transformed joints (22,3)")
    ap.add_argument("--scene_transforms_py", default=None,
                    help="scene_transforms.py (needed if you want transformed check)")
    ap.add_argument("--scene", default=None,
                    help="scene key in SCENE_TRANSFORMS (needed if transformed check)")

    ap.add_argument("--model_path", required=True,
                    help="Directory that contains SMPLX_NEUTRAL.pkl (your smpl_files/smplx)")
    ap.add_argument("--gender", default="neutral", choices=["male", "female", "neutral"])
    ap.add_argument("--num_betas", type=int, default=10)
    ap.add_argument("--device", default="cuda")

    ap.add_argument("--frame_sample", type=int, default=20,
                    help="Use N frames to compute mapping cost (speed). 0 = all frames.")
    ap.add_argument("--joint_index", default=None,
                    help="If given, skip auto mapping. Comma-separated 22 indices into pred joints.")
    ap.add_argument("--save_joint_index", default=None,
                    help="Save inferred joint_index to .npy")
    ap.add_argument("--kin_parent", default=None,
                    help="Optional kin_parent.npy to print bone-length sanity check (len=22)")

    args = ap.parse_args()

    gt_ori = load_joints_dir(args.joints_dir_original)  # (Tg,22,3)
    print(f"[GT original] {gt_ori.shape}  dir={args.joints_dir_original}")

    pred_full, meta = forward_smplx_joints_from_npz(
        args.smpl_npz_original, args.model_path, args.gender, args.num_betas, args.device
    )
    print(f"[Pred SMPLX] {pred_full.shape}  meta={meta}")

    # crop T
    pred_full, gt_ori = crop_to_common_T(pred_full, gt_ori)

    # mapping
    if args.joint_index is not None:
        idx = np.array([int(x) for x in args.joint_index.split(",")], dtype=np.int64)
        if idx.shape[0] != 22:
            raise ValueError("joint_index must have 22 integers.")
        print("[MAP] using provided joint_index")
    else:
        idx, C = hungarian_mapping(pred_full, gt_ori, frame_sample=args.frame_sample)
        print("[MAP] inferred joint_index (len=22):")
        print("      " + ",".join(str(int(x)) for x in idx.tolist()))

        if args.save_joint_index:
            np.save(args.save_joint_index, idx)
            print(f"[MAP] saved joint_index to {args.save_joint_index}")

    pred22_ori = pred_full[:, idx, :]  # (T,22,3)

    # stats: original
    st_ori = stats_l2(pred22_ori, gt_ori)
    print("\n================== CHECK: ORIGINAL ==================")
    print(f"mean={st_ori['mean']:.6e}  p99={st_ori['p99']:.6e}  max={st_ori['max']:.6e}  "
          f"worst_frame={st_ori['worst_frame']}  worst_joint={st_ori['worst_joint']}")

    # optional: bone-length sanity
    if args.kin_parent:
        kin_parent = np.load(args.kin_parent).astype(np.int32)
        if kin_parent.shape[0] != 22:
            raise ValueError(f"kin_parent must be len=22, got {kin_parent.shape}")
        bl_pred = bone_lengths(pred22_ori, kin_parent)
        bl_gt = bone_lengths(gt_ori, kin_parent)
        rel = np.mean(np.abs(bl_pred - bl_gt) / (bl_gt + 1e-8))
        print("\n[BONE] mean relative bone-length diff (original):", float(rel))

    # transformed check (optional)
    if args.joints_dir_transformed:
        if not args.scene_transforms_py or not args.scene:
            raise ValueError("Need --scene_transforms_py and --scene if --joints_dir_transformed is set.")

        gt_tr = load_joints_dir(args.joints_dir_transformed)
        print(f"\n[GT transformed] {gt_tr.shape}  dir={args.joints_dir_transformed}")

        # crop
        pred22_ori_c, gt_tr = crop_to_common_T(pred22_ori, gt_tr)

        # load T_scene
        mod = SourceFileLoader("st_dyn", args.scene_transforms_py).load_module()
        T_scene = np.array(mod.SCENE_TRANSFORMS[args.scene], dtype=np.float32)
        print("[Scene T] loaded:", args.scene)

        pred22_tr = apply_T_to_joints(pred22_ori_c, T_scene)
        st_tr = stats_l2(pred22_tr, gt_tr)

        print("\n================== CHECK: TRANSFORMED ==================")
        print(f"mean={st_tr['mean']:.6e}  p99={st_tr['p99']:.6e}  max={st_tr['max']:.6e}  "
              f"worst_frame={st_tr['worst_frame']}  worst_joint={st_tr['worst_joint']}")

        if args.kin_parent:
            kin_parent = np.load(args.kin_parent).astype(np.int32)
            bl_pred = bone_lengths(pred22_tr, kin_parent)
            bl_gt = bone_lengths(gt_tr, kin_parent)
            rel = np.mean(np.abs(bl_pred - bl_gt) / (bl_gt + 1e-8))
            print("\n[BONE] mean relative bone-length diff (transformed):", float(rel))


if __name__ == "__main__":
    main()
