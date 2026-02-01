#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, re, glob, argparse
import numpy as np
import torch
import smplx
from scipy.spatial.transform import Rotation as Rot

def load_joints_dir(joints_dir: str, pattern="*.npy"):
    files = glob.glob(os.path.join(joints_dir, pattern))
    if not files:
        raise FileNotFoundError(f"No {pattern} in {joints_dir}")
    def key(p):
        name=os.path.basename(p)
        nums=re.findall(r"\d+", name)
        return int(nums[-1]) if nums else name
    files=sorted(files, key=key)
    arr=[]
    for f in files:
        a=np.load(f).astype(np.float32)
        if a.ndim==3 and a.shape[0]==1: a=a[0]
        if a.shape!=(22,3):
            raise ValueError(f"{f} shape {a.shape} != (22,3)")
        arr.append(a)
    return np.stack(arr,0)  # (T,22,3)

def crop_T(a,b):
    Tmin=min(a.shape[0], b.shape[0])
    if a.shape[0]!=b.shape[0]:
        print(f"[WARN] T mismatch {a.shape[0]} vs {b.shape[0]} -> crop {Tmin}")
    return a[:Tmin], b[:Tmin]

def split_pose_smplx(poses_t):
    T,D=poses_t.shape
    if D==156:
        pad=torch.zeros((T,9), device=poses_t.device, dtype=poses_t.dtype)
        poses_t=torch.cat([poses_t,pad],1)
        D=165
    if D!=165:
        raise ValueError(f"need 165 or 156->pad, got {D}")
    return dict(
        global_orient=poses_t[:,0:3],
        body_pose=poses_t[:,3:66],
        left_hand_pose=poses_t[:,66:111],
        right_hand_pose=poses_t[:,111:156],
        jaw_pose=poses_t[:,156:159],
        leye_pose=poses_t[:,159:162],
        reye_pose=poses_t[:,162:165],
    )

def make_betas_batch(betas, T, num_betas):
    if betas is None:
        betas=np.zeros((num_betas,), np.float32)
    betas=betas.astype(np.float32)
    if betas.ndim==1:
        b=betas[:num_betas]
        return np.repeat(b[None,:], T, 0)
    b=betas[:T,:num_betas]
    if b.shape[0]!=T:
        b=np.repeat(b[:1], T, 0)
    return b

def forward_smplx_joints(npz_path, model_path, gender="neutral", num_betas=10, device="cuda"):
    d=np.load(npz_path, allow_pickle=True)
    poses=d["poses"].astype(np.float32)
    trans=d["trans"].astype(np.float32)
    betas=d["betas"].astype(np.float32) if "betas" in d.files else None

    T=poses.shape[0]
    poses_t=torch.from_numpy(poses).to(device)
    trans_t=torch.from_numpy(trans).to(device)
    betas_b=make_betas_batch(betas, T, num_betas)
    betas_t=torch.from_numpy(betas_b).to(device)

    kwargs=split_pose_smplx(poses_t)
    kwargs["transl"]=trans_t
    kwargs["betas"]=betas_t

    model=smplx.create(
        model_path=model_path, model_type="smplx", gender=gender,
        num_betas=num_betas, use_pca=False, batch_size=T
    ).to(device)
    model.eval()
    with torch.no_grad():
        out=model(**kwargs, return_verts=False)
        joints=out.joints.detach().cpu().numpy().astype(np.float32) # (T,Jm,3)
    return joints

def stats(pred, gt):
    l2=np.linalg.norm(pred-gt, axis=-1) # (T,22)
    return dict(
        mean=float(l2.mean()),
        p99=float(np.percentile(l2,99)),
        max=float(l2.max())
    )

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cam_npz", nargs="+", required=True, help="cam_Rt and cam_As npz paths")
    ap.add_argument("--gt_joints_dir_cam", required=True,
                    help="GT joints in camera space (22,3 per frame). If you only have world joints, don't use this.")
    ap.add_argument("--joint_index_22", required=True, help="saved mapping .npy (len=22)")
    ap.add_argument("--model_path", required=True, help="SMPLX model dir (contains SMPLX_NEUTRAL.pkl)")
    ap.add_argument("--gender", default="neutral", choices=["male","female","neutral"])
    ap.add_argument("--num_betas", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    args=ap.parse_args()

    gt=load_joints_dir(args.gt_joints_dir_cam)
    idx=np.load(args.joint_index_22).astype(np.int64)
    if idx.shape[0]!=22:
        raise ValueError("joint_index_22 must be len=22")

    best=None
    for npz in args.cam_npz:
        pred_full=forward_smplx_joints(npz, args.model_path, args.gender, args.num_betas, args.device)
        pred22=pred_full[:, idx, :]
        pred22, gt_c=crop_T(pred22, gt)
        st=stats(pred22, gt_c)
        print(f"\n[CAND] {npz}")
        print(f"  mean={st['mean']:.6e}  p99={st['p99']:.6e}  max={st['max']:.6e}")
        if best is None or st["mean"]<best[0]["mean"]:
            best=(st, npz)
    print("\n[RECOMMEND] best cam GT:")
    print(" ", best[1])

if __name__=="__main__":
    main()
