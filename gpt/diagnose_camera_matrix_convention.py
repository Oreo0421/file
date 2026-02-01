#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, re, glob, json, argparse
import numpy as np
import torch
import smplx

def load_joints_dir(dir_path: str, pattern="*.npy"):
    files = glob.glob(os.path.join(dir_path, pattern))
    if not files:
        raise FileNotFoundError(f"No {pattern} in {dir_path}")
    def k(p):
        name=os.path.basename(p)
        nums=re.findall(r"\d+", name)
        return int(nums[-1]) if nums else name
    files=sorted(files, key=k)
    arr=[]
    for f in files:
        a=np.load(f).astype(np.float32)
        if a.ndim==3 and a.shape[0]==1: a=a[0]
        if a.shape!=(22,3):
            raise ValueError(f"{f} shape {a.shape} != (22,3)")
        arr.append(a)
    return np.stack(arr,0)  # (T,22,3)

def apply_T(joints, T4):
    Tn,J,_ = joints.shape
    ones=np.ones((Tn,J,1), np.float32)
    jh=np.concatenate([joints, ones], -1)   # (T,J,4)
    out=(T4[None,None,:,:] @ jh[...,None]).squeeze(-1)
    return out[...,:3].astype(np.float32)

def normalize_T4(T):
    T=np.array(T, np.float32)
    if T.shape!=(4,4):
        raise ValueError(f"Expect 4x4, got {T.shape}")
    # if translation is in last row, transpose
    if abs(T[3,3]-1)<1e-4 and np.linalg.norm(T[:3,3])<1e-4 and np.linalg.norm(T[3,:3])>1e-4:
        T=T.T
    return T.astype(np.float32)

def load_matrix_from_camera_json(camera_json):
    cam=json.load(open(camera_json,"r"))
    # pick best-guess raw matrix (do NOT decide w2c/c2w here)
    for k in ["world_view_transform","w2c","extrinsic","world_to_camera","c2w"]:
        if k in cam:
            M=cam[k]
            M=M[0] if isinstance(M,list) else M
            return normalize_T4(M), k, list(cam.keys())
    raise KeyError(f"no matrix key in {camera_json}, keys={list(cam.keys())}")

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

def forward_smplx_joints_from_cam_npz(npz_path, model_path, gender="neutral", num_betas=10, device="cuda"):
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
        joints=out.joints.detach().cpu().numpy().astype(np.float32)  # (T,Jm,3)
    return joints

def err_stats(pred, gt):
    l2=np.linalg.norm(pred-gt, axis=-1)  # (T,22)
    return float(l2.mean()), float(np.percentile(l2,99)), float(l2.max())

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cam_as_npz", required=True)
    ap.add_argument("--joints_dir_transformed", required=True)  # world/scene joints
    ap.add_argument("--camera_json", required=True)
    ap.add_argument("--joint_index_22", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--gender", default="neutral", choices=["male","female","neutral"])
    ap.add_argument("--num_betas", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    args=ap.parse_args()

    gt_world = load_joints_dir(args.joints_dir_transformed)  # (T,22,3)
    idx = np.load(args.joint_index_22).astype(np.int64)
    if idx.shape[0]!=22:
        raise ValueError("joint_index_22 must be len=22")

    pred_full = forward_smplx_joints_from_cam_npz(
        args.cam_as_npz, args.model_path, args.gender, args.num_betas, args.device
    )
    pred22 = pred_full[:, idx, :]

    Tmin=min(gt_world.shape[0], pred22.shape[0])
    gt_world=gt_world[:Tmin]
    pred22=pred22[:Tmin]

    Mraw, key_used, keys = load_matrix_from_camera_json(args.camera_json)
    print("[INFO] camera key used:", key_used)
    print("[INFO] camera keys:", keys)

    # candidates for w2c
    candidates = []
    candidates.append(("M_as_w2c", Mraw))
    candidates.append(("inv(M)_as_w2c", np.linalg.inv(Mraw).astype(np.float32)))
    candidates.append(("M.T_as_w2c", Mraw.T.astype(np.float32)))
    candidates.append(("inv(M.T)_as_w2c", np.linalg.inv(Mraw.T).astype(np.float32)))

    # axis flips (apply in camera space => left-multiply)
    F_id  = np.eye(4, dtype=np.float32)
    F_yz  = np.diag([1,-1,-1,1]).astype(np.float32)  # OpenGL<->OpenCV common
    F_x   = np.diag([-1,1,1,1]).astype(np.float32)
    flips = [("F=I",F_id), ("F=diag(1,-1,-1,1)",F_yz), ("F=diag(-1,1,1,1)",F_x)]

    best=None
    for cname, W2C in candidates:
        for fname, F in flips:
            W2C2 = (F @ W2C).astype(np.float32)
            gt_cam = apply_T(gt_world, W2C2)
            mean,p99,mx = err_stats(pred22, gt_cam)
            tag=f"{cname} + {fname}"
            if best is None or mean < best[0]:
                best=(mean,p99,mx,tag)
            print(f"{tag:32s}  mean={mean:.6e}  p99={p99:.6e}  max={mx:.6e}")

    print("\n[BEST]")
    print(" ", best[3])
    print(f"  mean={best[0]:.6e}  p99={best[1]:.6e}  max={best[2]:.6e}")

if __name__=="__main__":
    main()
