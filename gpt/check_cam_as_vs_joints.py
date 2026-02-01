#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, re, glob, json, argparse
import numpy as np
import torch
import smplx

def normalize_T4(T):
    T = np.array(T, dtype=np.float32)
    if T.shape != (4,4):
        raise ValueError(f"Expect 4x4, got {T.shape}")
    # heuristic: translation stored in last row (tx,ty,tz,1) -> transpose
    if abs(T[3,3]-1.0)<1e-4 and np.linalg.norm(T[:3,3])<1e-4 and np.linalg.norm(T[3,:3])>1e-4:
        T = T.T
    return T.astype(np.float32)

def load_w2c(camera_json, frame_idx=0):
    cam = json.load(open(camera_json, "r"))
    if "world_view_transform" in cam:
        wvt = cam["world_view_transform"]
        M = wvt[min(frame_idx, len(wvt)-1)] if isinstance(wvt, list) else wvt
        return normalize_T4(M)
    if "w2c" in cam:
        M = cam["w2c"]
        M = M[min(frame_idx, len(M)-1)] if isinstance(M, list) else M
        return normalize_T4(M)
    if "extrinsic" in cam:
        M = cam["extrinsic"]
        M = M[min(frame_idx, len(M)-1)] if isinstance(M, list) else M
        return normalize_T4(M)
    if "c2w" in cam:
        C = cam["c2w"]
        C = C[min(frame_idx, len(C)-1)] if isinstance(C, list) else C
        C = normalize_T4(C)
        return np.linalg.inv(C).astype(np.float32)
    raise KeyError(f"Cannot find w2c/world_view_transform/c2w in {camera_json}. keys={list(cam.keys())}")

def load_joints_dir(dir_path, pattern="*.npy"):
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
    return np.stack(arr,0), files

def apply_T_to_joints(joints, T4):
    # joints: (T,22,3)
    Tn,J,_ = joints.shape
    ones = np.ones((Tn,J,1), dtype=np.float32)
    jh = np.concatenate([joints, ones], axis=-1)              # (T,J,4)
    out = (T4[None,None,:,:] @ jh[...,None]).squeeze(-1)      # (T,J,4)
    return out[...,:3].astype(np.float32)

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
    T,D = poses_t.shape
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

def forward_smplx_joints_from_npz(npz_path, model_path, gender="neutral", num_betas=10, device="cuda"):
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
        joints=out.joints.detach().cpu().numpy().astype(np.float32)  # (T,127,3)
    return joints

def stats(pred, gt):
    l2=np.linalg.norm(pred-gt, axis=-1) # (T,22)
    return dict(mean=float(l2.mean()), median=float(np.median(l2)),
                p90=float(np.percentile(l2,90)), p99=float(np.percentile(l2,99)), max=float(l2.max()))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cam_as_npz", required=True, help="smpl_*_cam_As_0_99.npz")
    ap.add_argument("--joints_dir_transformed", required=True, help="world/scene-space joints (22,3) per frame")
    ap.add_argument("--camera_json", required=True)
    ap.add_argument("--joint_index_22", required=True, help="len=22 mapping .npy")
    ap.add_argument("--model_path", required=True, help=".../smpl_files/smplx")
    ap.add_argument("--gender", default="neutral", choices=["male","female","neutral"])
    ap.add_argument("--num_betas", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save_gt_cam_dir", default=None, help="optional: save computed GT cam joints to this dir")
    args=ap.parse_args()

    # GT: transformed(world) -> cam
    gt_world, files = load_joints_dir(args.joints_dir_transformed)
    W2C = load_w2c(args.camera_json, frame_idx=0)
    gt_cam = apply_T_to_joints(gt_world, W2C)

    # Pred: cam_As npz -> smplx joints -> pick 22
    idx = np.load(args.joint_index_22).astype(np.int64)
    if idx.shape[0]!=22:
        raise ValueError("joint_index_22 must be len=22")
    pred_full = forward_smplx_joints_from_npz(args.cam_as_npz, args.model_path,
                                              args.gender, args.num_betas, args.device)
    pred22 = pred_full[:, idx, :]

    # crop to common T
    Tmin = min(gt_cam.shape[0], pred22.shape[0])
    if gt_cam.shape[0]!=pred22.shape[0]:
        print(f"[WARN] T mismatch gt={gt_cam.shape[0]} pred={pred22.shape[0]} -> crop {Tmin}")
    gt_cam = gt_cam[:Tmin]
    pred22 = pred22[:Tmin]

    st = stats(pred22, gt_cam)
    print("\n================== cam_As vs GT_cam_joints ==================")
    print("cam_as_npz :", args.cam_as_npz)
    print("gt_dir(w)  :", args.joints_dir_transformed)
    print("camera_json:", args.camera_json)
    print(f"error(m): mean={st['mean']:.6e} median={st['median']:.6e} p90={st['p90']:.6e} p99={st['p99']:.6e} max={st['max']:.6e}")

    if args.save_gt_cam_dir:
        os.makedirs(args.save_gt_cam_dir, exist_ok=True)
        for i in range(Tmin):
            out = os.path.join(args.save_gt_cam_dir, f"{i:08d}.npy")
            np.save(out, gt_cam[i].astype(np.float32))
        print("[OK] saved GT cam joints to:", args.save_gt_cam_dir)

if __name__=="__main__":
    main()
