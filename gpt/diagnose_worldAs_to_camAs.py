#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, argparse, numpy as np
import torch, smplx

def normalize_T4(T):
    T = np.array(T, dtype=np.float32)
    if T.shape != (4,4):
        raise ValueError(f"Expect 4x4, got {T.shape}")
    # if translation is in last row -> transpose
    if abs(T[3,3]-1.0)<1e-4 and np.linalg.norm(T[:3,3])<1e-4 and np.linalg.norm(T[3,:3])>1e-4:
        T = T.T
    return T.astype(np.float32)

def get_raw(cam_json):
    cam=json.load(open(cam_json,"r"))
    def take(key):
        v=cam[key]
        return v[0] if isinstance(v,list) else v
    M_wvt = normalize_T4(take("world_view_transform")) if "world_view_transform" in cam else None
    C_c2w = normalize_T4(take("c2w")) if "c2w" in cam else None
    return M_wvt, C_c2w, cam

def apply_T_col(joints, T4):
    # column-vector: p' = T @ p
    Tn,J,_=joints.shape
    ones=np.ones((Tn,J,1),np.float32)
    ph=np.concatenate([joints,ones],-1)            # (T,J,4)
    out=(T4[None,None,:,:] @ ph[...,None]).squeeze(-1)
    return out[...,:3].astype(np.float32)

def apply_T_row(joints, T4):
    # row-vector: p' = p @ T   (equiv to column with T^T)
    return apply_T_col(joints, T4.T)

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

def forward_joints(npz_path, model_path, gender="neutral", num_betas=10, device="cuda"):
    d=np.load(npz_path, allow_pickle=True)
    poses=d["poses"].astype(np.float32)
    trans=d["trans"].astype(np.float32)
    betas=d["betas"].astype(np.float32) if "betas" in d.files else None
    T=poses.shape[0]

    poses_t=torch.from_numpy(poses).to(device)
    trans_t=torch.from_numpy(trans).to(device)
    betas_t=torch.from_numpy(make_betas_batch(betas,T,num_betas)).to(device)

    kwargs=split_pose_smplx(poses_t)
    kwargs["transl"]=trans_t
    kwargs["betas"]=betas_t

    model=smplx.create(model_path=model_path, model_type="smplx", gender=gender,
                       num_betas=num_betas, use_pca=False, batch_size=T).to(device)
    model.eval()
    with torch.no_grad():
        out=model(**kwargs, return_verts=False)
        return out.joints.detach().cpu().numpy().astype(np.float32)  # (T,127,3)

def stats(a,b):
    l2=np.linalg.norm(a-b, axis=-1)
    return float(l2.mean()), float(np.percentile(l2,99)), float(l2.max())

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--world_as_npz", required=True)
    ap.add_argument("--cam_as_npz", required=True)
    ap.add_argument("--camera_json", required=True)
    ap.add_argument("--model_path", required=True)   # MUST be .../smpl_files/smplx
    ap.add_argument("--gender", default="neutral", choices=["male","female","neutral"])
    ap.add_argument("--num_betas", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    args=ap.parse_args()

    jw=forward_joints(args.world_as_npz, args.model_path, args.gender, args.num_betas, args.device)
    jc=forward_joints(args.cam_as_npz,   args.model_path, args.gender, args.num_betas, args.device)
    Tmin=min(jw.shape[0], jc.shape[0])
    jw=jw[:Tmin]; jc=jc[:Tmin]

    M_wvt, C_c2w, cam = get_raw(args.camera_json)

    candidates=[]
    if M_wvt is not None:
        candidates += [
            ("WVT as W2C (col)", M_wvt, "col"),
            ("WVT as W2C (row)", M_wvt, "row"),
            ("inv(WVT) as W2C (col)", np.linalg.inv(M_wvt).astype(np.float32), "col"),
            ("inv(WVT) as W2C (row)", np.linalg.inv(M_wvt).astype(np.float32), "row"),
        ]
    if C_c2w is not None:
        W2C = np.linalg.inv(C_c2w).astype(np.float32)
        candidates += [
            ("inv(C2W) as W2C (col)", W2C, "col"),
            ("inv(C2W) as W2C (row)", W2C, "row"),
        ]

    best=None
    for name, W2C, mode in candidates:
        jw_cam = apply_T_col(jw, W2C) if mode=="col" else apply_T_row(jw, W2C)
        mean,p99,mx = stats(jw_cam, jc)
        print(f"{name:28s}  mean={mean:.6e}  p99={p99:.6e}  max={mx:.6e}")
        if best is None or mean < best[0]:
            best=(mean,p99,mx,name)

    print("\n[BEST]")
    print(" ", best[3])
    print(f"  mean={best[0]:.6e}  p99={best[1]:.6e}  max={best[2]:.6e}")

if __name__=="__main__":
    main()
