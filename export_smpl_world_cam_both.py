#!/usr/bin/env python3
import os, json, argparse
import numpy as np
from importlib.machinery import SourceFileLoader
from scipy.spatial.transform import Rotation as Rot

# ------------------------- utils -------------------------
def load_scene_transforms(scene_transforms_py: str):
    mod = SourceFileLoader("scene_transforms_dyn", scene_transforms_py).load_module()
    if not hasattr(mod, "SCENE_TRANSFORMS"):
        raise AttributeError(f"{scene_transforms_py} must define SCENE_TRANSFORMS dict")
    return mod.SCENE_TRANSFORMS

def load_camera_w2c(camera_json: str, cam_index: int):
    with open(camera_json, "r") as f:
        cam = json.load(f)
    if "c2w" not in cam:
        raise KeyError(f"{camera_json} missing key 'c2w'")
    c2w_list = cam["c2w"]
    if len(c2w_list) == 0:
        raise ValueError(f"{camera_json} has empty c2w list")
    if cam_index < 0 or cam_index >= len(c2w_list):
        raise IndexError(f"cam_index {cam_index} out of range, c2w len={len(c2w_list)}")

    c2w = np.array(c2w_list[cam_index], dtype=np.float32)
    if c2w.shape != (4, 4):
        raise ValueError(f"c2w must be 4x4, got {c2w.shape}")
    w2c = np.linalg.inv(c2w).astype(np.float32)
    return w2c

def find_key(npz, candidates):
    for k in candidates:
        if k in npz:
            return k
    return None

def closest_rotation(A3: np.ndarray) -> np.ndarray:
    """Project 3x3 matrix to nearest proper rotation SO(3) using SVD."""
    U, _, Vt = np.linalg.svd(A3)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R.astype(np.float32)

def decompose_A(A3: np.ndarray):
    """
    Decompose A3 into:
      s: average column norm (uniform-ish scale)
      R: nearest rotation of (A3 / s)
    Also return column norms for diagnostics.
    """
    col_norms = np.linalg.norm(A3, axis=0)
    s = float(np.mean(col_norms))
    if s < 1e-8:
        s = 1.0
    R0 = A3 / s
    R = closest_rotation(R0)
    return s, R, col_norms

def apply_global_orient(poses: np.ndarray, R_left: np.ndarray) -> np.ndarray:
    """poses[:,0:3] axis-angle, left-multiply rotation: R_new = R_left * R_old"""
    aa = poses[:, 0:3].astype(np.float32)
    R_old = Rot.from_rotvec(aa)
    R_l = Rot.from_matrix(R_left.astype(np.float32))
    R_new = R_l * R_old
    aa_new = R_new.as_rotvec().astype(np.float32)
    out = poses.copy().astype(np.float32)
    out[:, 0:3] = aa_new
    return out

def apply_trans(trans: np.ndarray, A: np.ndarray, t: np.ndarray) -> np.ndarray:
    """trans' = A @ trans + t"""
    trans = trans.astype(np.float32)
    return (A.astype(np.float32) @ trans.T).T + t.astype(np.float32)[None, :]

def save_npz_like(src_npz, out_path, poses_key, trans_key, poses, trans, meta: dict):
    out = {k: src_npz[k] for k in src_npz.files}
    out[poses_key] = poses
    out[trans_key] = trans
    # attach meta (safe types)
    for k, v in meta.items():
        out[k] = v
    np.savez(out_path, **out)

# ------------------------- main export -------------------------
def export_one_scene(npz_in, out_dir, base, scene_name, T_scene, w2c=None, cam_tag="cam00"):
    d = np.load(npz_in, allow_pickle=True)

    poses_key = find_key(d, ["poses", "fullpose"])
    if poses_key is None:
        raise KeyError(f"{npz_in} missing poses/fullpose. keys={list(d.files)}")
    trans_key = find_key(d, ["trans", "transl"])
    if trans_key is None:
        trans_key = "trans"
        trans = np.zeros((d[poses_key].shape[0], 3), dtype=np.float32)
    else:
        trans = d[trans_key].astype(np.float32)

    poses = d[poses_key].astype(np.float32)

    T_scene = np.array(T_scene, dtype=np.float32)
    if T_scene.shape != (4, 4):
        raise ValueError(f"{scene_name}: T_scene must be 4x4, got {T_scene.shape}")

    A_raw = T_scene[:3, :3].astype(np.float32)
    t_scene = T_scene[:3, 3].astype(np.float32)

    s, R_scene, col_norms = decompose_A(A_raw)

    # ---------- Variant 1: Rt (rotation-only + translation)
    poses_world_Rt = apply_global_orient(poses, R_scene)
    trans_world_Rt = apply_trans(trans, R_scene, t_scene)

    meta_Rt = {
        "scene_name": np.array(scene_name),
        "scene_scale": np.array(s, dtype=np.float32),
        "scene_col_norms": col_norms.astype(np.float32),
        "scene_A_raw": A_raw.astype(np.float32),
        "scene_R_used": R_scene.astype(np.float32),
        "scene_t": t_scene.astype(np.float32),
        "variant": np.array("world_Rt"),
    }

    p_world_Rt = os.path.join(out_dir, f"{base}_{scene_name}_world_Rt.npz")
    save_npz_like(d, p_world_Rt, poses_key, trans_key, poses_world_Rt, trans_world_Rt, meta_Rt)

    # ---------- Variant 2: As (full A + translation for trans; rotation-only for global_orient)
    # Note: global_orient still MUST use rotation (cannot encode scale/shear)
    poses_world_As = apply_global_orient(poses, R_scene)
    trans_world_As = apply_trans(trans, A_raw, t_scene)

    meta_As = meta_Rt.copy()
    meta_As["variant"] = np.array("world_As")

    p_world_As = os.path.join(out_dir, f"{base}_{scene_name}_world_As.npz")
    save_npz_like(d, p_world_As, poses_key, trans_key, poses_world_As, trans_world_As, meta_As)

    out_paths = [p_world_Rt, p_world_As]

    # ---------- Camera-space exports (optional)
    if w2c is not None:
        R_w2c = w2c[:3, :3].astype(np.float32)
        t_w2c = w2c[:3, 3].astype(np.float32)

        # camera from world_Rt
        poses_cam_Rt = apply_global_orient(poses_world_Rt, R_w2c)
        trans_cam_Rt = apply_trans(trans_world_Rt, R_w2c, t_w2c)

        meta_cam_Rt = meta_Rt.copy()
        meta_cam_Rt["variant"] = np.array(f"{cam_tag}_Rt")
        meta_cam_Rt["w2c_R"] = R_w2c
        meta_cam_Rt["w2c_t"] = t_w2c

        p_cam_Rt = os.path.join(out_dir, f"{base}_{scene_name}_{cam_tag}_Rt.npz")
        save_npz_like(d, p_cam_Rt, poses_key, trans_key, poses_cam_Rt, trans_cam_Rt, meta_cam_Rt)

        # camera from world_As
        poses_cam_As = apply_global_orient(poses_world_As, R_w2c)
        trans_cam_As = apply_trans(trans_world_As, R_w2c, t_w2c)

        meta_cam_As = meta_As.copy()
        meta_cam_As["variant"] = np.array(f"{cam_tag}_As")
        meta_cam_As["w2c_R"] = R_w2c
        meta_cam_As["w2c_t"] = t_w2c

        p_cam_As = os.path.join(out_dir, f"{base}_{scene_name}_{cam_tag}_As.npz")
        save_npz_like(d, p_cam_As, poses_key, trans_key, poses_cam_As, trans_cam_As, meta_cam_As)

        out_paths += [p_cam_Rt, p_cam_As]

    return out_paths, s, col_norms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_npz", required=True)
    ap.add_argument("--scene_transforms_py", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--scenes", nargs="*", default=None, help="default: all scenes")
    ap.add_argument("--camera_json", default=None, help="if set, also export camera-space (use inv(c2w))")
    ap.add_argument("--cam_index", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    SCENE_TRANSFORMS = load_scene_transforms(args.scene_transforms_py)
    scenes = args.scenes if args.scenes else list(SCENE_TRANSFORMS.keys())

    base = os.path.splitext(os.path.basename(args.input_npz))[0]

    w2c = None
    cam_tag = None
    if args.camera_json:
        w2c = load_camera_w2c(args.camera_json, args.cam_index)
        cam_tag = f"cam{args.cam_index:02d}"

    # export
    for sname in scenes:
        if sname not in SCENE_TRANSFORMS:
            raise KeyError(f"scene '{sname}' not in SCENE_TRANSFORMS. available={list(SCENE_TRANSFORMS.keys())}")

        out_paths, scale_s, col_norms = export_one_scene(
            npz_in=args.input_npz,
            out_dir=args.out_dir,
            base=base,
            scene_name=sname,
            T_scene=SCENE_TRANSFORMS[sname],
            w2c=w2c,
            cam_tag=cam_tag if cam_tag else "cam00",
        )

        print(f"[OK] {sname}  scale≈{scale_s:.6f}  col_norms={col_norms} ")
        for p in out_paths:
            print("   ->", p)

if __name__ == "__main__":
    main()
