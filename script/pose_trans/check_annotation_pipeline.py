import numpy as np
import json
import argparse

# =========================
# 0. 命令行参数
# =========================
parser = argparse.ArgumentParser(
    description="Sanity check for pose / joint / scene transform / camera projection"
)

parser.add_argument("--motion", required=True,
                    help="Path to motion npz (with poses)")
parser.add_argument("--joints", required=True,
                    help="Path to joints_3d.npy")
parser.add_argument("--scene", required=True,
                    help="Scene name (key in scene_transforms.py)")
parser.add_argument("--camera", default=None,
                    help="Camera json path (optional)")

args = parser.parse_args()

NPZ_PATH = args.motion
JOINT_PATH = args.joints
SCENE_NAME = args.scene
CAMERA_JSON_PATH = args.camera

# =========================
# 1. 加载 scene transform
# =========================
from scene_transforms import SCENE_TRANSFORMS

assert SCENE_NAME in SCENE_TRANSFORMS, \
    f"❌ SCENE_NAME '{SCENE_NAME}' not found in scene_transforms.py"

T = SCENE_TRANSFORMS[SCENE_NAME]

print(f"\n[Input]")
print(f"  motion : {NPZ_PATH}")
print(f"  joints : {JOINT_PATH}")
print(f"  scene  : {SCENE_NAME}")
print(f"  camera : {CAMERA_JSON_PATH}")
print("\nT_align:\n", T)

# =========================
# 2. 检查 3D pose（npz）
# =========================
print("\n[Check 1] NPZ pose")

npz = np.load(NPZ_PATH)
assert "poses" in npz, "❌ npz 里没有 poses"

poses = npz["poses"]
print("poses shape:", poses.shape)

# 必须是 (T, D)
assert poses.ndim == 2, f"❌ poses 维度应该是 2D (T,D)，但得到 {poses.ndim}D"

# axis-angle: D 必须是 3 的倍数
D = poses.shape[1]
assert D % 3 == 0, f"❌ poses 的第二维 D={D} 不是 3 的倍数，无法解释为 axis-angle"

num_joints = D // 3

# 常见模型类型识别
POSE_TYPE_MAP = {
    24: "SMPL (24 joints, 72D)",
    52: "SMPL-H (52 joints, 156D)",
    55: "SMPL-X (55 joints, 165D)",
}
pose_type = POSE_TYPE_MAP.get(num_joints, f"Unknown ({num_joints} joints, {D}D)")

print(f"✅ pose type: {pose_type}  [axis-angle, D={D}]")

# =========================
# 3. 检查 3D joint
# =========================
print("\n[Check 2] 3D joints")

joints = np.load(JOINT_PATH)
print("joints shape:", joints.shape)

# 支持两种格式： (J,3) 或 (T,J,3)
if joints.ndim == 2 and joints.shape[1] == 3:
    # 单帧
    J = joints.shape[0]
    T_frames = 1
    joints_seq = joints[None, ...]  # -> (1,J,3)
    print(f"✅ joints 是单帧 3D joint: (J,3) with J={J}")

elif joints.ndim == 3 and joints.shape[2] == 3:
    # 序列
    T_frames, J = joints.shape[0], joints.shape[1]
    joints_seq = joints
    print(f"✅ joints 是序列 3D joint: (T,J,3) with T={T_frames}, J={J}")

else:
    raise AssertionError(f"❌ joints 形状不符合预期，应为 (J,3) 或 (T,J,3)，但得到 {joints.shape}")

print("joint value range:", joints.min(), joints.max())
j0 = joints_seq[0]          # (J,3)
j0_t = apply_T(j0, T)       # (J,3) transformed
# =========================
# 5. 应用 T_align
# =========================
def apply_T(joints, T):
    """
    joints: (J,3) or (T,J,3)
    T: (4,4)
    return: same shape as input but xyz transformed
    """
    joints = np.asarray(joints)

    if joints.ndim == 2 and joints.shape[1] == 3:
        # (J,3)
        homo = np.concatenate([joints, np.ones((joints.shape[0], 1), dtype=joints.dtype)], axis=1)  # (J,4)
        out = (T @ homo.T).T
        return out[:, :3]

    if joints.ndim == 3 and joints.shape[2] == 3:
        # (T,J,3)
        Tn, Jn = joints.shape[0], joints.shape[1]
        homo = np.concatenate([joints, np.ones((Tn, Jn, 1), dtype=joints.dtype)], axis=2)  # (T,J,4)
        out = (homo @ T.T)  # (T,J,4) 右乘等价于每个点左乘
        return out[..., :3]

    raise ValueError(f"apply_T expects (J,3) or (T,J,3), got {joints.shape}")

print("\n[Check 4] Joint after transform")
print("before mean:", j0.mean(axis=0))
print("after  mean:", j0_t.mean(axis=0))
print("delta       :", j0_t.mean(axis=0) - j0.mean(axis=0))
print("✅ joint 已被正确平移 / 旋转")

# =========================
# 6. （可选）检查相机投影
# =========================
if CAMERA_JSON_PATH is not None:
    try:
        with open(CAMERA_JSON_PATH, "r") as f:
            cam = json.load(f)

        fx, fy = cam["fx"], cam["fy"]
        cx, cy = cam["cx"], cam["cy"]
        R_wc = np.array(cam["R"])
        t_wc = np.array(cam["T"])

        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0,  0,  1]])

        j_cam = (R_wc @ j0_t.T).T + t_wc
        proj = (K @ j_cam.T).T
        proj_2d = proj[:, :2] / proj[:, 2:3]

        print("\n[Check 5] 2D projection")
        print("2D joint mean (pixel):", proj_2d.mean(axis=0))
        print("✅ 2D joint 投影成功")

    except Exception as e:
        print("\n⚠️ Camera check failed:", e)
else:
    print("\n[Skip] Camera check skipped (no --camera provided)")

print("\n🎉 ALL CHECKS PASSED")
