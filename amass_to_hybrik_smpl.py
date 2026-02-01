#!/usr/bin/env python3
"""
Convert AMASS motion sequence NPZ to HybrIK-compatible SMPL training data.

Pipeline:
1. Load AMASS NPZ (poses, trans, betas)
2. Apply scene transform (CloudCompare 4x4 matrix) → world coordinates
3. Apply camera transform → camera coordinates
4. Project joints to 2D
5. Save in HybrIK training format

Usage:
    python amass_to_hybrik_smpl.py \
        --amass_npz /path/to/motion.npz \
        --scene djr_p1 \
        --camera_json /path/to/top.json \
        --output_dir /path/to/output \
        --subject_name 101010 \
        --pose_name walking

    # With SMPL model for joint generation
    python amass_to_hybrik_smpl.py \
        --amass_npz /path/to/motion.npz \
        --scene djr_p1 \
        --camera_json /path/to/top.json \
        --output_dir /path/to/output \
        --smpl_model_path /path/to/smpl_models
"""

import os
import json
import numpy as np
from scipy.spatial.transform import Rotation
from argparse import ArgumentParser
from tqdm import tqdm


# ============================================================
# Scene Transforms (from scene_transforms.py)
# ============================================================
def load_scene_transforms(transform_file=None):
    """Load SCENE_TRANSFORMS dict from file."""
    if transform_file is None:
        try:
            from scene_transforms import SCENE_TRANSFORMS
            return SCENE_TRANSFORMS
        except ImportError:
            raise ImportError(
                "Could not import scene_transforms.py. "
                "Either place it in current directory or specify --transform_file"
            )
    else:
        import importlib.util
        spec = importlib.util.spec_from_file_location("scene_transforms", transform_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.SCENE_TRANSFORMS


# ============================================================
# Camera Utils
# ============================================================
def parse_3dgs_camera(json_path, frame_idx=0):
    """
    Parse 3DGS style camera JSON file.
    
    Note: 3DGS stores matrices in column-major (OpenGL) format, need to transpose.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # World-to-Camera transform (need transpose for row-major)
    W2C = np.array(data['world_view_transform'][frame_idx], dtype=np.float64).T
    
    R_cam = W2C[:3, :3]  # Rotation: world → camera
    T_cam = W2C[:3, 3]   # Translation: world → camera
    
    # Image size
    H = data['image_height'][frame_idx]
    W = data['image_width'][frame_idx]
    
    # FOV
    fovx = data.get('fovx', [None])[frame_idx]
    fovy = data.get('fovy', [None])[frame_idx]
    
    # Intrinsics: try to load directly, or compute from FOV
    if 'intrinsics' in data:
        K = np.array(data['intrinsics'][frame_idx], dtype=np.float64)
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
    elif fovx is not None and fovy is not None:
        # Compute intrinsics from FOV
        # fx = W / (2 * tan(fovx / 2))
        fx = W / (2 * np.tan(fovx / 2))
        fy = H / (2 * np.tan(fovy / 2))
        cx = W / 2.0
        cy = H / 2.0
        
        K = np.array([
            [fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1]
        ], dtype=np.float64)
    else:
        raise ValueError("Camera JSON must have either 'intrinsics' or 'fovx'/'fovy'")
    
    return {
        'W2C': W2C,
        'R_cam': R_cam,
        'T_cam': T_cam,
        'K': K,
        'H': H,
        'W': W,
        'fovx': fovx,
        'fovy': fovy,
        'fx': fx,
        'fy': fy,
        'cx': cx,
        'cy': cy,
    }


def get_camera_per_frame(json_path, num_frames):
    """
    Get camera parameters for each frame.
    If camera file has fewer entries, repeat the last one.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    num_cams = len(data['world_view_transform'])
    cameras = []
    
    for i in range(num_frames):
        cam_idx = min(i, num_cams - 1)
        cameras.append(parse_3dgs_camera(json_path, cam_idx))
    
    return cameras


# ============================================================
# Transform Utils
# ============================================================
def decompose_transform(T_4x4):
    """Decompose 4x4 matrix into pure rotation, translation, and scale."""
    R = T_4x4[:3, :3]
    t = T_4x4[:3, 3]
    
    scale = np.array([
        np.linalg.norm(R[:, 0]),
        np.linalg.norm(R[:, 1]),
        np.linalg.norm(R[:, 2])
    ])
    
    R_pure = R.copy()
    R_pure[:, 0] /= scale[0]
    R_pure[:, 1] /= scale[1]
    R_pure[:, 2] /= scale[2]
    
    det = np.linalg.det(R_pure)
    if det < 0:
        R_pure[:, 0] *= -1
        scale[0] *= -1
    
    return R_pure, t, scale


def transform_points(points, T_4x4):
    """Transform points using 4x4 homogeneous matrix."""
    if points.ndim == 1:
        points = points.reshape(1, -1)
    N = points.shape[0]
    points_h = np.hstack([points, np.ones((N, 1))])
    transformed = (T_4x4 @ points_h.T).T[:, :3]
    return transformed


def transform_global_orient(global_orient, R):
    """Transform global orientation by rotation matrix."""
    go_new = []
    for go in global_orient:
        R_body = Rotation.from_rotvec(go).as_matrix()
        R_new = R @ R_body
        go_new.append(Rotation.from_matrix(R_new).as_rotvec())
    return np.array(go_new)


# ============================================================
# Projection Utils
# ============================================================
def world_to_camera(points, R_cam, T_cam):
    """Transform points from world to camera coordinates."""
    return (R_cam @ points.T).T + T_cam


def camera_to_pixel(points_cam, K):
    """Project camera coordinates to pixel coordinates."""
    points_proj = (K @ points_cam.T).T
    points_2d = points_proj[:, :2] / points_proj[:, 2:3]
    return points_2d


def project_to_2d(points_world, camera):
    """Full projection: world → camera → pixel."""
    points_cam = world_to_camera(points_world, camera['R_cam'], camera['T_cam'])
    points_2d = camera_to_pixel(points_cam, camera['K'])
    return points_2d, points_cam


# ============================================================
# SMPL Utils
# ============================================================
def load_amass_npz(npz_path):
    """Load AMASS format NPZ file."""
    data = dict(np.load(npz_path, allow_pickle=True))
    
    result = {
        'poses': data['poses'],           # (N, 156) or (N, 72)
        'trans': data['trans'],           # (N, 3)
        'betas': data['betas'],           # (16,) or (10,)
    }
    
    # Optional fields
    if 'gender' in data:
        gender = data['gender']
        if isinstance(gender, np.ndarray):
            gender = str(gender)
        result['gender'] = gender
    else:
        result['gender'] = 'neutral'
    
    if 'mocap_framerate' in data:
        result['mocap_framerate'] = float(data['mocap_framerate'])
    
    return result


def get_smpl_joints(smpl_params, smpl_model_path, model_type='smpl'):
    """
    Generate SMPL joints from parameters using smplx library.
    
    Returns:
        joints: (N, J, 3) joint positions
    """
    try:
        import torch
        import smplx
    except ImportError:
        raise ImportError("Please install smplx and torch: pip install smplx torch")
    
    poses = smpl_params['poses']
    trans = smpl_params['trans']
    betas = smpl_params['betas']
    gender = smpl_params.get('gender', 'neutral')
    
    # Clean gender string
    if isinstance(gender, bytes):
        gender = gender.decode('utf-8')
    if gender.startswith("b'"):
        gender = gender[2:-1]
    if gender not in ['male', 'female', 'neutral']:
        gender = 'neutral'
    
    N = poses.shape[0]
    pose_dim = poses.shape[1]
    
    # Determine model type and body pose end index
    if pose_dim <= 72:
        model_type = 'smpl'
        body_pose_end = 72
        num_joints = 24
    else:
        model_type = 'smplx'
        body_pose_end = 66
        num_joints = 22
    
    # Load SMPL model
    body_model = smplx.create(
        smpl_model_path,
        model_type=model_type,
        gender=gender,
        batch_size=1
    )
    
    # Generate joints frame by frame
    all_joints = []
    
    for i in tqdm(range(N), desc="Generating SMPL joints"):
        with torch.no_grad():
            output = body_model(
                global_orient=torch.tensor(poses[i:i+1, :3]).float(),
                body_pose=torch.tensor(poses[i:i+1, 3:body_pose_end]).float(),
                betas=torch.tensor(betas[:10]).float().unsqueeze(0),
                transl=torch.tensor(trans[i:i+1]).float()
            )
            
            # Get joints (first 24 for SMPL compatibility)
            joints = output.joints[0, :24].numpy()
            all_joints.append(joints)
    
    return np.array(all_joints)


def extract_smpl_for_hybrik(poses, trans, betas):
    """
    Extract SMPL parameters in HybrIK format.
    
    SMPL-X (156 dims) → SMPL (72 dims):
    - global_orient: [0:3]
    - body_pose: [3:66] (21 joints * 3)
    
    SMPL (72 dims):
    - global_orient: [0:3]
    - body_pose: [3:72] (23 joints * 3)
    """
    pose_dim = poses.shape[1]
    
    if pose_dim >= 156:
        # SMPL-X → SMPL
        global_orient = poses[:, :3]
        body_pose = poses[:, 3:66]  # 21 joints for SMPL-X body
    elif pose_dim >= 72:
        # SMPL
        global_orient = poses[:, :3]
        body_pose = poses[:, 3:72]  # 23 joints for SMPL body
    else:
        raise ValueError(f"Unexpected pose dimension: {pose_dim}")
    
    return {
        'global_orient': global_orient,  # (N, 3)
        'body_pose': body_pose,          # (N, 63) or (N, 69)
        'betas': betas[:10],             # (10,)
        'trans': trans,                  # (N, 3)
    }


# ============================================================
# Main Processing
# ============================================================
def process_amass_to_hybrik(
    amass_npz,
    scene_transform,
    camera_json,
    output_dir,
    subject_name='unknown',
    pose_name='unknown',
    smpl_model_path=None,
    joints_npz=None,
):
    """
    Full pipeline: AMASS NPZ → HybrIK training data.
    
    Args:
        amass_npz: Path to AMASS motion NPZ
        scene_transform: 4x4 numpy array (CloudCompare transform)
        camera_json: Path to 3DGS camera JSON
        output_dir: Output directory
        subject_name: Subject identifier
        pose_name: Pose/motion identifier
        smpl_model_path: Path to SMPL models (optional, for joint generation)
        joints_npz: Path to pre-computed joints NPZ (optional)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # ========== 1. Load AMASS data ==========
    print("\n[1] Loading AMASS data...")
    amass_data = load_amass_npz(amass_npz)
    
    poses = amass_data['poses']
    trans = amass_data['trans']
    betas = amass_data['betas']
    gender = amass_data['gender']
    
    N = poses.shape[0]
    print(f"    Frames: {N}")
    print(f"    Pose dim: {poses.shape[1]}")
    print(f"    Gender: {gender}")
    
    # ========== 2. Load/Generate joints ==========
    print("\n[2] Loading/Generating joints...")
    
    if joints_npz is not None:
        # Load pre-computed joints
        joints_data = np.load(joints_npz)
        joints_original = joints_data['joints_3d']
        print(f"    Loaded joints from: {joints_npz}")
    elif smpl_model_path is not None:
        # Generate joints using SMPL model
        joints_original = get_smpl_joints(amass_data, smpl_model_path)
        print(f"    Generated joints using SMPL model")
    else:
        print("    WARNING: No joints provided. Use --smpl_model_path or --joints_npz")
        joints_original = None
    
    if joints_original is not None:
        print(f"    Joints shape: {joints_original.shape}")
    
    # ========== 3. Decompose scene transform ==========
    print("\n[3] Analyzing scene transform...")
    R_scene_pure, t_scene, scale_scene = decompose_transform(scene_transform)
    print(f"    Scale: {scale_scene}")
    print(f"    Has non-uniform scale: {not np.allclose(scale_scene, 1.0, atol=1e-3)}")
    
    # ========== 4. Load camera ==========
    print("\n[4] Loading camera parameters...")
    camera = parse_3dgs_camera(camera_json, frame_idx=0)
    print(f"    Image size: {camera['W']} x {camera['H']}")
    print(f"    Focal length: fx={camera['fx']:.2f}, fy={camera['fy']:.2f}")
    print(f"    Principal point: cx={camera['cx']:.2f}, cy={camera['cy']:.2f}")
    
    # ========== 5. Transform to world (scene) coordinates ==========
    print("\n[5] Transforming to scene (world) coordinates...")
    
    # Transform trans
    trans_world = transform_points(trans, scene_transform)
    
    # Transform global_orient (use pure rotation, no scale)
    global_orient_world = transform_global_orient(poses[:, :3], R_scene_pure)
    
    # Combine poses
    poses_world = np.concatenate([global_orient_world, poses[:, 3:]], axis=1)
    
    # Transform joints
    if joints_original is not None:
        joints_world = np.zeros_like(joints_original)
        for i in range(N):
            joints_world[i] = transform_points(joints_original[i], scene_transform)
    
    print(f"    Trans world: {trans_world.shape}")
    
    # ========== 6. Transform to camera coordinates ==========
    print("\n[6] Transforming to camera coordinates...")
    
    R_cam = camera['R_cam']
    T_cam = camera['T_cam']
    
    # Transform trans to camera coords
    trans_cam = world_to_camera(trans_world, R_cam, T_cam)
    
    # Transform global_orient to camera coords
    global_orient_cam = transform_global_orient(global_orient_world, R_cam)
    
    # Combine poses in camera coords
    poses_cam = np.concatenate([global_orient_cam, poses[:, 3:]], axis=1)
    
    # Transform joints to camera coords
    if joints_original is not None:
        joints_cam = np.zeros_like(joints_world)
        for i in range(N):
            joints_cam[i] = world_to_camera(joints_world[i], R_cam, T_cam)
    
    print(f"    Trans camera: {trans_cam.shape}")
    
    # ========== 7. Project to 2D ==========
    print("\n[7] Projecting to 2D...")
    
    if joints_original is not None:
        K = camera['K']
        joints_2d = np.zeros((N, joints_cam.shape[1], 2))
        
        for i in range(N):
            joints_2d[i] = camera_to_pixel(joints_cam[i], K)
        
        # Check visibility (inside image bounds)
        H, W = camera['H'], camera['W']
        visibility = np.ones((N, joints_2d.shape[1]), dtype=np.float32)
        
        for i in range(N):
            for j in range(joints_2d.shape[1]):
                x, y = joints_2d[i, j]
                if x < 0 or x >= W or y < 0 or y >= H:
                    visibility[i, j] = 0.0
                # Also check if behind camera
                if joints_cam[i, j, 2] <= 0:
                    visibility[i, j] = 0.0
        
        print(f"    Joints 2D: {joints_2d.shape}")
        print(f"    Avg visibility: {visibility.mean():.2%}")
    
    # ========== 8. Extract HybrIK format ==========
    print("\n[8] Extracting HybrIK format...")
    
    smpl_hybrik = extract_smpl_for_hybrik(poses_cam, trans_cam, betas)
    
    # ========== 9. Save results ==========
    print("\n[9] Saving results...")
    
    prefix = f"{subject_name}_{pose_name}"
    
    # Save SMPL parameters (camera coordinates)
    smpl_output = {
        # SMPL in camera coordinates (for HybrIK training)
        'global_orient': smpl_hybrik['global_orient'].astype(np.float32),
        'body_pose': smpl_hybrik['body_pose'].astype(np.float32),
        'betas': smpl_hybrik['betas'].astype(np.float32),
        'trans': smpl_hybrik['trans'].astype(np.float32),
        
        # Also save world coordinates version
        'global_orient_world': global_orient_world.astype(np.float32),
        'trans_world': trans_world.astype(np.float32),
        
        # Original AMASS poses (for reference)
        'poses_original': poses.astype(np.float32),
        'trans_original': trans.astype(np.float32),
        
        # Metadata
        'gender': np.array(gender),
        'num_frames': np.array(N),
        'scene_transform': scene_transform.astype(np.float64),
        'scene_scale': scale_scene.astype(np.float32),
    }
    
    smpl_path = os.path.join(output_dir, f"{prefix}_smpl.npz")
    np.savez(smpl_path, **smpl_output)
    print(f"    SMPL saved: {smpl_path}")
    
    # Save joints
    if joints_original is not None:
        joints_output = {
            'joints_3d_cam': joints_cam.astype(np.float32),
            'joints_3d_world': joints_world.astype(np.float32),
            'joints_2d': joints_2d.astype(np.float32),
            'visibility': visibility.astype(np.float32),
        }
        
        joints_path = os.path.join(output_dir, f"{prefix}_joints.npz")
        np.savez(joints_path, **joints_output)
        print(f"    Joints saved: {joints_path}")
    
    # Save camera parameters
    camera_output = {
        'K': camera['K'].astype(np.float64),
        'R': camera['R_cam'].astype(np.float64),
        'T': camera['T_cam'].astype(np.float64),
        'W2C': camera['W2C'].astype(np.float64),
        'image_width': np.array(camera['W']),
        'image_height': np.array(camera['H']),
        'fx': np.array(camera['fx']),
        'fy': np.array(camera['fy']),
        'cx': np.array(camera['cx']),
        'cy': np.array(camera['cy']),
    }
    
    camera_path = os.path.join(output_dir, f"{prefix}_camera.npz")
    np.savez(camera_path, **camera_output)
    print(f"    Camera saved: {camera_path}")
    
    # Save per-frame annotations (HybrIK style)
    print("\n[10] Saving per-frame annotations...")
    
    annot_dir = os.path.join(output_dir, 'annotations')
    os.makedirs(annot_dir, exist_ok=True)
    
    annotations = []
    for i in tqdm(range(N), desc="Creating annotations"):
        annot = {
            'frame_idx': i,
            'image_path': f"images/{i:08d}.png",  # Adjust path as needed
            
            # SMPL parameters (camera coords)
            'smpl': {
                'global_orient': smpl_hybrik['global_orient'][i].tolist(),
                'body_pose': smpl_hybrik['body_pose'][i].tolist(),
                'betas': smpl_hybrik['betas'].tolist(),
                'trans': smpl_hybrik['trans'][i].tolist(),
            },
            
            # Camera
            'camera': {
                'fx': float(camera['fx']),
                'fy': float(camera['fy']),
                'cx': float(camera['cx']),
                'cy': float(camera['cy']),
            },
        }
        
        if joints_original is not None:
            annot['joints_3d'] = joints_cam[i].tolist()
            annot['joints_2d'] = joints_2d[i].tolist()
            annot['visibility'] = visibility[i].tolist()
        
        annotations.append(annot)
    
    annot_path = os.path.join(annot_dir, f"{prefix}_annotations.json")
    with open(annot_path, 'w') as f:
        json.dump(annotations, f, indent=2)
    print(f"    Annotations saved: {annot_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Subject: {subject_name}")
    print(f"Pose: {pose_name}")
    print(f"Frames: {N}")
    print(f"Output directory: {output_dir}")
    print(f"\nFiles created:")
    print(f"  - {prefix}_smpl.npz (SMPL parameters)")
    if joints_original is not None:
        print(f"  - {prefix}_joints.npz (3D/2D joints)")
    print(f"  - {prefix}_camera.npz (camera parameters)")
    print(f"  - annotations/{prefix}_annotations.json (per-frame)")
    
    return {
        'smpl_path': smpl_path,
        'joints_path': joints_path if joints_original is not None else None,
        'camera_path': camera_path,
        'annot_path': annot_path,
        'num_frames': N,
    }


def main():
    parser = ArgumentParser(description="Convert AMASS NPZ to HybrIK training data")
    
    # Required
    parser.add_argument('--amass_npz', type=str, required=True,
                        help='Input AMASS motion NPZ file')
    parser.add_argument('--scene', type=str, required=True,
                        help='Scene key in scene_transforms.py (e.g., djr_p1)')
    parser.add_argument('--camera_json', type=str, required=True,
                        help='Path to 3DGS camera JSON file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory')
    
    # Optional
    parser.add_argument('--transform_file', type=str, default=None,
                        help='Path to scene_transforms.py')
    parser.add_argument('--subject_name', type=str, default='unknown',
                        help='Subject identifier')
    parser.add_argument('--pose_name', type=str, default='unknown',
                        help='Pose/motion identifier')
    parser.add_argument('--smpl_model_path', type=str, default=None,
                        help='Path to SMPL model files (for joint generation)')
    parser.add_argument('--joints_npz', type=str, default=None,
                        help='Path to pre-computed joints NPZ')
    
    args = parser.parse_args()
    
    # Load scene transform
    SCENE_TRANSFORMS = load_scene_transforms(args.transform_file)
    
    if args.scene not in SCENE_TRANSFORMS:
        print(f"Error: Scene '{args.scene}' not found!")
        print(f"Available: {list(SCENE_TRANSFORMS.keys())}")
        return
    
    scene_transform = np.array(SCENE_TRANSFORMS[args.scene], dtype=np.float64)
    
    # Run pipeline
    process_amass_to_hybrik(
        amass_npz=args.amass_npz,
        scene_transform=scene_transform,
        camera_json=args.camera_json,
        output_dir=args.output_dir,
        subject_name=args.subject_name,
        pose_name=args.pose_name,
        smpl_model_path=args.smpl_model_path,
        joints_npz=args.joints_npz,
    )


if __name__ == '__main__':
    main()
