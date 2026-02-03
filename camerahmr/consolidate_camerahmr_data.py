#!/usr/bin/env python3
"""
Consolidate SMPL data for CameraHMR training.

CameraHMR (https://github.com/pixelite1201/CameraHMR) data format:
- Uses similar format to HMR2.0 / 4DHumans
- Requires images, SMPL params, camera params
- Key focus: camera parameter prediction

Output structure:
    /output_dir/{subject}/
    ├── train.npz
    ├── val.npz
    ├── train_images.txt (list of image paths)
    └── stats.json

NPZ contents per sample:
    - imgname: image path
    - center: bbox center [cx, cy]
    - scale: bbox scale
    - pose: SMPL pose (72,) - axis-angle
    - shape: SMPL shape (10,)
    - has_smpl: flag (1 or 0)
    - keypoints_2d: 2D joints (J, 3) with confidence
    - keypoints_3d: 3D joints (J, 4) with confidence
    - cam_int: intrinsics [fx, fy, cx, cy]
    - cam_ext: extrinsics (4, 4)
    - focal_length: focal length value
    - img_h, img_w: image dimensions

Usage:
    python consolidate_camerahmr_data.py \
        --input_base /mnt/data_hdd/fzhi/output \
        --subject 101010 \
        --output_dir /mnt/data_hdd/fzhi/camerahmr_data \
        --val_ratio 0.1
"""

import os
import json
import glob
import pickle
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm


def compute_bbox_from_joints(joints_2d, img_shape, scale_factor=1.2):
    """
    Compute bbox center and scale from 2D joints.
    
    Returns:
        center: [cx, cy]
        scale: scalar (bbox size / 200, following SPIN convention)
    """
    H, W = img_shape
    
    valid_mask = ~np.isnan(joints_2d).any(axis=1)
    if not valid_mask.any():
        return np.array([W/2, H/2]), max(W, H) / 200.0
    
    valid_joints = joints_2d[valid_mask]
    
    x_min, y_min = valid_joints.min(axis=0)
    x_max, y_max = valid_joints.max(axis=0)
    
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    
    w = x_max - x_min
    h = y_max - y_min
    size = max(w, h) * scale_factor
    
    # Scale following SPIN/HMR convention (bbox_size / 200)
    scale = size / 200.0
    
    return np.array([cx, cy]), scale


def smplx_to_smpl_pose(global_orient, body_pose):
    """Convert SMPL-X pose to SMPL 72-dim pose."""
    if len(body_pose) == 63:
        # SMPL-X: pad with zeros for hand joints
        body_pose_padded = np.concatenate([body_pose, np.zeros(6)])
    elif len(body_pose) == 69:
        body_pose_padded = body_pose
    else:
        body_pose_padded = np.zeros(69)
        body_pose_padded[:min(len(body_pose), 69)] = body_pose[:69]
    
    smpl_pose = np.concatenate([global_orient, body_pose_padded])
    return smpl_pose  # (72,)


def find_all_smpl_dirs(input_base, subject):
    """Find all SMPL directories for a subject."""
    subject_dir = os.path.join(input_base, subject)
    smpl_dirs = []
    
    for root, dirs, files in os.walk(subject_dir):
        if 'smpl' in dirs:
            smpl_base = os.path.join(root, 'smpl')
            for camera in os.listdir(smpl_base):
                camera_dir = os.path.join(smpl_base, camera)
                if os.path.isdir(camera_dir):
                    if glob.glob(os.path.join(camera_dir, '*_smpl.npz')):
                        image_dir = os.path.join(root, 'image', camera)
                        if os.path.exists(image_dir):
                            smpl_dirs.append({
                                'smpl_dir': camera_dir,
                                'image_dir': image_dir,
                                'root': root,
                                'camera': camera,
                            })
    
    return smpl_dirs


def process_single_sequence(smpl_dir, image_dir, image_base_for_path):
    """Process a single sequence."""
    annotations = []
    
    smpl_files = glob.glob(os.path.join(smpl_dir, '*_smpl.npz'))
    if not smpl_files:
        return []
    
    smpl_file = smpl_files[0]
    joints_file = smpl_file.replace('_smpl.npz', '_joints.npz')
    camera_file = smpl_file.replace('_smpl.npz', '_camera.npz')
    
    if not os.path.exists(joints_file) or not os.path.exists(camera_file):
        return []
    
    # Load data
    smpl_data = np.load(smpl_file, allow_pickle=True)
    joints_data = np.load(joints_file, allow_pickle=True)
    camera_data = np.load(camera_file, allow_pickle=True)
    
    global_orient = smpl_data['global_orient']  # (N, 3)
    body_pose = smpl_data['body_pose']          # (N, 63)
    betas = smpl_data['betas']                  # (10,)
    trans = smpl_data['trans']                  # (N, 3)
    
    joints_3d_cam = joints_data['joints_3d_cam']  # (N, J, 3)
    joints_2d = joints_data['joints_2d']          # (N, J, 2)
    visibility = joints_data.get('visibility', np.ones((len(joints_2d), joints_2d.shape[1])))
    
    # Camera parameters
    fx = float(camera_data['fx'])
    fy = float(camera_data['fy'])
    cx = float(camera_data['cx'])
    cy = float(camera_data['cy'])
    img_w = int(camera_data['image_width'])
    img_h = int(camera_data['image_height'])
    
    # Full camera matrices
    K = camera_data['K']  # (3, 3)
    R = camera_data['R']  # (3, 3)
    T = camera_data['T']  # (3,)
    W2C = camera_data['W2C']  # (4, 4)
    
    N = len(global_orient)
    
    # Find image pattern
    image_patterns = [
        (os.path.join(image_dir, '{:08d}.png'), '.png'),
        (os.path.join(image_dir, '{:08d}.jpg'), '.jpg'),
        (os.path.join(image_dir, 'frame_{:08d}.png'), '.png'),
        (os.path.join(image_dir, 'frame_{:08d}.jpg'), '.jpg'),
    ]
    
    working_pattern = None
    for pattern, ext in image_patterns:
        if os.path.exists(pattern.format(0)):
            working_pattern = pattern
            break
    
    if working_pattern is None:
        return []
    
    for i in range(N):
        img_path_full = working_pattern.format(i)
        if not os.path.exists(img_path_full):
            continue
        
        # Relative path
        img_path_rel = os.path.relpath(img_path_full, image_base_for_path)
        
        # SMPL pose (72-dim)
        smpl_pose = smplx_to_smpl_pose(global_orient[i], body_pose[i])
        
        # Bbox
        center, scale = compute_bbox_from_joints(joints_2d[i], [img_h, img_w])
        
        # 2D keypoints with confidence (J, 3)
        kp2d = np.zeros((joints_2d.shape[1], 3), dtype=np.float32)
        kp2d[:, :2] = joints_2d[i]
        kp2d[:, 2] = visibility[i]
        
        # 3D keypoints with confidence (J, 4) - in camera coords
        kp3d = np.zeros((joints_3d_cam.shape[1], 4), dtype=np.float32)
        kp3d[:, :3] = joints_3d_cam[i]
        kp3d[:, 3] = visibility[i]
        
        # Root (pelvis) translation
        root_trans = joints_3d_cam[i, 0]
        
        annot = {
            # Image info
            'imgname': img_path_rel,
            'img_h': img_h,
            'img_w': img_w,
            
            # Bbox
            'center': center.astype(np.float32),
            'scale': float(scale),
            
            # SMPL
            'pose': smpl_pose.astype(np.float32),  # (72,)
            'shape': betas[:10].astype(np.float32),  # (10,)
            'trans': trans[i].astype(np.float32),  # (3,) - in camera coords
            'has_smpl': 1,
            
            # Keypoints
            'keypoints_2d': kp2d,  # (J, 3)
            'keypoints_3d': kp3d,  # (J, 4)
            
            # Camera intrinsics
            'cam_int': np.array([fx, fy, cx, cy], dtype=np.float32),
            'focal_length': float((fx + fy) / 2),
            'K': K.astype(np.float32),
            
            # Camera extrinsics
            'cam_ext': W2C.astype(np.float32),  # (4, 4)
            'R': R.astype(np.float32),  # (3, 3)
            'T': T.astype(np.float32),  # (3,)
            
            # Additional info for CameraHMR
            'root_trans': root_trans.astype(np.float32),  # pelvis in cam coords
            
            'frame_idx': i,
        }
        
        annotations.append(annot)
    
    return annotations


def consolidate_camerahmr_data(
    input_base,
    subject,
    output_dir,
    image_base_for_path=None,
    val_ratio=0.1,
    save_as='both',  # 'npz', 'pkl', or 'both'
):
    """Consolidate all data for CameraHMR."""
    
    output_dir = os.path.join(output_dir, subject)
    os.makedirs(output_dir, exist_ok=True)
    
    if image_base_for_path is None:
        image_base_for_path = input_base
    
    print(f"\n{'='*60}")
    print("Consolidating CameraHMR Training Data")
    print(f"{'='*60}")
    print(f"Input: {input_base}")
    print(f"Subject: {subject}")
    print(f"Output: {output_dir}")
    
    # Find all SMPL directories
    print("\n[1] Finding SMPL directories...")
    smpl_dirs = find_all_smpl_dirs(input_base, subject)
    print(f"    Found {len(smpl_dirs)} sequences")
    
    if len(smpl_dirs) == 0:
        print("No SMPL data found!")
        return
    
    # Process all sequences
    print("\n[2] Processing sequences...")
    all_annotations = []
    
    for item in tqdm(smpl_dirs, desc="Processing"):
        annots = process_single_sequence(
            smpl_dir=item['smpl_dir'],
            image_dir=item['image_dir'],
            image_base_for_path=image_base_for_path,
        )
        all_annotations.extend(annots)
    
    print(f"    Total samples: {len(all_annotations)}")
    
    if len(all_annotations) == 0:
        print("No annotations generated!")
        return
    
    # Shuffle and split
    print("\n[3] Splitting train/val...")
    np.random.seed(42)
    indices = np.random.permutation(len(all_annotations))
    
    val_size = int(len(all_annotations) * val_ratio)
    train_size = len(all_annotations) - val_size
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_annots = [all_annotations[i] for i in train_indices]
    val_annots = [all_annotations[i] for i in val_indices]
    
    print(f"    Train: {len(train_annots)}")
    print(f"    Val: {len(val_annots)}")
    
    # Convert to numpy arrays for NPZ
    def annots_to_arrays(annots):
        """Convert list of dicts to dict of arrays."""
        if not annots:
            return {}
        
        keys = annots[0].keys()
        result = {}
        
        for key in keys:
            values = [a[key] for a in annots]
            if isinstance(values[0], str):
                result[key] = np.array(values, dtype=object)
            elif isinstance(values[0], (int, float)):
                result[key] = np.array(values)
            elif isinstance(values[0], np.ndarray):
                result[key] = np.stack(values, axis=0)
            else:
                result[key] = np.array(values, dtype=object)
        
        return result
    
    # Save as NPZ
    if save_as in ['npz', 'both']:
        print("\n[4] Saving NPZ files...")
        
        train_arrays = annots_to_arrays(train_annots)
        val_arrays = annots_to_arrays(val_annots)
        all_arrays = annots_to_arrays(all_annotations)
        
        np.savez(os.path.join(output_dir, 'train.npz'), **train_arrays)
        np.savez(os.path.join(output_dir, 'val.npz'), **val_arrays)
        np.savez(os.path.join(output_dir, 'all.npz'), **all_arrays)
        
        print(f"    Saved: {output_dir}/train.npz")
        print(f"    Saved: {output_dir}/val.npz")
        print(f"    Saved: {output_dir}/all.npz")
    
    # Save as PKL (some HMR codebases prefer this)
    if save_as in ['pkl', 'both']:
        print("\n[5] Saving PKL files...")
        
        with open(os.path.join(output_dir, 'train.pkl'), 'wb') as f:
            pickle.dump(train_annots, f)
        with open(os.path.join(output_dir, 'val.pkl'), 'wb') as f:
            pickle.dump(val_annots, f)
        with open(os.path.join(output_dir, 'all.pkl'), 'wb') as f:
            pickle.dump(all_annotations, f)
        
        print(f"    Saved: {output_dir}/train.pkl")
        print(f"    Saved: {output_dir}/val.pkl")
        print(f"    Saved: {output_dir}/all.pkl")
    
    # Save image list (for easy loading)
    print("\n[6] Saving image lists...")
    
    with open(os.path.join(output_dir, 'train_images.txt'), 'w') as f:
        for a in train_annots:
            f.write(a['imgname'] + '\n')
    
    with open(os.path.join(output_dir, 'val_images.txt'), 'w') as f:
        for a in val_annots:
            f.write(a['imgname'] + '\n')
    
    print(f"    Saved: {output_dir}/train_images.txt")
    print(f"    Saved: {output_dir}/val_images.txt")
    
    # Save statistics
    print("\n[7] Saving statistics...")
    
    # Get sample annotation for format reference
    sample = all_annotations[0]
    
    stats = {
        'subject': subject,
        'total_samples': len(all_annotations),
        'train_samples': len(train_annots),
        'val_samples': len(val_annots),
        'num_sequences': len(smpl_dirs),
        'image_base': image_base_for_path,
        'data_format': {
            'imgname': 'str - relative image path',
            'img_h': 'int - image height',
            'img_w': 'int - image width',
            'center': '(2,) - bbox center [cx, cy]',
            'scale': 'float - bbox scale (size/200)',
            'pose': '(72,) - SMPL pose (axis-angle)',
            'shape': '(10,) - SMPL betas',
            'trans': '(3,) - translation in camera coords',
            'keypoints_2d': '(J, 3) - 2D joints [x, y, conf]',
            'keypoints_3d': '(J, 4) - 3D joints [x, y, z, conf] in camera coords',
            'cam_int': '(4,) - [fx, fy, cx, cy]',
            'focal_length': 'float - average focal length',
            'K': '(3, 3) - intrinsic matrix',
            'cam_ext': '(4, 4) - extrinsic matrix (W2C)',
            'R': '(3, 3) - rotation matrix',
            'T': '(3,) - translation vector',
            'root_trans': '(3,) - pelvis position in camera coords',
        },
        'sample_shapes': {
            k: list(v.shape) if isinstance(v, np.ndarray) else type(v).__name__
            for k, v in sample.items()
        }
    }
    
    stats_path = os.path.join(output_dir, 'stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"    Saved: {stats_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Subject: {subject}")
    print(f"Total samples: {len(all_annotations)}")
    print(f"Train: {len(train_annots)}")
    print(f"Val: {len(val_annots)}")
    print(f"\nOutput files:")
    print(f"  {output_dir}/")
    print(f"  ├── train.npz / train.pkl")
    print(f"  ├── val.npz / val.pkl")
    print(f"  ├── train_images.txt")
    print(f"  ├── val_images.txt")
    print(f"  └── stats.json")
    
    print("\n[Sample data shapes]")
    for k, v in sample.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: {v.shape} ({v.dtype})")
        else:
            print(f"  {k}: {type(v).__name__}")


def main():
    parser = ArgumentParser(description="Consolidate SMPL data for CameraHMR training")
    
    parser.add_argument('--input_base', type=str, required=True,
                        help='Base input directory')
    parser.add_argument('--subject', type=str, required=True,
                        help='Subject ID (e.g., 101010)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--image_base', type=str, default=None,
                        help='Base path for relative image paths')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Validation set ratio (default: 0.1)')
    parser.add_argument('--save_as', choices=['npz', 'pkl', 'both'], default='both',
                        help='Output format (default: both)')
    
    args = parser.parse_args()
    
    consolidate_camerahmr_data(
        input_base=args.input_base,
        subject=args.subject,
        output_dir=args.output_dir,
        image_base_for_path=args.image_base,
        val_ratio=args.val_ratio,
        save_as=args.save_as,
    )


if __name__ == '__main__':
    main()
