#!/usr/bin/env python3
"""
Script to apply transformation matrix to a sequence of human PLY files
This aligns all frames with the scene using the transformation from the first aligned frame
"""
"""
run command

 python scripts/transform_human_sequence.py   --input_dir "/home/zhiyw/Desktop/ml-hugs/transformed_humans/hugs/ply"   --output_dir "/home/zhiyw/Desktop/ml-hugs/transformed_humans/mip_nerf_360"

"""
import torch
import numpy as np
import argparse
from pathlib import Path
from plyfile import PlyData, PlyElement
from loguru import logger
import os
from tqdm import tqdm


def load_ply_gaussians(ply_path):
    """Load Gaussian data from PLY file"""
    logger.info(f"Loading PLY file: {ply_path}")
    plydata = PlyData.read(ply_path)
    
    # Extract XYZ coordinates
    xyz = np.stack((
        np.asarray(plydata.elements[0]["x"]),
        np.asarray(plydata.elements[0]["y"]),
        np.asarray(plydata.elements[0]["z"])
    ), axis=1)
    
    # Extract opacity
    opacity = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
    
    # Extract DC features (RGB)
    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
    
    # Extract REST features (higher order SH)
    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
    
    if len(extra_f_names) > 0:
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        # Calculate SH degree from number of features
        total_coeffs = 3 + len(extra_f_names)
        sh_coeffs_per_channel = total_coeffs // 3
        sh_degree = int(np.sqrt(sh_coeffs_per_channel)) - 1
        
        # Reshape features_extra to (P, 3, SH_coeffs - 1)
        features_extra = features_extra.reshape((xyz.shape[0], 3, sh_coeffs_per_channel - 1))
        
        # Combine DC and REST features: (P, 3, SH_coeffs)
        shs = np.concatenate([features_dc, features_extra], axis=2)
    else:
        # Only DC features
        shs = features_dc
        sh_degree = 0
    
    # Transpose to get (P, SH_coeffs, 3) format
    shs = np.transpose(shs, (0, 2, 1))
    
    # Extract scales
    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])
    
    # Apply exponential activation to scales (PLY stores log scales)
    scales = np.exp(scales)
    
    # Extract rotations (quaternions)
    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
    rotations = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rotations[:, idx] = np.asarray(plydata.elements[0][attr_name])
    
    # Normalize quaternions
    rotations = rotations / np.linalg.norm(rotations, axis=1, keepdims=True)
    
    # Apply sigmoid activation to opacity (PLY stores logit opacity)
    opacity = 1.0 / (1.0 + np.exp(-opacity))
    
    return {
        'xyz': xyz,
        'scales': scales,
        'rotq': rotations,
        'shs': shs,
        'opacity': opacity,
        'active_sh_degree': sh_degree
    }


def apply_transformation_to_gaussians(gaussians, transform_matrix):
    """Apply 4x4 transformation matrix to Gaussian positions and rotations"""
    xyz = gaussians['xyz']
    rotq = gaussians['rotq']
    
    # Transform positions
    # Convert to homogeneous coordinates
    xyz_homo = np.hstack([xyz, np.ones((xyz.shape[0], 1))])
    
    # Apply transformation
    xyz_transformed = (transform_matrix @ xyz_homo.T).T
    
    # Convert back to 3D coordinates
    xyz_new = xyz_transformed[:, :3]
    
    # Transform rotations
    # Extract rotation part of transformation matrix
    rotation_matrix = transform_matrix[:3, :3]
    
    # Convert quaternions to rotation matrices
    def quat_to_rotation_matrix(q):
        """Convert quaternion to rotation matrix"""
        w, x, y, z = q
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ])
    
    def rotation_matrix_to_quat(R):
        """Convert rotation matrix to quaternion"""
        trace = np.trace(R)
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s
            z = (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        return np.array([w, x, y, z])
    
    # Transform each quaternion
    rotq_new = np.zeros_like(rotq)
    for i in range(rotq.shape[0]):
        # Convert quaternion to rotation matrix
        orig_rot_mat = quat_to_rotation_matrix(rotq[i])
        
        # Apply transformation
        new_rot_mat = rotation_matrix @ orig_rot_mat
        
        # Convert back to quaternion
        rotq_new[i] = rotation_matrix_to_quat(new_rot_mat)
    
    # Create new gaussian dictionary with transformed values
    gaussians_new = gaussians.copy()
    gaussians_new['xyz'] = xyz_new
    gaussians_new['rotq'] = rotq_new
    
    return gaussians_new


def save_gaussians_to_ply(gaussians, output_path):
    """Save Gaussian data to PLY file"""
    xyz = gaussians['xyz']
    scales = gaussians['scales']
    rotq = gaussians['rotq']
    shs = gaussians['shs']
    opacity = gaussians['opacity']
    
    # Convert scales back to log space for PLY
    scales_log = np.log(scales)
    
    # Convert opacity back to logit space for PLY
    opacity_clipped = np.clip(opacity, 1e-6, 1 - 1e-6)  # Avoid log(0)
    opacity_logit = np.log(opacity_clipped / (1 - opacity_clipped))
    
    # Prepare SH features
    shs_transposed = np.transpose(shs, (0, 2, 1))  # (P, 3, SH_coeffs)
    features_dc = shs_transposed[:, :, 0]  # (P, 3) - DC components
    
    # Prepare vertex data
    vertex_data = []
    for i in range(xyz.shape[0]):
        vertex = [
            xyz[i, 0], xyz[i, 1], xyz[i, 2],  # x, y, z
            rotq[i, 0], rotq[i, 1], rotq[i, 2], rotq[i, 3],  # rot_0, rot_1, rot_2, rot_3
            scales_log[i, 0], scales_log[i, 1], scales_log[i, 2],  # scale_0, scale_1, scale_2
            opacity_logit[i, 0],  # opacity
            features_dc[i, 0], features_dc[i, 1], features_dc[i, 2]  # f_dc_0, f_dc_1, f_dc_2
        ]
        
        # Add higher order SH features if they exist
        if shs_transposed.shape[2] > 1:
            for sh_idx in range(1, shs_transposed.shape[2]):
                for color_idx in range(3):
                    vertex.append(shs_transposed[i, color_idx, sh_idx])
        
        vertex_data.append(tuple(vertex))
    
    # Create property list
    properties = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('opacity', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4')
    ]
    
    # Add higher order SH properties if they exist
    if shs_transposed.shape[2] > 1:
        rest_count = (shs_transposed.shape[2] - 1) * 3
        for i in range(rest_count):
            properties.append((f'f_rest_{i}', 'f4'))
    
    # Create PLY element
    vertex_element = PlyElement.describe(np.array(vertex_data, dtype=properties), 'vertex')
    
    # Write PLY file
    PlyData([vertex_element], text=False).write(output_path)
    logger.info(f"Saved transformed PLY to: {output_path}")


def save_gaussians_to_pt(gaussians, output_path):
    """Save Gaussian data to PT file"""
    gs_data = {
        'xyz': torch.tensor(gaussians['xyz'], dtype=torch.float32),
        'scales': torch.tensor(gaussians['scales'], dtype=torch.float32),
        'rotq': torch.tensor(gaussians['rotq'], dtype=torch.float32),
        'shs': torch.tensor(gaussians['shs'], dtype=torch.float32),
        'opacity': torch.tensor(gaussians['opacity'], dtype=torch.float32),
        'active_sh_degree': gaussians['active_sh_degree']
    }
    
    torch.save(gs_data, output_path)
    logger.info(f"Saved transformed PT to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Apply transformation to human PLY sequence")
    parser.add_argument("--input_dir", "-i", type=str, required=True,
                        help="Directory containing PLY files (00000000.ply to 00000099.ply)")
    parser.add_argument("--output_dir", "-o", type=str, required=True,
                        help="Output directory for transformed PLY files")
    parser.add_argument("--output_format", choices=['ply', 'pt', 'both'], default='both',
                        help="Output format: ply, pt, or both")
    parser.add_argument("--start_frame", type=int, default=0,
                        help="Start frame number (default: 0)")
    parser.add_argument("--end_frame", type=int, default=99,
                        help="End frame number (default: 99)")
    
    args = parser.parse_args()

     # First transform (the one you used earlier on the human position)
    transform = np.array([
        [1.417337, -3.719793,  5.757977, 20.600479],
        [-0.786601, -5.929179, -3.636770,  2.822414],
        [6.809730,  0.089329, -1.618520, 25.250027],
        [0.000000,  0.000000,  0.000000,  1.000000]
    ])

    # Define the transformation matrix from your aligned first frame
    transform_matrix_scene = np.array([
        [0.004506839905, -0.124592848122, 0.083404511213, -3.700955867767],
        [0.149711236358, 0.008269036189, 0.004262818955, -2.735711812973],
        [-0.008138610050, 0.083115860820, 0.124601446092, -4.244910240173],
        [0.000000, 0.000000, 0.000000, 1.000000]
    ])
    # Apply both transforms in sequence
    transform_matrix = transform_matrix_scene @ transform

    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for different formats
    if args.output_format in ['ply', 'both']:
        (output_dir / 'ply').mkdir(exist_ok=True)
    if args.output_format in ['pt', 'both']:
        (output_dir / 'pt').mkdir(exist_ok=True)
    
    input_dir = Path(args.input_dir)
    
    logger.info(f"Processing frames {args.start_frame} to {args.end_frame}")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Output format: {args.output_format}")
    
    # Process each frame
    for frame_idx in tqdm(range(args.start_frame, args.end_frame + 1), desc="Processing frames"):
        # Input file path
        input_file = input_dir / f"{frame_idx:08d}.ply"
        
        if not input_file.exists():
            logger.warning(f"Skipping missing file: {input_file}")
            continue
        
        try:
            # Load PLY file
            gaussians = load_ply_gaussians(str(input_file))
            
            # Apply transformation
            gaussians_transformed = apply_transformation_to_gaussians(gaussians, transform_matrix)
            
            # Save in requested format(s)
            if args.output_format in ['ply', 'both']:
                output_ply = output_dir / 'ply' / f"{frame_idx:08d}.ply"
                save_gaussians_to_ply(gaussians_transformed, str(output_ply))
            
            if args.output_format in ['pt', 'both']:
                output_pt = output_dir / 'pt' / f"{frame_idx:08d}.pt"
                save_gaussians_to_pt(gaussians_transformed, str(output_pt))
            
            logger.info(f" Processed frame {frame_idx:08d}")
            
        except Exception as e:
            logger.error(f" Error processing frame {frame_idx:08d}: {str(e)}")
            continue
    
    logger.info(" All frames processed successfully!")
    
    # Print summary
    print("\n" + "="*60)
    print("TRANSFORMATION SUMMARY")
    print("="*60)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Frames processed: {args.start_frame} to {args.end_frame}")
    print(f"Output format: {args.output_format}")
    print(f"Transformation matrix applied:")
    print(transform_matrix)
    print("="*60)


if __name__ == "__main__":
    main()
