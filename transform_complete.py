#!/usr/bin/env python3
"""
Script to apply transformation matrix to a sequence of human PLY files
This aligns all frames with the scene using the transformation from the first aligned frame
"""
"""
run command

 python scripts/transform_human_sequence.py   --input_dir "/home...ir "/home/zhiyw/Desktop/ml-hugs/transformed_humans/mip_nerf_360"

"""
import torch
import numpy as np
import argparse
from pathlib import Path
from plyfile import PlyData, PlyElement
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    opacity = np.asarray(plydata.elements[0]["opacity"]).reshape(-1, 1)

    # Extract rotation quaternions
    rotq = np.stack((
        np.asarray(plydata.elements[0]["rot_0"]),
        np.asarray(plydata.elements[0]["rot_1"]),
        np.asarray(plydata.elements[0]["rot_2"]),
        np.asarray(plydata.elements[0]["rot_3"])
    ), axis=1)

    # Extract scales (log-space)
    scales = np.stack((
        np.asarray(plydata.elements[0]["scale_0"]),
        np.asarray(plydata.elements[0]["scale_1"]),
        np.asarray(plydata.elements[0]["scale_2"])
    ), axis=1)

    # Extract SH features
    # DC components
    f_dc = np.stack((
        np.asarray(plydata.elements[0]["f_dc_0"]),
        np.asarray(plydata.elements[0]["f_dc_1"]),
        np.asarray(plydata.elements[0]["f_dc_2"])
    ), axis=1)

    # Check for higher order SH components
    f_rest = []
    sh_degree = 0

    # Determine maximum SH degree present from property names
    num_properties = len(plydata.elements[0].data.dtype.names)
    expected_base_props = 3 + 4 + 3 + 1 + 3  # xyz, rot, scale, opacity, f_dc
    num_sh_rest_properties = num_properties - expected_base_props

    if num_sh_rest_properties > 0:
        # SH coefficients are stored as f_rest_0, f_rest_1, ..., f_rest_N
        # Each coefficient has 3 channels (RGB)
        # Find how many SH coefficients we have per channel
        sh_rest_props = [name for name in plydata.elements[0].data.dtype.names
                         if name.startswith('f_rest_')]
        sh_rest_props_sorted = sorted(
            sh_rest_props,
            key=lambda x: int(x.split('_')[-1])
        )

        for prop_name in sh_rest_props_sorted:
            f_rest.append(np.asarray(plydata.elements[0][prop_name]))

        if f_rest:
            f_rest = np.stack(f_rest, axis=1)  # (P, num_rest)
            # Determine SH degree from number of coefficients
            # Total SH coefficients per color = 1 (DC) + num_rest/3
            total_coeffs = 1 + f_rest.shape[1] // 3
            # Find smallest n such that (n+1)^2 = total_coeffs
            # because number of SH basis functions up to degree n is (n+1)^2
            for n in range(10):
                if (n + 1) ** 2 == total_coeffs:
                    sh_degree = n
                    break
            else:
                logger.warning(f"Could not determine SH degree from {total_coeffs} coefficients, defaulting to 0")
                sh_degree = 0

    # Combine SH coefficients into the format used in training
    # Format: (P, 3, (degree+1)^2)
    if sh_degree > 0:
        # We have higher order SH coefficients
        num_coeffs = (sh_degree + 1) ** 2
        shs = np.zeros((xyz.shape[0], 3, num_coeffs), dtype=np.float32)

        # DC components go first
        shs[:, :, 0] = f_dc

        # Remaining coefficients are stored in f_rest (P, num_rest)
        # Each coefficient has 3 channels (RGB)
        num_rest = num_coeffs - 1
        assert f_rest.shape[1] == num_rest * 3, \
            f"Expected {num_rest * 3} f_rest components, got {f_rest.shape[1]}"

        for i in range(num_rest):
            for c in range(3):  # RGB channels
                shs[:, c, i + 1] = f_rest[:, i * 3 + c]
    else:
        # Only DC components
        shs = np.zeros((xyz.shape[0], 3, 1), dtype=np.float32)
        shs[:, :, 0] = f_dc

    logger.info(f"Loaded {xyz.shape[0]} Gaussians")

    return {
        'xyz': xyz,
        'scales': scales,
        'rotq': rotq,
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
    # Extract 3x3 rotation matrix from transform_matrix
    rotation_matrix = transform_matrix[:3, :3]

    # Define helper functions for quaternion to rotation matrix and back
    def quat_to_rotation_matrix(q):
        """Convert quaternion to rotation matrix"""
        w, x, y, z = q
        R = np.array([
            [1 - 2*(y**2 + z**2),     2*(x*y - z*w),         2*(x*z + y*w)],
            [2*(x*y + z*w),           1 - 2*(x**2 + z**2),   2*(y*z - x*w)],
            [2*(x*z - y*w),           2*(y*z + x*w),         1 - 2*(x**2 + y**2)]
        ])
        return R

    def rotation_matrix_to_quat(R):
        """Convert rotation matrix to quaternion"""
        m00, m01, m02 = R[0]
        m10, m11, m12 = R[1]
        m20, m21, m22 = R[2]

        tr = m00 + m11 + m22

        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2
            qw = 0.25 * S
            qx = (m21 - m12) / S
            qy = (m02 - m20) / S
            qz = (m10 - m01) / S
        elif (m00 > m11) and (m00 > m22):
            S = np.sqrt(1.0 + m00 - m11 - m22) * 2
            qw = (m21 - m12) / S
            qx = 0.25 * S
            qy = (m01 + m10) / S
            qz = (m02 + m20) / S
        elif m11 > m22:
            S = np.sqrt(1.0 + m11 - m00 - m22) * 2
            qw = (m02 - m20) / S
            qx = (m01 + m10) / S
            qy = 0.25 * S
            qz = (m12 + m21) / S
        else:
            S = np.sqrt(1.0 + m22 - m00 - m11) * 2
            qw = (m10 - m01) / S
            qx = (m02 + m20) / S
            qy = (m12 + m21) / S
            qz = 0.25 * S

        return np.array([qw, qx, qy, qz])

    # Transform each quaternion
    rotq_new = np.zeros_like(rotq)
    for i in range(rotq.shape[0]):
        # Convert quaternion to rotation matrix
        orig_rot_mat = quat_to_rotation_matrix(rotq[i])

        # Apply transformation
        new_rot_mat = rotation_matrix @ orig_rot_mat

        # Convert back to quaternion
        rotq_new[i] = rotation_matrix_to_quat(new_rot_mat)

    # Update gaussians
    gaussians['xyz'] = xyz_new
    gaussians['rotq'] = rotq_new

    logger.info("Applied transformation to Gaussians")

    return gaussians


def save_gaussians_to_ply(gaussians, output_path):
    """Save transformed Gaussians back to PLY format"""
    logger.info(f"Saving transformed PLY to: {output_path}")

    xyz = gaussians['xyz']
    scales = gaussians['scales']
    rotq = gaussians['rotq']
    shs = gaussians['shs']
    opacity = gaussians['opacity']
    sh_degree = gaussians['active_sh_degree']

    # Convert scales to log-space if they aren't already
    # Assuming they are in log-space already, as in the original format
    scales_log = scales

    # Convert opacity from logit to (0,1) if needed
    # Here we assume opacity is stored as logit (as in original format)
    opacity_logit = opacity

    # SH features: convert from (P, 3, (degree+1)^2) to DC + rest
    shs_transposed = shs.transpose(0, 2, 1)  # (P, (degree+1)^2, 3)
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

    # Add SH rest properties if needed
    if sh_degree > 0:
        num_coeffs = (sh_degree + 1) ** 2
        num_rest = num_coeffs - 1
        for i in range(num_rest * 3):
            properties.append((f'f_rest_{i}', 'f4'))

    # Define the PLY dtype
    dtype = np.dtype(properties)

    # Create numpy array
    vertex_array = np.array(vertex_data, dtype=dtype)

    # Create PlyElement
    el = PlyElement.describe(vertex_array, 'vertex')

    # Write to file
    PlyData([el]).write(output_path)
    logger.info("PLY file saved successfully")


def save_gaussians_to_pt(gaussians, output_path):
    """Save Gaussians to PT format compatible with training"""
    logger.info(f"Saving transformed PT to: {output_path}")

    # Convert numpy arrays to tensors
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

    # Second transform (existing transform_matrix in your script)
    transform_matrix = np.array([
        [0.004506839905, -0.124592848122, 0.083404511213, -3.700955867767],
        [0.149711236358,  0.008269036189, 0.004262818955, -2.735711812973],
        [-0.008138610050, 0.083115860820, 0.124601446092, -4.244910240173],
        [0.000000,        0.000000,        0.000000,       1.000000]
    ])

    # Apply `transform` first, then `transform_matrix`
    # final_transform = transform_matrix ∘ transform
    final_transform = transform_matrix @ transform

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
    logger.info(f"First transform (transform):\n{transform}")
    logger.info(f"Second transform (transform_matrix):\n{transform_matrix}")
    logger.info(f"Final combined transform (transform_matrix @ transform):\n{final_transform}")

    # Process each frame
    for frame_idx in tqdm(range(args.start_frame, args.end_frame + 1), desc="Processing frames"):
        input_file = input_dir / f"{frame_idx:08d}.ply"

        if not input_file.exists():
            logger.warning(f"Skipping missing file: {input_file}")
            continue

        try:
            # Load PLY file
            gaussians = load_ply_gaussians(str(input_file))

            # Apply combined transformation (transform THEN transform_matrix)
            gaussians_transformed = apply_transformation_to_gaussians(gaussians, final_transform)

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
    print("First transform (transform):")
    print(transform)
    print("Second transform (transform_matrix):")
    print(transform_matrix)
    print("Final combined transform (transform_matrix @ transform):")
    print(final_transform)
    print("="*60)



if __name__ == "__main__":
    main()
