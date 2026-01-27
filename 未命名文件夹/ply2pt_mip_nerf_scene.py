import sys
import pathlib
import os
import torch
import numpy as np
import argparse
from plyfile import PlyData

parser = argparse.ArgumentParser()
parser.add_argument("--in_ply", required=True)
parser.add_argument("--out_pt", required=True)
parser.add_argument("--sh_degree", type=int, default=3)
args = parser.parse_args()

PLY_PATH = args.in_ply
OUT_PT_PATH = args.out_pt
SH_DEGREE = args.sh_degree


root = pathlib.Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

def load_scene_gaussians(ply_path: str, sh_degree: int):
    from hugs.models import SceneGS

    # Read PLY data and manually extract fields to avoid stride issues
    print("Loading PLY and extracting fields manually...")
    plydata = PlyData.read(ply_path)
    vertex_data = plydata['vertex'].data

    print(f"Loaded {len(vertex_data)} vertices")
    print(f"Original opacity strides: {vertex_data['opacity'].strides}")

    # Extract each field as a separate contiguous array
    print("Extracting individual fields...")

    xyz = np.column_stack([
        vertex_data['x'].astype(np.float32, copy=True),
        vertex_data['y'].astype(np.float32, copy=True),
        vertex_data['z'].astype(np.float32, copy=True)
    ])

    normals = np.column_stack([
        vertex_data['nx'].astype(np.float32, copy=True),
        vertex_data['ny'].astype(np.float32, copy=True),
        vertex_data['nz'].astype(np.float32, copy=True)
    ])

    # Extract SH coefficients
    sh_coeffs = []
    # DC components
    sh_coeffs.append(vertex_data['f_dc_0'].astype(np.float32, copy=True))
    sh_coeffs.append(vertex_data['f_dc_1'].astype(np.float32, copy=True))
    sh_coeffs.append(vertex_data['f_dc_2'].astype(np.float32, copy=True))

    # Rest components
    for i in range(45):  # 45 rest coefficients for degree 3
        field_name = f'f_rest_{i}'
        if field_name in vertex_data.dtype.names:
            sh_coeffs.append(vertex_data[field_name].astype(np.float32, copy=True))

    sh_features = np.column_stack(sh_coeffs)

    opacities = vertex_data['opacity'].astype(np.float32, copy=True)

    scales = np.column_stack([
        vertex_data['scale_0'].astype(np.float32, copy=True),
        vertex_data['scale_1'].astype(np.float32, copy=True),
        vertex_data['scale_2'].astype(np.float32, copy=True)
    ])

    rotations = np.column_stack([
        vertex_data['rot_0'].astype(np.float32, copy=True),
        vertex_data['rot_1'].astype(np.float32, copy=True),
        vertex_data['rot_2'].astype(np.float32, copy=True),
        vertex_data['rot_3'].astype(np.float32, copy=True)
    ])

    colors = np.column_stack([
        vertex_data['red'].astype(np.uint8, copy=True),
        vertex_data['green'].astype(np.uint8, copy=True),
        vertex_data['blue'].astype(np.uint8, copy=True)
    ]) if all(f in vertex_data.dtype.names for f in ['red', 'green', 'blue']) else None

    print(f"✓ Extracted arrays with proper strides:")
    print(f"  xyz: {xyz.shape}, strides: {xyz.strides}")
    print(f"  opacities: {opacities.shape}, strides: {opacities.strides}")
    print(f"  scales: {scales.shape}, strides: {scales.strides}")
    print(f"  rotations: {rotations.shape}, strides: {rotations.strides}")

    # Create SceneGS and set parameters directly
    scene_gs = SceneGS(sh_degree=sh_degree)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Convert to PyTorch tensors (should work now with proper arrays)
    scene_gs._xyz = torch.nn.Parameter(
        torch.from_numpy(xyz).to(device).requires_grad_(True)
    )

    scene_gs._opacity = torch.nn.Parameter(
        torch.from_numpy(opacities).unsqueeze(1).to(device).requires_grad_(True)
    )

    scene_gs._scaling = torch.nn.Parameter(
        torch.from_numpy(scales).to(device).requires_grad_(True)
    )

    scene_gs._rotation = torch.nn.Parameter(
        torch.from_numpy(rotations).to(device).requires_grad_(True)
    )

    # Set SH features
    n_points = xyz.shape[0]
    shs_dc = sh_features[:, :3].reshape(n_points, 1, 3)
    shs_rest = sh_features[:, 3:].reshape(n_points, -1, 3) if sh_features.shape[1] > 3 else np.zeros((n_points, 0, 3))

    scene_gs._features_dc = torch.nn.Parameter(
        torch.from_numpy(shs_dc).to(device).requires_grad_(True)
    )

    scene_gs._features_rest = torch.nn.Parameter(
        torch.from_numpy(shs_rest).to(device).requires_grad_(True)
    )

    scene_gs.active_sh_degree = sh_degree

    result = scene_gs.forward()
    print("✓ Successfully created Gaussians")
    return result

def main():
    if not os.path.isfile(PLY_PATH):
        raise FileNotFoundError(f"Could not find PLY file at: {PLY_PATH!r}")

    try:
        scene_gs_out = load_scene_gaussians(PLY_PATH, SH_DEGREE)
        torch.save(scene_gs_out, OUT_PT_PATH)
        print("✓ Saved dictionary to {OUT_PT_PATH}")

        print("\nExtracted scene Gaussian-value dictionary:")
        for k, v in scene_gs_out.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k:15s}: shape={tuple(v.shape)}, dtype={v.dtype}")
            else:
                print(f"  {k:15s}: {v!r}")

    except Exception as e:
        print(f"Error: {e}")
        print("\nDebugging PLY file structure...")

        # Debug info
        plydata = PlyData.read(PLY_PATH)
        vertex_data = plydata['vertex'].data
        print(f"Vertex count: {len(vertex_data)}")
        print(f"Properties: {list(vertex_data.dtype.names)}")
        print(f"Data contiguous: {vertex_data.flags.c_contiguous}")
        print(f"Opacity strides: {vertex_data['opacity'].strides}")

        raise

if __name__ == "__main__":
    main()
