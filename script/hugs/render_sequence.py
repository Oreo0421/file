#!/usr/bin/env python3
"""

usage
python render_sequence.py \
  --human_pt_dir "transformed_humans/pt" \
  --scene_pt "output/gs_out/scene_gs_out.pt" \
  --output_dir "./rendered_sequence" \
  --start_frame 0 \
  --end_frame 99 \
  --camera_json "camera_params/lab/train/camera_params.json" \
  --render_mode human_scene

Batch render script to process 100 frames of human PT files with scene
Renders each human frame (00000000.pt to 00000099.pt) combined with scene.pt
Outputs rendered images for the entire sequence
"""

#!/usr/bin/env python3
"""
Simplified fixed version of render_sequence.py with memory management
Focuses only on fixing the CUDA out of memory issue
"""

#!/usr/bin/env python3
"""
Simplified fixed version of render_sequence.py with memory management
Focuses only on fixing the CUDA out of memory issue
"""

import math
import torch
import os
import numpy as np
import argparse
import json
import gc
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from loguru import logger

from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer
)


def load_gs_data(file_path):
    """Load Gaussian Splatting data from PT file"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"GS file not found: {file_path}")

    gs_data = torch.load(file_path, map_location='cuda')

    for key in gs_data:
        if isinstance(gs_data[key], torch.Tensor):
            gs_data[key] = gs_data[key].cuda()

    required_keys = ['xyz', 'scales', 'rotq', 'shs', 'opacity', 'active_sh_degree']
    for key in required_keys:
        if key not in gs_data:
            raise KeyError(f"Missing required key in GS data: {key}")

    return gs_data


def cleanup_tensors(*tensors):
    """Clean up tensors and free memory"""
    for tensor in tensors:
        if tensor is not None:
            del tensor
    torch.cuda.empty_cache()
    gc.collect()


def render(means3D, feats, opacity, scales, rotations, data, scaling_modifier=1.0, bg_color=None, active_sh_degree=0):
    """Core rendering function"""
    if bg_color is None:
        bg_color = torch.zeros(3, dtype=torch.float32, device="cuda")

    screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=False, device="cuda")
    means2D = screenspace_points

    tanfovx = math.tan(data['fovx'] * 0.5)
    tanfovy = math.tan(data['fovy'] * 0.5)

    shs, rgb = None, None
    if len(feats.shape) == 2:
        rgb = feats
    else:
        shs = feats

    raster_settings = GaussianRasterizationSettings(
        image_height=int(data['image_height']),
        image_width=int(data['image_width']),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=data['world_view_transform'],
        projmatrix=data['full_proj_transform'],
        sh_degree=active_sh_degree,
        campos=data['camera_center'],
        prefiltered=False,
        debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    with torch.no_grad():
        rendered_image, radii = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            colors_precomp=rgb,
        )

    rendered_image = torch.clamp(rendered_image, 0.0, 1.0)

    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
    }


def render_human_scene(data, human_gs_out, scene_gs_out, bg_color, scaling_modifier=1.0, render_mode='human_scene'):
    """Render human and scene together"""
    if render_mode == 'human_scene':
        feats = torch.cat([human_gs_out['shs'], scene_gs_out['shs']], dim=0)
        means3D = torch.cat([human_gs_out['xyz'], scene_gs_out['xyz']], dim=0)
        opacity = torch.cat([human_gs_out['opacity'], scene_gs_out['opacity']], dim=0)
        scales = torch.cat([human_gs_out['scales'], scene_gs_out['scales']], dim=0)
        rotations = torch.cat([human_gs_out['rotq'], scene_gs_out['rotq']], dim=0)
        active_sh_degree = human_gs_out['active_sh_degree']
    elif render_mode == 'human':
        feats = human_gs_out['shs']
        means3D = human_gs_out['xyz']
        opacity = human_gs_out['opacity']
        scales = human_gs_out['scales']
        rotations = human_gs_out['rotq']
        active_sh_degree = human_gs_out['active_sh_degree']
    elif render_mode == 'scene':
        feats = scene_gs_out['shs']
        means3D = scene_gs_out['xyz']
        opacity = scene_gs_out['opacity']
        scales = scene_gs_out['scales']
        rotations = scene_gs_out['rotq']
        active_sh_degree = scene_gs_out['active_sh_degree']

    render_pkg = render(
        means3D=means3D,
        feats=feats,
        opacity=opacity,
        scales=scales,
        rotations=rotations,
        data=data,
        scaling_modifier=scaling_modifier,
        bg_color=bg_color,
        active_sh_degree=active_sh_degree,
    )

    return render_pkg


def save_rendered_image(image_tensor, save_path, frame_idx=None):
    """Save rendered image to file"""
    if len(image_tensor.shape) == 3:
        image_np = image_tensor.permute(1, 2, 0).detach().cpu().numpy()
    else:
        image_np = image_tensor.detach().cpu().numpy()

    image_np = np.clip(image_np, 0.0, 1.0)
    image_uint8 = (image_np * 255).astype(np.uint8)

    pil_image = Image.fromarray(image_uint8)
    pil_image.save(save_path)

    if frame_idx is not None:
        logger.info(f"Frame {frame_idx:08d} rendered and saved")


def load_camera_data(camera_json_path, frame_idx=0):
    """Load camera data from JSON file"""
    with open(camera_json_path, 'r') as f:
        cam_j = json.load(f)

    i = min(frame_idx, len(cam_j['fovx']) - 1)

    data = {
        'fovx': float(cam_j['fovx'][i]),
        'fovy': float(cam_j['fovy'][i]),
        'image_width': int(cam_j['image_width'][i]),
        'image_height': int(cam_j['image_height'][i]),
        'world_view_transform': torch.tensor(
            cam_j['world_view_transform'][i],
            dtype=torch.float32, device='cuda'),
        'full_proj_transform': torch.tensor(
            cam_j['full_proj_transform'][i],
            dtype=torch.float32, device='cuda'),
        'camera_center': torch.tensor(
            cam_j['camera_center'][i],
            dtype=torch.float32, device='cuda'),
    }
    return data


def main():
    parser = argparse.ArgumentParser(description="Batch render human PT sequence with scene")
    parser.add_argument("--human_pt_dir", type=str, required=True)
    parser.add_argument("--scene_pt", type=str, required=True)
    parser.add_argument("--output_dir", "-o", type=str, required=True)
    parser.add_argument("--camera_json", type=str, required=True)
    parser.add_argument("--render_mode", type=str, default='human_scene',
                        choices=['human', 'scene', 'human_scene'])
    parser.add_argument("--bg_color", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=99)
    parser.add_argument("--scaling_modifier", type=float, default=1.0)

    args = parser.parse_args()

    # Setup paths
    human_pt_dir = Path(args.human_pt_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load scene data once
    logger.info(f"Loading scene data from: {args.scene_pt}")
    scene_gs_out = load_gs_data(args.scene_pt)
    logger.info(f"Scene loaded: {scene_gs_out['xyz'].shape[0]} Gaussians")

    bg_color = torch.tensor(args.bg_color, dtype=torch.float32, device='cuda')

    logger.info(f"Starting batch rendering frames {args.start_frame} to {args.end_frame}")

    successful_frames = 0
    failed_frames = []

    for frame_idx in tqdm(range(args.start_frame, args.end_frame + 1), desc="Rendering frames"):
        human_pt_file = human_pt_dir / f"{frame_idx:08d}.pt"

        if not human_pt_file.exists():
            logger.warning(f"Skipping missing file: {human_pt_file}")
            failed_frames.append(frame_idx)
            continue

        human_gs_out = None
        render_pkg = None
        camera_data = None

        try:
            # Load data for this frame
            if args.render_mode in ['human', 'human_scene']:
                human_gs_out = load_gs_data(str(human_pt_file))

            camera_data = load_camera_data(args.camera_json, frame_idx)

            # Render
            render_pkg = render_human_scene(
                data=camera_data,
                human_gs_out=human_gs_out,
                scene_gs_out=scene_gs_out,
                bg_color=bg_color,
                scaling_modifier=args.scaling_modifier,
                render_mode=args.render_mode
            )

            # Save image
            output_image = output_dir / f"frame_{frame_idx:08d}.png"
            save_rendered_image(render_pkg['render'], str(output_image), frame_idx)
            successful_frames += 1

        except Exception as e:
            logger.error(f"Error processing frame {frame_idx:08d}: {str(e)}")
            failed_frames.append(frame_idx)

        finally:
            # Clean up memory after each frame
            if human_gs_out is not None:
                for key in list(human_gs_out.keys()):
                    if isinstance(human_gs_out[key], torch.Tensor):
                        del human_gs_out[key]
                del human_gs_out

            if render_pkg is not None:
                cleanup_tensors(
                    render_pkg.get('render'),
                    render_pkg.get('viewspace_points'),
                    render_pkg.get('visibility_filter'),
                    render_pkg.get('radii')
                )
                del render_pkg

            if camera_data is not None:
                for key in list(camera_data.keys()):
                    if isinstance(camera_data[key], torch.Tensor):
                        del camera_data[key]
                del camera_data

            torch.cuda.empty_cache()
            gc.collect()

    # Final summary
    total_frames = args.end_frame - args.start_frame + 1
    logger.info(f"Completed! {successful_frames}/{total_frames} frames rendered successfully.")
    if failed_frames:
        logger.info(f"Failed frames: {failed_frames}")


if __name__ == "__main__":
    main()
