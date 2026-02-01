#!/usr/bin/env python3
"""
Batch process all cameras for a subject to generate SMPL parameters.

Directory structure expected:
    /mnt/data_hdd/fzhi/output/{subject}/{pose}/{scene}/{position}/
        ├── image/
        │   ├── top/
        │   ├── front/
        │   └── ...
        └── joint/
            └── npy/
                └── transformed/

Output structure:
    /mnt/data_hdd/fzhi/output/{subject}/{pose}/{scene}/{position}/
        └── smpl/
            ├── top/
            │   ├── {subject}_{pose}_smpl.npz
            │   ├── {subject}_{pose}_joints.npz
            │   └── ...
            ├── front/
            └── ...

Usage:
    # Process single subject, all poses, scenes, cameras
    python batch_process_smpl.py \
        --subject 101010 \
        --smpl_model_path /path/to/smpl_files \
        --amass_base_dir /path/to/AMASS \
        --camera_base_dir /path/to/cameras \
        --output_base_dir /mnt/data_hdd/fzhi/output

    # Process specific pose
    python batch_process_smpl.py \
        --subject 101010 \
        --pose brooming \
        --smpl_model_path /path/to/smpl_files \
        ...
"""

import os
import glob
import subprocess
from argparse import ArgumentParser
from tqdm import tqdm


def find_all_camera_dirs(base_path):
    """Find all camera directories under image/"""
    image_dir = os.path.join(base_path, 'image')
    if not os.path.exists(image_dir):
        return []
    
    cameras = []
    for item in os.listdir(image_dir):
        cam_path = os.path.join(image_dir, item)
        if os.path.isdir(cam_path):
            # Check if it has images
            images = glob.glob(os.path.join(cam_path, '*.png')) + \
                     glob.glob(os.path.join(cam_path, '*.jpg'))
            if images:
                cameras.append(item)
    
    return cameras


def find_amass_npz(amass_base_dir, pose_name):
    """
    Find AMASS npz file for a given pose name.
    Searches common AMASS directory structures.
    """
    # Common search patterns
    search_patterns = [
        f"**/{pose_name}.npz",
        f"**/{pose_name}_poses.npz",
        f"**/CMU/**/{pose_name}.npz",
        f"**/omilabpose/{pose_name}.npz",
    ]
    
    for pattern in search_patterns:
        matches = glob.glob(os.path.join(amass_base_dir, pattern), recursive=True)
        if matches:
            return matches[0]
    
    return None


def find_camera_json(camera_base_dir, scene, position, camera_name):
    """
    Find camera JSON file.
    Expected structure: camera_base_dir/{scene}/{position}/{camera_name}.json
    """
    possible_paths = [
        os.path.join(camera_base_dir, scene, position, f"{camera_name}.json"),
        os.path.join(camera_base_dir, f"{scene}_{position}", f"{camera_name}.json"),
        os.path.join(camera_base_dir, scene, f"{camera_name}.json"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None


def get_scene_key(scene, position):
    """Generate scene transform key, e.g., 'djr_p1'"""
    return f"{scene}_{position}"


def process_single_camera(
    amass_npz,
    joints_dir,
    camera_json,
    output_dir,
    smpl_model_path,
    scene_key,
    subject_name,
    pose_name,
    script_path,
    dry_run=False
):
    """Run amass_to_hybrik_with_rendered_joints.py for a single camera."""
    
    cmd = [
        'python', script_path,
        '--amass_npz', amass_npz,
        '--rendered_joints_dir', joints_dir,
        '--camera_json', camera_json,
        '--output_dir', output_dir,
        '--smpl_model_path', smpl_model_path,
        '--scene', scene_key,
        '--subject_name', subject_name,
        '--pose_name', pose_name,
    ]
    
    if dry_run:
        print(f"[DRY RUN] Would run:")
        print(f"  {' '.join(cmd)}")
        return True
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error processing {output_dir}:")
            print(result.stderr)
            return False
        return True
    except Exception as e:
        print(f"Exception: {e}")
        return False


def batch_process_subject(
    subject,
    output_base_dir,
    smpl_model_path,
    amass_base_dir,
    camera_base_dir,
    script_path,
    pose_filter=None,
    scene_filter=None,
    camera_filter=None,
    dry_run=False
):
    """
    Process all poses, scenes, positions, and cameras for a subject.
    """
    subject_dir = os.path.join(output_base_dir, subject)
    
    if not os.path.exists(subject_dir):
        print(f"Subject directory not found: {subject_dir}")
        return
    
    # Find all pose directories
    poses = [d for d in os.listdir(subject_dir) 
             if os.path.isdir(os.path.join(subject_dir, d))]
    
    if pose_filter:
        poses = [p for p in poses if pose_filter in p]
    
    print(f"\n{'='*60}")
    print(f"Subject: {subject}")
    print(f"Found {len(poses)} poses: {poses}")
    print(f"{'='*60}")
    
    total_processed = 0
    total_failed = 0
    
    for pose in poses:
        pose_dir = os.path.join(subject_dir, pose)
        
        # Find AMASS npz for this pose
        amass_npz = find_amass_npz(amass_base_dir, pose)
        if amass_npz is None:
            print(f"\n⚠️  AMASS npz not found for pose: {pose}")
            print(f"   Searched in: {amass_base_dir}")
            continue
        
        print(f"\n📁 Pose: {pose}")
        print(f"   AMASS: {amass_npz}")
        
        # Find all scene directories (e.g., djr, room, garden)
        scenes = [d for d in os.listdir(pose_dir) 
                  if os.path.isdir(os.path.join(pose_dir, d))]
        
        if scene_filter:
            scenes = [s for s in scenes if scene_filter in s]
        
        for scene in scenes:
            scene_dir = os.path.join(pose_dir, scene)
            
            # Find all position directories (e.g., p1, p2)
            positions = [d for d in os.listdir(scene_dir) 
                        if os.path.isdir(os.path.join(scene_dir, d))]
            
            for position in positions:
                pos_dir = os.path.join(scene_dir, position)
                
                # Check for joints directory
                joints_dir = os.path.join(pos_dir, 'joint', 'npy', 'transformed')
                if not os.path.exists(joints_dir):
                    print(f"   ⚠️  No joints found: {joints_dir}")
                    continue
                
                # Find all cameras
                cameras = find_all_camera_dirs(pos_dir)
                
                if camera_filter:
                    cameras = [c for c in cameras if camera_filter in c]
                
                if not cameras:
                    print(f"   ⚠️  No cameras found in: {pos_dir}/image/")
                    continue
                
                scene_key = get_scene_key(scene, position)
                print(f"   🎬 Scene: {scene}/{position} ({scene_key})")
                print(f"      Cameras: {cameras}")
                
                for camera in cameras:
                    # Find camera JSON
                    camera_json = find_camera_json(
                        camera_base_dir, scene, position, camera
                    )
                    
                    if camera_json is None:
                        print(f"      ⚠️  Camera JSON not found: {scene}/{position}/{camera}.json")
                        continue
                    
                    # Output directory
                    output_dir = os.path.join(pos_dir, 'smpl', camera)
                    
                    print(f"      📷 {camera}: {output_dir}")
                    
                    success = process_single_camera(
                        amass_npz=amass_npz,
                        joints_dir=joints_dir,
                        camera_json=camera_json,
                        output_dir=output_dir,
                        smpl_model_path=smpl_model_path,
                        scene_key=scene_key,
                        subject_name=subject,
                        pose_name=pose,
                        script_path=script_path,
                        dry_run=dry_run
                    )
                    
                    if success:
                        total_processed += 1
                        print(f"         ✅ Done")
                    else:
                        total_failed += 1
                        print(f"         ❌ Failed")
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Processed: {total_processed}")
    print(f"Failed: {total_failed}")


def main():
    parser = ArgumentParser(description="Batch process SMPL for all cameras")
    
    parser.add_argument('--subject', type=str, required=True,
                        help='Subject ID (e.g., 101010)')
    parser.add_argument('--smpl_model_path', type=str, required=True,
                        help='Path to SMPL model files')
    parser.add_argument('--amass_base_dir', type=str, required=True,
                        help='Base directory for AMASS npz files')
    parser.add_argument('--camera_base_dir', type=str, required=True,
                        help='Base directory for camera JSON files')
    parser.add_argument('--output_base_dir', type=str, 
                        default='/mnt/data_hdd/fzhi/output',
                        help='Base directory for output')
    parser.add_argument('--script_path', type=str,
                        default='amass_to_hybrik_with_rendered_joints.py',
                        help='Path to conversion script')
    
    # Filters
    parser.add_argument('--pose', type=str, default=None,
                        help='Filter by pose name')
    parser.add_argument('--scene', type=str, default=None,
                        help='Filter by scene name')
    parser.add_argument('--camera', type=str, default=None,
                        help='Filter by camera name')
    
    parser.add_argument('--dry_run', action='store_true',
                        help='Show what would be done without executing')
    
    args = parser.parse_args()
    
    batch_process_subject(
        subject=args.subject,
        output_base_dir=args.output_base_dir,
        smpl_model_path=args.smpl_model_path,
        amass_base_dir=args.amass_base_dir,
        camera_base_dir=args.camera_base_dir,
        script_path=args.script_path,
        pose_filter=args.pose,
        scene_filter=args.scene,
        camera_filter=args.camera,
        dry_run=args.dry_run
    )


if __name__ == '__main__':
    main()
