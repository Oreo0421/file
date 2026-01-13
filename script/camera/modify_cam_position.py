#!/usr/bin/env python3
"""
Modify camera positions by moving them in 4 directions (front, back, left, right)
Each direction moves 0.5 meters from original position
"""

import argparse
import numpy as np
from scipy.spatial.transform import Rotation


def parse_images_file(filepath):
    """Parse COLMAP images.txt file"""
    images = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Store header
    header_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith('#') or line.strip() == '':
            header_lines.append(line)
            i += 1
            continue
        
        # Parse image data
        parts = line.strip().split()
        if len(parts) >= 10:
            image_data = {
                'image_id': int(parts[0]),
                'qw': float(parts[1]),
                'qx': float(parts[2]),
                'qy': float(parts[3]),
                'qz': float(parts[4]),
                'tx': float(parts[5]),
                'ty': float(parts[6]),
                'tz': float(parts[7]),
                'camera_id': int(parts[8]),
                'name': parts[9],
                'line': line
            }
            
            # Next line (2D points or empty)
            i += 1
            points_line = lines[i] if i < len(lines) else '\n'
            image_data['points_line'] = points_line
            
            images.append(image_data)
        
        i += 1
    
    return header_lines, images


def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    """Convert quaternion to rotation matrix"""
    r = Rotation.from_quat([qx, qy, qz, qw])
    return r.as_matrix()


def get_camera_axes(qw, qx, qy, qz):
    """
    Get camera coordinate axes in world frame
    
    In COLMAP/OpenCV convention:
    - X axis points right
    - Y axis points down
    - Z axis points forward (viewing direction)
    
    Returns:
        right, down, forward vectors in world coordinates
    """
    R = quaternion_to_rotation_matrix(qw, qx, qy, qz)
    
    # Camera axes in camera frame
    x_cam = np.array([1, 0, 0])  # right
    y_cam = np.array([0, 1, 0])  # down
    z_cam = np.array([0, 0, 1])  # forward
    
    # Transform to world frame
    right = R @ x_cam
    down = R @ y_cam
    forward = R @ z_cam
    
    return right, down, forward


def modify_camera_positions(images, offset_distance=0.5):
    """
    Create 4 variants for each camera:
    - front: move in forward direction
    - back: move in backward direction
    - left: move in left direction
    - right: move in right direction
    
    Args:
        images: List of image data
        offset_distance: Distance to move in meters (default: 0.5)
    
    Returns:
        Dictionary with 4 lists of modified images
    """
    variants = {
        'front': [],
        'back': [],
        'left': [],
        'right': []
    }
    
    for img in images:
        # Get camera axes
        right, down, forward = get_camera_axes(
            img['qw'], img['qx'], img['qy'], img['qz']
        )
        
        # Original position
        pos = np.array([img['tx'], img['ty'], img['tz']])
        
        # Calculate new positions
        # Front: move in forward direction (Z+)
        pos_front = pos + forward * offset_distance
        
        # Back: move in backward direction (Z-)
        pos_back = pos - forward * offset_distance
        
        # Left: move in left direction (-X)
        pos_left = pos - right * offset_distance
        
        # Right: move in right direction (X+)
        pos_right = pos + right * offset_distance
        
        # Create modified entries (keep same rotation, only change translation)
        for direction, new_pos in [
            ('front', pos_front),
            ('back', pos_back),
            ('left', pos_left),
            ('right', pos_right)
        ]:
            modified = img.copy()
            modified['tx'] = new_pos[0]
            modified['ty'] = new_pos[1]
            modified['tz'] = new_pos[2]
            # Modify filename to indicate direction
            name_parts = img['name'].rsplit('.', 1)
            if len(name_parts) == 2:
                modified['name'] = f"{name_parts[0]}_{direction}.{name_parts[1]}"
            else:
                modified['name'] = f"{img['name']}_{direction}"
            
            variants[direction].append(modified)
    
    return variants


def write_images_file(filepath, header_lines, images):
    """Write modified images to file"""
    with open(filepath, 'w') as f:
        # Write header
        for line in header_lines[:-1]:
            f.write(line)
        
        # Update image count in last header line
        f.write(f"# Number of images: {len(images)}\n")
        
        # Write image data
        for img in images:
            # Write image line
            f.write(f"{img['image_id']} {img['qw']:.6f} {img['qx']:.6f} {img['qy']:.6f} {img['qz']:.6f} "
                   f"{img['tx']:.6f} {img['ty']:.6f} {img['tz']:.6f} {img['camera_id']} {img['name']}\n")
            # Write points line
            f.write(img['points_line'])


def main():
    parser = argparse.ArgumentParser(
        description='Modify camera positions by moving them in 4 directions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Create 4 files with 0.5m offset in each direction
  python modify_camera_positions.py --input images.txt --output-dir ./modified --distance 0.5
  
  # Custom offset distance
  python modify_camera_positions.py --input images.txt --output-dir ./modified --distance 1.0
  
  # Create single file with all variants
  python modify_camera_positions.py --input images.txt --output images_all.txt --combine
        '''
    )
    
    parser.add_argument('--input', type=str, required=True,
                       help='Input COLMAP images.txt file')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for separate files (one per direction)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for combined results (all directions)')
    parser.add_argument('--distance', type=float, default=0.5,
                       help='Distance to move cameras in meters (default: 0.5)')
    parser.add_argument('--combine', action='store_true',
                       help='Combine all directions into single file')
    parser.add_argument('--directions', type=str, nargs='+', 
                       choices=['front', 'back', 'left', 'right'],
                       default=['front', 'back', 'left', 'right'],
                       help='Directions to generate (default: all)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("Camera Position Modifier")
    print("="*70)
    print(f"\nInput file: {args.input}")
    print(f"Offset distance: {args.distance}m")
    print(f"Directions: {', '.join(args.directions)}")
    
    # Parse input file
    print("\nParsing input file...")
    header_lines, images = parse_images_file(args.input)
    print(f"Found {len(images)} cameras")
    
    # Modify positions
    print("\nModifying camera positions...")
    variants = modify_camera_positions(images, args.distance)
    
    # Print statistics
    print("\nGenerated variants:")
    for direction in args.directions:
        print(f"  {direction}: {len(variants[direction])} cameras")
    
    # Write output files
    if args.combine or args.output:
        # Combine all directions into one file
        output_file = args.output if args.output else 'images_modified_all.txt'
        
        # Combine selected directions
        combined_images = []
        current_id = 1
        for direction in args.directions:
            for img in variants[direction]:
                img_copy = img.copy()
                img_copy['image_id'] = current_id
                combined_images.append(img_copy)
                current_id += 1
        
        write_images_file(output_file, header_lines, combined_images)
        print(f"\n✓ Combined file written: {output_file}")
        print(f"  Total cameras: {len(combined_images)}")
    
    if args.output_dir:
        # Write separate files for each direction
        import os
        os.makedirs(args.output_dir, exist_ok=True)
        
        for direction in args.directions:
            output_file = os.path.join(args.output_dir, f'images_{direction}.txt')
            
            # Reassign image IDs
            direction_images = []
            for idx, img in enumerate(variants[direction], start=1):
                img_copy = img.copy()
                img_copy['image_id'] = idx
                direction_images.append(img_copy)
            
            write_images_file(output_file, header_lines, direction_images)
            print(f"\n✓ {direction.capitalize()} file written: {output_file}")
            print(f"  Cameras: {len(direction_images)}")
    
    if not args.combine and not args.output and not args.output_dir:
        print("\n⚠ No output specified. Use --output, --output-dir, or --combine")
        print("  Example: --output images_modified.txt")
    
    print("\n" + "="*70)
    print("Done!")
    print("="*70)


if __name__ == "__main__":
    main()