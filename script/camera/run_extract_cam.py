#!/usr/bin/env python3
"""
Improved camera parameter extraction script
- Can run from any directory
- Automatically finds project root
- Supports both HUGS dataset and COLMAP files
"""

import argparse
import sys
import os
from pathlib import Path

# ============================================================================
# Automatically find and add project root to Python path
# ============================================================================

def find_project_root(marker_files=['hugs', 'setup.py', 'requirements.txt', '.git']):
    """
    Find project root directory by searching upward from current directory
    
    Args:
        marker_files: Files/folders used to identify the root directory
    
    Returns:
        Path object or None
    """
    current = Path.cwd()
    
    # Search up to 10 levels
    for _ in range(10):
        # Check if marker files exist
        for marker in marker_files:
            if (current / marker).exists():
                return current
        
        # Reached root directory
        if current.parent == current:
            break
        
        current = current.parent
    
    return None

def setup_python_path(project_root=None):
    """
    Setup Python path to import hugs module
    
    Args:
        project_root: Project root directory path (optional)
    
    Returns:
        Project root directory path
    """
    if project_root is None:
        project_root = find_project_root()
    
    if project_root:
        project_root_str = str(project_root)
        if project_root_str not in sys.path:
            sys.path.insert(0, project_root_str)
            print(f"✓ Found project root: {project_root}")
        return project_root
    else:
        print("⚠ Warning: Project root not found")
        return None

# Setup path
PROJECT_ROOT = setup_python_path()

# ============================================================================
# Import modules
# ============================================================================

# Try to import HUGS dataset
try:
    from hugs.datasets import NeumanDataset
    HUGS_AVAILABLE = True
    print("✓ HUGS dataset module loaded successfully")
except ImportError as e:
    HUGS_AVAILABLE = False
    print(f"⚠ HUGS dataset unavailable: {e}")
    print("  Will use COLMAP file mode")

# Import extract_camera module
try:
    from extract_camera import CameraParamsExtractor
    print("✓ extract_camera module loaded successfully")
except ImportError:
    print("✗ Error: Cannot find extract_camera module")
    print("  Please ensure extract_camera.py is in one of these locations:")
    print(f"  1. Current directory: {Path.cwd()}")
    print(f"  2. Script directory: {Path(__file__).parent}")
    if PROJECT_ROOT:
        print(f"  3. Project root: {PROJECT_ROOT}")
    sys.exit(1)

# ============================================================================
# Helper functions
# ============================================================================

def find_colmap_files(seq=None, split='train', search_dirs=None):
    """
    Intelligently find COLMAP format files
    
    Args:
        seq: Sequence name (optional)
        split: Dataset split
        search_dirs: List of directories to search (optional)
    
    Returns:
        tuple: (images_file, cameras_file) or (None, None)
    """
    if search_dirs is None:
        # Default search directories
        search_dirs = [
            Path.cwd(),  # Current directory
            Path.cwd() / 'sparse',
            Path.cwd() / 'output',
        ]
        
        if PROJECT_ROOT:
            search_dirs.extend([
                PROJECT_ROOT,
                PROJECT_ROOT / 'sparse',
                PROJECT_ROOT / 'output',
                PROJECT_ROOT / 'data',
            ])
        
        if seq:
            search_dirs.extend([
                Path.cwd() / seq,
                Path.cwd() / seq / 'sparse',
            ])
            if PROJECT_ROOT:
                search_dirs.extend([
                    PROJECT_ROOT / seq,
                    PROJECT_ROOT / seq / 'sparse',
                    PROJECT_ROOT / 'data' / seq,
                ])
    
    # Possible filename combinations
    if seq:
        filename_patterns = [
            (f'images_{seq}_{split}.txt', f'cameras_{seq}.txt'),
            (f'images_{seq}.txt', f'cameras_{seq}.txt'),
            (f'images_djr_1m.txt', 'cameras.txt'),
            ('images.txt', 'cameras.txt'),
        ]
    else:
        filename_patterns = [
            (f'images_djr_1m.txt', 'cameras.txt'),
            ('images.txt', 'cameras.txt'),
        ]
    
    # Search for files
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        
        for images_name, cameras_name in filename_patterns:
            images_path = search_dir / images_name
            cameras_path = search_dir / cameras_name
            
            if images_path.exists() and cameras_path.exists():
                print(f"\n✓ Found COLMAP files:")
                print(f"  Images:  {images_path}")
                print(f"  Cameras: {cameras_path}")
                return str(images_path), str(cameras_path)
    
    return None, None

def validate_output_dir(output_path):
    """
    Validate and create output directory
    
    Args:
        output_path: Output path
    
    Returns:
        Path object
    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory: {output_dir.absolute()}")
    return output_dir

# ============================================================================
# Main function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Extract camera parameters from HUGS dataset or COLMAP files (can run from any directory)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Usage examples:

  1. Use COLMAP files (auto-find):
     python run_extract_camera.py --seq lab --split train --output ./output
  
  2. Specify COLMAP files explicitly:
     python run_extract_camera.py --images images.txt --cameras cameras.txt --output ./output
  
  3. Use HUGS dataset:
     python run_extract_camera.py --seq seattle --split train --use-hugs --output ./output
  
  4. Specify project root:
     python run_extract_camera.py --seq lab --project-root /path/to/project --output ./output
        '''
    )
    
    # Basic parameters
    parser.add_argument('--seq', type=str, help='Sequence name (e.g., seattle, citron, lab)')
    parser.add_argument('--split', type=str, default='train', 
                       choices=['train', 'val', 'test', 'anim'], 
                       help='Dataset split (default: train)')
    parser.add_argument('--output', type=str, default='./output', 
                       help='Output directory (default: ./output)')
    
    # COLMAP file parameters
    parser.add_argument('--images', type=str, default=None, 
                       help='COLMAP images.txt file path (optional)')
    parser.add_argument('--cameras', type=str, default=None,
                       help='COLMAP cameras.txt file path (optional)')
    parser.add_argument('--search-dirs', type=str, nargs='+', default=None,
                       help='List of directories to search for COLMAP files (optional)')
    
    # Advanced parameters
    parser.add_argument('--project-root', type=str, default=None,
                       help='Project root directory (for importing hugs module)')
    parser.add_argument('--use-hugs', action='store_true',
                       help='Force using HUGS dataset (requires --seq)')
    parser.add_argument('--verbose', action='store_true',
                       help='Show verbose output')
    
    args = parser.parse_args()
    
    # If project root is specified, re-setup path
    if args.project_root:
        global PROJECT_ROOT
        PROJECT_ROOT = setup_python_path(Path(args.project_root))
    
    print("\n" + "="*70)
    print("Camera Parameter Extraction Tool")
    print("="*70 + "\n")
    
    # Validate output directory
    output_dir = validate_output_dir(args.output)
    
    # Decide which mode to use
    use_hugs = args.use_hugs
    images_file = args.images
    cameras_file = args.cameras
    
    # ========================================================================
    # Mode 1: Explicitly use HUGS dataset
    # ========================================================================
    if use_hugs:
        if not HUGS_AVAILABLE:
            print("✗ Error: HUGS dataset unavailable")
            print("  Please ensure:")
            print("  1. HUGS dependencies are installed")
            print("  2. Running from correct project directory")
            print("  3. Or use --project-root to specify project root")
            sys.exit(1)
        
        if not args.seq:
            print("✗ Error: --seq parameter required when using HUGS dataset")
            sys.exit(1)
        
        print(f"Mode: HUGS dataset")
        print(f"Sequence: {args.seq}")
        print(f"Split: {args.split}")
        
        try:
            dataset = NeumanDataset(seq=args.seq, split=args.split)
            extractor = CameraParamsExtractor(dataset=dataset)
            output_path = output_dir / f'{args.seq}_{args.split}_camera_params'
        except Exception as e:
            print(f"✗ Error: Cannot load HUGS dataset: {e}")
            sys.exit(1)
    
    # ========================================================================
    # Mode 2: Use COLMAP files
    # ========================================================================
    else:
        # If files not specified, try to find them
        if images_file is None or cameras_file is None:
            search_dirs = [Path(d) for d in args.search_dirs] if args.search_dirs else None
            images_file, cameras_file = find_colmap_files(
                args.seq, args.split, search_dirs
            )
        
        if images_file and cameras_file:
            print(f"\nMode: COLMAP files")
            
            # Verify files exist
            if not Path(images_file).exists():
                print(f"✗ Error: Images file does not exist: {images_file}")
                sys.exit(1)
            if not Path(cameras_file).exists():
                print(f"✗ Error: Cameras file does not exist: {cameras_file}")
                sys.exit(1)
            
            extractor = CameraParamsExtractor(
                images_file=images_file,
                cameras_file=cameras_file
            )
            
            # Determine output filename
            if args.seq:
                output_path = output_dir / f'{args.seq}_{args.split}_camera_params'
            else:
                output_path = output_dir / f'camera_params'
        
        # If HUGS available and seq specified, try using HUGS
        elif HUGS_AVAILABLE and args.seq:
            print(f"\nMode: HUGS dataset (fallback)")
            print(f"Sequence: {args.seq}")
            print(f"Split: {args.split}")
            
            try:
                dataset = NeumanDataset(seq=args.seq, split=args.split)
                extractor = CameraParamsExtractor(dataset=dataset)
                output_path = output_dir / f'{args.seq}_{args.split}_camera_params'
            except Exception as e:
                print(f"✗ Error: Cannot load data: {e}")
                print("\nCannot find valid data source. Please provide one of:")
                print("  1. --images and --cameras parameters for COLMAP files")
                print("  2. --seq parameter (if HUGS dataset available)")
                print("  3. Place COLMAP files in current directory or standard locations")
                sys.exit(1)
        
        else:
            print("\n✗ Error: Cannot find valid data source")
            print("\nPlease provide one of:")
            print("  1. Use --images and --cameras to specify COLMAP files")
            print("  2. Use --seq parameter (requires HUGS dataset or COLMAP files)")
            print("  3. Place COLMAP files in current directory or these locations:")
            print("     - ./images.txt and ./cameras.txt")
            print("     - ./sparse/images.txt and ./sparse/cameras.txt")
            if PROJECT_ROOT:
                print(f"     - {PROJECT_ROOT}/images.txt and {PROJECT_ROOT}/cameras.txt")
            sys.exit(1)
    
    # ========================================================================
    # Extract and save camera parameters
    # ========================================================================
    print(f"\n{'='*70}")
    print("Starting camera parameter extraction...")
    print(f"{'='*70}\n")
    
    try:
        camera_params = extractor.extract_and_save(str(output_path))
        
        print(f"\n{'='*70}")
        print("✓ Extraction successful!")
        print(f"{'='*70}")
        print(f"\nOutput files:")
        print(f"  Full format: {output_path}.json")
        print(f"  DJR format:  {output_path}_djr.json")
        print(f"\nNumber of cameras: {len(camera_params['frames'])}")
        
        if args.verbose:
            print(f"\nCamera intrinsics:")
            intrinsics = camera_params['camera_intrinsics']
            for key, value in intrinsics.items():
                print(f"  {key}: {value}")
        
        print()
        
    except Exception as e:
        print(f"\n✗ Error: Extraction failed")
        print(f"  {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()