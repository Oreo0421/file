#!/usr/bin/env python3
import argparse
import sys
sys.path.append('.')

from extract_camera import CameraParamsExtractor
from hugs.datasets import NeumanDataset

def main():
    parser = argparse.ArgumentParser(description='Extract camera parameters from HUGS dataset')
    parser.add_argument('--seq', type=str, required=True, help='Sequence name (e.g., seattle, citron)')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'anim'], help='Dataset split')
    parser.add_argument('--output', type=str, default='./output', help='Output directory')

    args = parser.parse_args()

    # Create dataset
    dataset = NeumanDataset(seq=args.seq, split=args.split)

    # Extract camera parameters
    extractor = CameraParamsExtractor(dataset)
    output_path = f'{args.output}/{args.seq}_{args.split}_camera_params'
    camera_params = extractor.extract_and_save(output_path)

    print(f"Successfully extracted camera parameters for {args.seq} {args.split}")
    print(f"Output saved to: {output_path}json")

if __name__ == "__main__":
    main()
