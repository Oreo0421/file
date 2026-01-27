#!/usr/bin/env python3
"""
Camera Parameters Extractor for HUGS Dataset
This script extracts camera parameters from the dataset and saves them as JSON files.
"""

import os
import json
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path

class CameraParamsExtractor:
    def __init__(self, dataset):

        self.dataset = dataset

    def extract_camera_params(self, indices=None):

        if indices is None:
            indices = range(len(self.dataset))

        camera_params = {
            'world_view_transform': [],
            'c2w': [],
            'full_proj_transform': [],
            'camera_center': [],
            'cam_intrinsics': [],
            'image_height': [],
            'image_width': [],
            'fovx': [],
            'fovy': [],
            'near': [],
            'far': [],
            'frame_ids': [],
            'view_ids': [],
            'cam_ids': []
        }

        print(f"Extracting camera parameters from {len(indices)} samples...")

        for idx in tqdm(indices, desc="Processing"):
            try:
                data = self.dataset[idx]

                # Extract camera matrices and transforms
                if 'world_view_transform' in data:
                    camera_params['world_view_transform'].append(
                        data['world_view_transform'].cpu().numpy() if torch.is_tensor(data['world_view_transform'])
                        else data['world_view_transform']
                    )

                if 'c2w' in data:
                    camera_params['c2w'].append(
                        data['c2w'].cpu().numpy() if torch.is_tensor(data['c2w'])
                        else data['c2w']
                    )

                if 'full_proj_transform' in data:
                    camera_params['full_proj_transform'].append(
                        data['full_proj_transform'].cpu().numpy() if torch.is_tensor(data['full_proj_transform'])
                        else data['full_proj_transform']
                    )

                if 'camera_center' in data:
                    camera_params['camera_center'].append(
                        data['camera_center'].cpu().numpy() if torch.is_tensor(data['camera_center'])
                        else data['camera_center']
                    )

                # Extract camera intrinsics (changed from 'intrinsics')
                if 'cam_intrinsics' in data:
                    camera_params['cam_intrinsics'].append(
                        data['cam_intrinsics'].cpu().numpy() if torch.is_tensor(data['cam_intrinsics'])
                        else data['cam_intrinsics']
                    )

                # Extract camera parameters
                for param in ['image_height', 'image_width', 'fovx', 'fovy']:
                    if param in data:
                        value = data[param]
                        if torch.is_tensor(value):
                            value = value.cpu().numpy()
                        camera_params[param].append(value)

                # Extract near and far (changed from znear, zfar)
                if 'near' in data:
                    value = data['near']
                    if torch.is_tensor(value):
                        value = value.cpu().numpy()
                    camera_params['near'].append(value)

                if 'far' in data:
                    value = data['far']
                    if torch.is_tensor(value):
                        value = value.cpu().numpy()
                    camera_params['far'].append(value)



                # Extract frame and view information
                if 'frame_id' in data:
                    camera_params['frame_ids'].append(data['frame_id'])
                if 'view_id' in data:
                    camera_params['view_ids'].append(data['view_id'])
                if 'cam_id' in data:
                    camera_params['cam_ids'].append(data['cam_id'])

            except Exception as e:
                print(f"Error processing index {idx}: {e}")
                continue

        return camera_params

    def save_camera_params_json(self, camera_params, output_dir):

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Convert numpy arrays to lists for JSON serialization
        json_params = {}
        for key, value in camera_params.items():
            if isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], np.ndarray):
                    json_params[key] = [v.tolist() for v in value]
                else:
                    json_params[key] = value
            else:
                json_params[key] = value

        # Save as JSON
        output_path = output_dir / 'camera_params.json'
        with open(output_path, 'w') as f:
            json.dump(json_params, f, indent=2)
        print(f"Saved camera parameters as JSON to {output_path}")

    def extract_and_save(self, output_dir, indices=None):

        camera_params = self.extract_camera_params(indices)
        self.save_camera_params_json(camera_params, output_dir)
        return camera_params

    def load_camera_params(self, file_path):

        file_path = Path(file_path)

        if file_path.suffix == '.json':
            with open(file_path, 'r') as f:
                return json.load(f)
        else:
            raise ValueError(f"Only JSON format is supported, got: {file_path.suffix}")

def extract_camera_params_from_trainer(trainer, split='train', output_dir='./camera_params'):

    if split == 'train':
        dataset = trainer.train_dataset
    elif split == 'val':
        dataset = trainer.val_dataset
    elif split == 'anim':
        dataset = trainer.anim_dataset
    else:
        raise ValueError(f"Unknown split: {split}")

    if dataset is None:
        print(f"Dataset for split '{split}' is None")
        return None

    extractor = CameraParamsExtractor(dataset)
    output_path = Path(output_dir) / split
    camera_params = extractor.extract_and_save(output_path)

    print(f"Extracted camera parameters for {split} split:")
    for key, value in camera_params.items():
        if isinstance(value, list):
            print(f"  {key}: {len(value)} items")
        else:
            print(f"  {key}: {value}")

    return camera_params

# Example usage function
def main():
    """
    Example usage of the camera parameters extractor.
    """
    print("Example usage:")
    print("1. From a trainer instance:")
    print("   camera_params = extract_camera_params_from_trainer(trainer, 'train', './output/camera_params')")
    print()
    print("2. From a dataset directly:")
    print("   extractor = CameraParamsExtractor(dataset)")
    print("   camera_params = extractor.extract_and_save('./output/camera_params')")
    print()
    print("3. Load saved parameters:")
    print("   extractor = CameraParamsExtractor(None)")
    print("   params = extractor.load_camera_params('./output/camera_params/camera_params.json')")

if __name__ == "__main__":
    main()
