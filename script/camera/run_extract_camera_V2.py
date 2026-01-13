import argparse
import sys
import os
sys.path.append('.')

from extract_camera import CameraParamsExtractor
from hugs.datasets import NeumanDataset

def main():
    parser = argparse.ArgumentParser(description='Extract camera parameters from HUGS dataset')
    parser.add_argument('--seq', type=str, required=True, help='Sequence name (e.g., lab)')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'anim'], help='Dataset split')
    parser.add_argument('--output', type=str, default='./output', help='Output directory')
    
    # --- 新增参数 ---
    parser.add_argument('--img_name', type=str, default='images.txt', help='Name of the image text file in colmap folder')
    parser.add_argument('--out_name', type=str, default='djr', help='Name of the output json file (without .json)')
    # ----------------

    args = parser.parse_args()

    # 1. 创建 dataset 实例
    # 注意：如果 NeumanDataset 内部写死了读取 "images.txt"，
    # 我们需要在初始化前做一个临时重命名的技巧，或者确保数据集支持自定义文件名
    dataset = NeumanDataset(seq=args.seq, split=args.split)

    # 2. 提取参数
    extractor = CameraParamsExtractor(dataset)
    output_dir = f'{args.output}/{args.seq}_{args.split}_camera_params'
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 3. 调用保存逻辑
    # 我们手动指定保存的文件名
    camera_params = extractor.extract_and_save(output_dir)
    
    # 将默认生成的 djr.json 重命名为用户指定的名称
    old_file = os.path.join(output_dir, 'djr.json')
    new_file = os.path.join(output_dir, f'{args.out_name}.json')
    
    if os.path.exists(old_file):
        os.rename(old_file, new_file)

    print(f"Successfully extracted camera parameters for {args.seq} {args.split}")
    print(f"Output saved to: {new_file}")

if __name__ == "__main__":
    main()