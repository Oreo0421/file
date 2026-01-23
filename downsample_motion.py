#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动降采样动作序列脚本
支持 .npz 文件和包含 .npy 文件的文件夹
自动检测帧数，每N帧抽取1帧

使用方法：
    # 处理单个文件（默认每5帧取1帧）
    python downsample_motion.py pose.npz
    
    # 指定采样步长
    python downsample_motion.py pose.npz -s 3
    
    # 指定输出路径
    python downsample_motion.py pose.npz -o output.npz
    
    # 处理文件夹
    python downsample_motion.py /path/to/motion_folder -s 5
    
    # 批量处理所有npz文件
    python downsample_motion.py . --batch -s 5
"""

import argparse
import numpy as np
from pathlib import Path


def downsample_npz(input_path, output_path, step=5):
    """处理 .npz 文件"""
    data = np.load(input_path, allow_pickle=True)
    
    # 检测帧数（从 poses 或 trans 获取）
    if 'poses' in data.files:
        original_frames = data['poses'].shape[0]
        key_to_check = 'poses'
    elif 'trans' in data.files or 'transl' in data.files:
        trans_key = 'trans' if 'trans' in data.files else 'transl'
        original_frames = data[trans_key].shape[0]
        key_to_check = trans_key
    else:
        raise KeyError(f"文件 {input_path} 中找不到 'poses' 或 'trans/transl' 键")
    
    print(f"处理: {input_path}")
    print(f"  原始帧数: {original_frames}")
    
    # 降采样所有时序数据
    sampled_data = {}
    for key in data.files:
        arr = data[key]
        
        # 判断是否是时序数据（第一维是帧数）
        if isinstance(arr, np.ndarray) and arr.ndim >= 1 and arr.shape[0] == original_frames:
            sampled_data[key] = arr[::step]
            print(f"  {key}: {arr.shape} -> {sampled_data[key].shape}")
        else:
            # 保持不变（如 betas, gender 等）
            sampled_data[key] = arr
            if isinstance(arr, np.ndarray):
                print(f"  {key}: {arr.shape} (保持不变)")
            else:
                print(f"  {key}: (标量/其他)")
    
    sampled_frames = sampled_data[key_to_check].shape[0]
    print(f"  采样后帧数: {sampled_frames}")
    print(f"  采样率: 1/{step}")
    
    # 保存
    np.savez(output_path, **sampled_data)
    print(f"✓ 已保存到: {output_path}\n")


def downsample_folder(input_dir, output_dir, step=5):
    """处理文件夹（包含 poses.npy, trans.npy 等）"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 检测帧数
    if (input_dir / "poses.npy").exists():
        poses = np.load(input_dir / "poses.npy")
        original_frames = poses.shape[0]
        key_to_check = "poses.npy"
    elif (input_dir / "trans.npy").exists():
        trans = np.load(input_dir / "trans.npy")
        original_frames = trans.shape[0]
        key_to_check = "trans.npy"
    else:
        raise FileNotFoundError(f"文件夹 {input_dir} 中找不到 poses.npy 或 trans.npy")
    
    print(f"处理文件夹: {input_dir}")
    print(f"  原始帧数: {original_frames}")
    
    # 处理所有 .npy 文件
    for npy_file in input_dir.glob("*.npy"):
        arr = np.load(npy_file)
        
        # 判断是否是时序数据
        if arr.ndim >= 1 and arr.shape[0] == original_frames:
            sampled = arr[::step]
            np.save(output_dir / npy_file.name, sampled)
            print(f"  {npy_file.name}: {arr.shape} -> {sampled.shape}")
        else:
            # 直接复制
            np.save(output_dir / npy_file.name, arr)
            print(f"  {npy_file.name}: {arr.shape} (保持不变)")
    
    sampled_frames = int(np.ceil(original_frames / step))
    print(f"  采样后帧数: {sampled_frames}")
    print(f"✓ 已保存到: {output_dir}\n")


def batch_process_npz(directory, step=5, suffix="_sampled"):
    """批量处理目录下所有 .npz 文件"""
    directory = Path(directory)
    npz_files = list(directory.glob("*.npz"))
    
    if not npz_files:
        print(f"在 {directory} 中没有找到 .npz 文件")
        return
    
    print(f"找到 {len(npz_files)} 个 .npz 文件")
    print("="*60)
    
    for npz_file in npz_files:
        try:
            output_path = npz_file.with_name(f"{npz_file.stem}{suffix}.npz")
            downsample_npz(str(npz_file), str(output_path), step)
        except Exception as e:
            print(f"✗ 处理 {npz_file} 时出错: {e}\n")
    
    print("="*60)
    print(f"批量处理完成！")


def main():
    parser = argparse.ArgumentParser(
        description="自动降采样动作序列 - 支持任意帧数",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 处理单个文件
  %(prog)s pose.npz
  %(prog)s pose.npz -s 3 -o output.npz
  
  # 处理文件夹
  %(prog)s /path/to/motion_folder -s 5
  
  # 批量处理当前目录所有.npz文件
  %(prog)s . --batch -s 5
        """
    )
    
    parser.add_argument("input", help="输入文件(.npz)、文件夹路径、或批量处理的目录")
    parser.add_argument("-o", "--output", help="输出路径（默认添加 _sampled 后缀）")
    parser.add_argument("-s", "--step", type=int, default=5, 
                       help="采样步长（每N帧取1帧，默认5）")
    parser.add_argument("--batch", action="store_true",
                       help="批量处理目录下所有 .npz 文件")
    parser.add_argument("--suffix", default="_sampled",
                       help="批量处理时的输出文件后缀（默认 _sampled）")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    # 批量处理模式
    if args.batch:
        if not input_path.is_dir():
            print(f"错误: 批量处理需要指定一个目录，而不是文件")
            return
        batch_process_npz(input_path, args.step, args.suffix)
        return
    
    # 自动生成输出路径
    if args.output:
        output_path = args.output
    else:
        if input_path.is_file() and input_path.suffix == '.npz':
            output_path = str(input_path.with_name(f"{input_path.stem}_sampled.npz"))
        elif input_path.is_dir():
            output_path = str(input_path.with_name(f"{input_path.name}_sampled"))
        else:
            raise ValueError("输入必须是 .npz 文件或文件夹")
    
    # 处理
    try:
        if input_path.is_file() and input_path.suffix == '.npz':
            downsample_npz(str(input_path), output_path, args.step)
        elif input_path.is_dir():
            downsample_folder(str(input_path), output_path, args.step)
        else:
            raise ValueError("输入必须是 .npz 文件或包含 .npy 文件的文件夹")
    except Exception as e:
        print(f"错误: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
