#!/usr/bin/env python3
"""
PLY to PT Converter
将Blender导出的PLY文件转换为PyTorch的PT格式
"""

import numpy as np
import torch
from pathlib import Path
from plyfile import PlyData, PlyElement
import argparse


def read_ply(ply_path):
    """
    读取PLY文件
    
    返回:
        xyz: 点的坐标 (N, 3)
        rgb: 点的颜色 (N, 3)，范围0-1
        normals: 点的法线 (N, 3)，如果有的话
    """
    print(f"读取PLY文件: {ply_path}")
    
    plydata = PlyData.read(ply_path)
    vertices = plydata['vertex']
    
    # 提取坐标
    xyz = np.stack([vertices['x'], vertices['y'], vertices['z']], axis=1)
    print(f"  点的数量: {len(xyz)}")
    
    # 提取颜色（如果有的话）
    rgb = None
    if 'red' in vertices.dtype.names:
        # 颜色通常是0-255的整数
        r = vertices['red'].astype(np.float32) / 255.0
        g = vertices['green'].astype(np.float32) / 255.0
        b = vertices['blue'].astype(np.float32) / 255.0
        rgb = np.stack([r, g, b], axis=1)
        print(f"  包含颜色信息")
    
    # 提取法线（如果有的话）
    normals = None
    if 'nx' in vertices.dtype.names:
        normals = np.stack([vertices['nx'], vertices['ny'], vertices['nz']], axis=1)
        print(f"  包含法线信息")
    
    return xyz, rgb, normals


def create_point_cloud_dict(xyz, rgb=None, normals=None):
    """
    创建点云字典（适用于大多数3D重建框架）
    
    参数:
        xyz: 点坐标 (N, 3)
        rgb: 点颜色 (N, 3)，范围0-1
        normals: 点法线 (N, 3)
    
    返回:
        point_cloud: 字典格式的点云数据
    """
    point_cloud = {
        'xyz': torch.from_numpy(xyz).float(),
    }
    
    if rgb is not None:
        point_cloud['rgb'] = torch.from_numpy(rgb).float()
    else:
        # 如果没有颜色，使用白色
        point_cloud['rgb'] = torch.ones((len(xyz), 3), dtype=torch.float32)
    
    if normals is not None:
        point_cloud['normals'] = torch.from_numpy(normals).float()
    
    # 添加一些常用的额外字段
    point_cloud['num_points'] = len(xyz)
    
    return point_cloud


def save_pt(point_cloud, output_path):
    """保存为PT文件"""
    print(f"\n保存PT文件到: {output_path}")
    torch.save(point_cloud, output_path)
    print(f"  成功保存")
    
    # 打印保存的内容摘要
    print(f"\n保存的数据:")
    for key, value in point_cloud.items():
        if torch.is_tensor(value):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: {value}")


def ply_to_pt(ply_path, pt_path):
    """
    主转换函数：PLY -> PT
    
    参数:
        ply_path: 输入的PLY文件路径
        pt_path: 输出的PT文件路径
    """
    # 读取PLY
    xyz, rgb, normals = read_ply(ply_path)
    
    # 创建点云字典
    point_cloud = create_point_cloud_dict(xyz, rgb, normals)
    
    # 保存PT
    save_pt(point_cloud, pt_path)
    
    return point_cloud


def verify_pt(pt_path):
    """验证PT文件"""
    print(f"\n验证PT文件: {pt_path}")
    
    data = torch.load(pt_path)
    
    print(f"文件内容:")
    for key, value in data.items():
        if torch.is_tensor(value):
            print(f"  {key}:")
            print(f"    shape: {value.shape}")
            print(f"    dtype: {value.dtype}")
            print(f"    device: {value.device}")
            if value.numel() > 0:
                print(f"    min: {value.min().item():.6f}")
                print(f"    max: {value.max().item():.6f}")
                print(f"    mean: {value.mean().item():.6f}")
        else:
            print(f"  {key}: {value}")


def main():
    parser = argparse.ArgumentParser(description='Convert PLY to PT format')
    parser.add_argument('input', type=str, help='Input PLY file path')
    parser.add_argument('output', type=str, nargs='?', help='Output PT file path (default: same as input with .pt extension)')
    parser.add_argument('--verify', action='store_true', help='Verify the output PT file')
    
    args = parser.parse_args()
    
    # 确定输出路径
    if args.output is None:
        output_path = Path(args.input).with_suffix('.pt')
    else:
        output_path = Path(args.output)
    
    # 转换
    print("="*80)
    print("PLY to PT Converter")
    print("="*80)
    
    point_cloud = ply_to_pt(args.input, output_path)
    
    # 验证
    if args.verify:
        print("\n" + "="*80)
        verify_pt(output_path)
    
    print("\n" + "="*80)
    print("✓ 转换完成！")
    print("="*80)


if __name__ == "__main__":
    # 如果没有命令行参数，显示使用说明
    import sys
    if len(sys.argv) == 1:
        print("PLY to PT Converter")
        print("="*80)
        print("\n使用方法:")
        print("  python ply_to_pt.py input.ply [output.pt] [--verify]")
        print("\n示例:")
        print("  python ply_to_pt.py scene.ply")
        print("  python ply_to_pt.py scene.ply scene_fixed.pt")
        print("  python ply_to_pt.py scene.ply --verify")
        print("\n参数:")
        print("  input.ply    : 输入的PLY文件（从Blender导出）")
        print("  output.pt    : 输出的PT文件（可选，默认与输入同名）")
        print("  --verify     : 转换后验证PT文件")
        print("\n在代码中使用:")
        print("  from ply_to_pt import ply_to_pt")
        print("  point_cloud = ply_to_pt('scene.ply', 'scene.pt')")
    else:
        main()
