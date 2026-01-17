import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="批量修改 images.txt 中相机的四元数和位置")
    parser.add_argument("--input", type=str, required=True, help="原始 images.txt 文件路径")
    parser.add_argument("--output", type=str, required=True, help="输出文件保存路径")
    parser.add_argument("--quat", nargs=4, type=float, required=True, metavar=('qw', 'qx', 'qy', 'qz'),
                        help="四元数 (qw qx qy qz)，例如：0.7071 -0.7071 0 0")
    parser.add_argument("--position", nargs=3, type=float, required=True, metavar=('tx', 'ty', 'tz'),
                        help="相机位置 (tx ty tz)，例如：0 2.5 1.0")
    return parser.parse_args()

def main():
    args = parse_args()

    qw, qx, qy, qz = args.quat
    tx, ty, tz = args.position
    input_path = Path(args.input)
    output_path = Path(args.output)

    modified_lines = []
    with open(input_path, "r") as f:
        for line in f:
            if line.strip() == "" or line.startswith("#"):
                modified_lines.append(line)
                continue
            parts = line.strip().split()
            parts[1:5] = [f"{qw:.6f}", f"{qx:.6f}", f"{qy:.6f}", f"{qz:.6f}"]
            parts[5:8] = [f"{tx:.6f}", f"{ty:.6f}", f"{tz:.6f}"]
            modified_lines.append(" ".join(parts) + "\n")

    with open(output_path, "w") as f:
        f.writelines(modified_lines)

    print(f"✅ 修改完成：保存至 {output_path}")

if __name__ == "__main__":
    main()
