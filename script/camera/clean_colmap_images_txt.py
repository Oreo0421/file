import os

images_dir = "data/neuman/dataset/lab/images"
images_txt = "data/neuman/dataset/lab/sparse/images.txt"
out_txt = "data/neuman/dataset/lab/sparse/images_clean.txt"

# 真实存在的图片名集合
valid_images = set(os.listdir(images_dir))

with open(images_txt, "r") as f:
    lines = f.readlines()

new_lines = []
i = 0
while i < len(lines):
    line = lines[i].strip()
    if line.startswith("#") or line == "":
        new_lines.append(lines[i])
        i += 1
        continue

    # 每两行一组：第一行是相机位姿+图片名，第二行是2D点
    parts = line.split()
    img_name = parts[-1]

    if img_name in valid_images:
        new_lines.append(lines[i])
        new_lines.append(lines[i+1])
    else:
        print("Removing missing image:", img_name)

    i += 2

with open(out_txt, "w") as f:
    f.writelines(new_lines)

print("Cleaned file saved to:", out_txt)
