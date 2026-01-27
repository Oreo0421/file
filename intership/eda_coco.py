import os
import json
import math
import argparse
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def safe_open_image_size(img_path: str):
    """Return (W, H) or (None, None) if missing/unreadable."""
    try:
        with Image.open(img_path) as im:
            return im.size[0], im.size[1]
    except Exception:
        return None, None


def load_coco(ann_path: str):
    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # categories: id -> name
    cat_id_to_name = {}
    for c in coco.get("categories", []):
        cat_id_to_name[int(c["id"])] = c.get("name", str(c["id"]))

    # images: id -> {file_name, width, height}
    img_id_to_info = {}
    for im in coco.get("images", []):
        img_id = int(im["id"])
        img_id_to_info[img_id] = {
            "file_name": im.get("file_name", ""),
            "width": im.get("width", None),
            "height": im.get("height", None),
        }

    anns = coco.get("annotations", [])
    return cat_id_to_name, img_id_to_info, anns


def build_bbox_df(cat_id_to_name, img_id_to_info, anns, img_dir: str):
    rows = []
    bad = Counter()

    for a in anns:
        img_id = int(a.get("image_id", -1))
        cat_id = int(a.get("category_id", -1))
        bbox = a.get("bbox", None)

        if img_id not in img_id_to_info:
            bad["ann_image_id_not_found"] += 1
            continue
        if bbox is None or len(bbox) != 4:
            bad["ann_bbox_missing"] += 1
            continue

        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            bad["ann_non_positive_wh"] += 1
            continue

        im_info = img_id_to_info[img_id]
        file_name = im_info.get("file_name", "")
        img_path = os.path.join(img_dir, file_name)

        W = im_info.get("width", None)
        H = im_info.get("height", None)

        # 如果 COCO json 没写 width/height，就尝试从图片读
        if (W is None or H is None) and file_name:
            W2, H2 = safe_open_image_size(img_path)
            W = W or W2
            H = H or H2

        if W is None or H is None:
            bad["image_wh_unknown"] += 1

        # basic validity checks (if W/H known)
        clipped = False
        out_of_bounds = False
        if W is not None and H is not None:
            if x < 0 or y < 0 or x + w > W or y + h > H:
                out_of_bounds = True
            # 轻微越界也很常见，这里不丢弃，只标记
            # 可选：裁剪到边界
            x2 = max(0.0, x)
            y2 = max(0.0, y)
            w2 = min(w, (W - x2)) if W is not None else w
            h2 = min(h, (H - y2)) if H is not None else h
            if x2 != x or y2 != y or w2 != w or h2 != h:
                clipped = True
                x, y, w, h = x2, y2, max(0.0, w2), max(0.0, h2)

        area = w * h
        ar = w / h if h != 0 else np.nan

        area_ratio = np.nan
        cx = x + w / 2
        cy = y + h / 2
        if W is not None and H is not None and W > 0 and H > 0:
            area_ratio = area / (W * H)

        rows.append({
            "image_id": img_id,
            "ann_id": int(a.get("id", -1)),
            "category_id": cat_id,
            "category_name": cat_id_to_name.get(cat_id, str(cat_id)),
            "file_name": file_name,
            "img_path": img_path,
            "img_w": W,
            "img_h": H,
            "x": float(x), "y": float(y), "w": float(w), "h": float(h),
            "cx": float(cx), "cy": float(cy),
            "area": float(area),
            "aspect_ratio": float(ar) if np.isfinite(ar) else np.nan,
            "area_ratio": float(area_ratio) if np.isfinite(area_ratio) else np.nan,
            "iscrowd": int(a.get("iscrowd", 0)),
            "clipped": bool(clipped),
            "out_of_bounds": bool(out_of_bounds),
        })

    df = pd.DataFrame(rows)
    return df, bad


def plot_hist(series, title, xlabel, out_path, logy=False, bins=50):
    plt.figure()
    s = series.dropna().astype(float)
    plt.hist(s, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    if logy:
        plt.yscale("log")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_bar(counter: pd.Series, title, xlabel, ylabel, out_path, topk=30):
    plt.figure(figsize=(10, 5))
    c = counter.sort_values(ascending=False).head(topk)
    plt.bar(range(len(c)), c.values)
    plt.xticks(range(len(c)), c.index, rotation=60, ha="right")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_scatter_wh(df, out_path):
    plt.figure()
    d = df[["img_w", "img_h"]].dropna()
    plt.scatter(d["img_w"], d["img_h"], s=5)
    plt.title("Image resolution scatter (W vs H)")
    plt.xlabel("W")
    plt.ylabel("H")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_center_heatmap(df, out_path, grid=64):
    """bbox center heatmap normalized to [0,1] using image size."""
    d = df.dropna(subset=["img_w", "img_h", "cx", "cy"]).copy()
    if len(d) == 0:
        return
    nx = grid
    ny = grid
    heat = np.zeros((ny, nx), dtype=np.float64)

    for _, r in d.iterrows():
        W, H = float(r["img_w"]), float(r["img_h"])
        if W <= 0 or H <= 0:
            continue
        ux = float(r["cx"]) / W
        uy = float(r["cy"]) / H
        ix = int(np.clip(math.floor(ux * nx), 0, nx - 1))
        iy = int(np.clip(math.floor(uy * ny), 0, ny - 1))
        heat[iy, ix] += 1.0

    plt.figure()
    plt.imshow(heat, origin="lower", aspect="auto")
    plt.title("BBox center heatmap (normalized)")
    plt.xlabel("x (bins)")
    plt.ylabel("y (bins)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def coco_size_bucket(area_px):
    # COCO 定义：small < 32^2, medium < 96^2, else large
    if area_px < 32 * 32:
        return "small"
    if area_px < 96 * 96:
        return "medium"
    return "large"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True, help="COCO instances_*.json")
    ap.add_argument("--imgdir", required=True, help="Image directory for file_name")
    ap.add_argument("--out", default="runs/eda", help="Output directory")
    args = ap.parse_args()

    ensure_dir(args.out)
    plots_dir = os.path.join(args.out, "plots")
    ensure_dir(plots_dir)

    cat_id_to_name, img_id_to_info, anns = load_coco(args.ann)
    df, bad = build_bbox_df(cat_id_to_name, img_id_to_info, anns, args.imgdir)

    # per-image stats
    img_counts = df.groupby("image_id").size().rename("num_boxes").reset_index()
    df = df.merge(img_counts, on="image_id", how="left")

    # bucket: COCO small/medium/large by bbox area(px)
    df["size_bucket"] = df["area"].apply(coco_size_bucket)

    # Export raw table
    df.to_csv(os.path.join(args.out, "bboxes.csv"), index=False)

    # Summary
    summary = {
        "num_images_in_json": len(img_id_to_info),
        "num_annotations_in_json": len(anns),
        "num_valid_bboxes": len(df),
        "num_categories": len(set(df["category_id"].tolist())) if len(df) else 0,
        "bad_counts": dict(bad),
        "clipped_bbox_count": int(df["clipped"].sum()) if len(df) else 0,
        "out_of_bounds_bbox_count": int(df["out_of_bounds"].sum()) if len(df) else 0,
    }
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Plots
    if len(df) == 0:
        print("No valid bboxes found. Check your COCO json & paths.")
        return

    # category frequency
    cat_counts = df["category_name"].value_counts()
    plot_bar(cat_counts, "Category frequency (TopK)", "category", "count",
             os.path.join(plots_dir, "category_frequency.png"), topk=30)

    # image W/H
    plot_scatter_wh(df.drop_duplicates("image_id"), os.path.join(plots_dir, "image_wh_scatter.png"))
    plot_hist(df.drop_duplicates("image_id")["img_w"], "Image width distribution", "W",
              os.path.join(plots_dir, "image_w_hist.png"), bins=60)
    plot_hist(df.drop_duplicates("image_id")["img_h"], "Image height distribution", "H",
              os.path.join(plots_dir, "image_h_hist.png"), bins=60)

    # per-image bbox count
    plot_hist(df.drop_duplicates("image_id")["num_boxes"], "BBoxes per image", "#boxes",
              os.path.join(plots_dir, "bboxes_per_image.png"), bins=60, logy=True)

    # bbox stats
    plot_hist(df["area"], "BBox area (px^2)", "area", os.path.join(plots_dir, "bbox_area.png"),
              bins=80, logy=True)
    plot_hist(df["area_ratio"], "BBox area ratio (area / image_area)", "area_ratio",
              os.path.join(plots_dir, "bbox_area_ratio.png"), bins=80, logy=True)
    plot_hist(df["aspect_ratio"], "BBox aspect ratio (w/h)", "w/h",
              os.path.join(plots_dir, "bbox_aspect_ratio.png"), bins=80, logy=True)
    plot_hist(df["w"], "BBox width (px)", "w", os.path.join(plots_dir, "bbox_w.png"),
              bins=80, logy=True)
    plot_hist(df["h"], "BBox height (px)", "h", os.path.join(plots_dir, "bbox_h.png"),
              bins=80, logy=True)

    # size bucket pie (as bar)
    bucket_counts = df["size_bucket"].value_counts()
    plot_bar(bucket_counts, "COCO size buckets", "bucket", "count",
             os.path.join(plots_dir, "size_buckets.png"), topk=10)

    # center heatmap
    plot_center_heatmap(df, os.path.join(plots_dir, "bbox_center_heatmap.png"), grid=64)

    # extra exports: per-category summary
    per_cat = df.groupby("category_name").agg(
        num_boxes=("ann_id", "count"),
        num_images=("image_id", "nunique"),
        mean_area=("area", "mean"),
        mean_area_ratio=("area_ratio", "mean"),
        mean_ar=("aspect_ratio", "mean"),
        small=("size_bucket", lambda x: (x == "small").sum()),
        medium=("size_bucket", lambda x: (x == "medium").sum()),
        large=("size_bucket", lambda x: (x == "large").sum()),
    ).reset_index()
    per_cat.to_csv(os.path.join(args.out, "per_category_summary.csv"), index=False)

    print(f"[OK] EDA outputs saved to: {args.out}")
    print("Key files: summary.json, bboxes.csv, per_category_summary.csv, plots/*.png")


if __name__ == "__main__":
    main()
