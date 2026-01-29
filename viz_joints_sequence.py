#!/usr/bin/env python3
import os
import re
import argparse
import numpy as np

# headless friendly
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def extract_idx(path: str):
    nums = re.findall(r"\d+", os.path.basename(path))
    return int(nums[-1]) if nums else None


def edges_from_kin_parent(kin_parent):
    kp = np.asarray(kin_parent).astype(int)
    J = kp.shape[0]
    edges = []
    for c in range(J):
        p = kp[c]
        if 0 <= p < J and p != c:
            edges.append((p, c))
    return edges


def load_joints_seq_from_dir(joints_dir: str):
    files = [os.path.join(joints_dir, f) for f in os.listdir(joints_dir) if f.endswith(".npy")]
    if not files:
        raise FileNotFoundError(f"No .npy files found in {joints_dir}")

    tmp = []
    for f in files:
        idx = extract_idx(f)
        tmp.append((idx if idx is not None else 10**18, f))
    files = [f for _, f in sorted(tmp, key=lambda x: x[0])]

    frames = []
    for f in files:
        a = np.load(f)
        a = np.asarray(a)
        if a.ndim != 2 or a.shape[1] != 3:
            raise ValueError(f"{f} must be (J,3). Got {a.shape}")
        frames.append(a)
    return np.stack(frames, axis=0), files  # (T,J,3)


def load_joints_seq_from_npy(path: str):
    a = np.load(path)
    a = np.asarray(a)
    if a.ndim == 2 and a.shape[1] == 3:
        return a[None, ...]  # (1,J,3)
    if a.ndim == 3 and a.shape[2] == 3:
        return a
    raise ValueError(f"{path} must be (J,3) or (T,J,3). Got {a.shape}")


def load_scene_points(scene_ply_path: str, max_points: int = 20000, prefer_vertices=True):
    """
    Load points from a scene ply/mesh and downsample to max_points.
    Works for mesh or pointcloud ply.
    """
    import trimesh

    scene = trimesh.load(scene_ply_path, force="mesh")
    pts = None

    if isinstance(scene, trimesh.Scene):
        verts = []
        for _, g in scene.geometry.items():
            if hasattr(g, "vertices") and len(g.vertices) > 0:
                verts.append(np.asarray(g.vertices))
        if not verts:
            raise ValueError(f"No vertices found in scene file: {scene_ply_path}")
        pts = np.concatenate(verts, axis=0)
    else:
        if hasattr(scene, "vertices") and len(scene.vertices) > 0 and prefer_vertices:
            pts = np.asarray(scene.vertices)
        else:
            if hasattr(scene, "faces") and len(scene.faces) > 0:
                pts, _ = trimesh.sample.sample_surface(scene, count=max_points)
            else:
                raise ValueError(f"Unsupported scene type / empty vertices: {type(scene)}")

    if pts is None or len(pts) == 0:
        raise ValueError(f"Failed to extract points from: {scene_ply_path}")

    if pts.shape[0] > max_points:
        idx = np.random.choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[idx]
    return pts.astype(np.float32)


def _apply_axis_order(P, order: str):
    """
    Reorder axes for points P.
    order like: xyz/xzy/yzx/zyx...
    """
    if order is None:
        return P
    order = order.strip().lower()
    if order not in {"xyz", "xzy", "yxz", "yzx", "zxy", "zyx"}:
        raise ValueError(f"Invalid axis order: {order}")
    m = {'x': 0, 'y': 1, 'z': 2}
    idx = [m[c] for c in order]
    return P[:, idx]


def _apply_flip(P, flip: str):
    """
    Flip axes sign.
    flip examples: "-y" or "-x,-z"
    """
    if flip is None:
        return P
    flip = flip.replace(" ", "")
    parts = flip.split(",")
    for s in parts:
        if s == "-x":
            P[:, 0] *= -1
        elif s == "-y":
            P[:, 1] *= -1
        elif s == "-z":
            P[:, 2] *= -1
        elif s == "":
            continue
        else:
            raise ValueError(f"Invalid flip token: '{s}'. Use like -x or -y or -z or '-x,-z'")
    return P


def set_axes_equal(ax, X, Y, Z):
    x_min, x_max = float(X.min()), float(X.max())
    y_min, y_max = float(Y.min()), float(Y.max())
    z_min, z_max = float(Z.min()), float(Z.max())

    cx, cy, cz = (x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2
    r = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2
    if r == 0:
        r = 1.0
    ax.set_xlim(cx - r, cx + r)
    ax.set_ylim(cy - r, cy + r)
    ax.set_zlim(cz - r, cz + r)


def draw_frame(ax, J3, edges=None, title="", scene_pts=None, scene_stride=1):
    if scene_pts is not None and len(scene_pts) > 0:
        sp = scene_pts[::max(1, int(scene_stride))]
        ax.scatter(sp[:, 0], sp[:, 1], sp[:, 2], s=1, alpha=0.25)

    X, Y, Z = J3[:, 0], J3[:, 1], J3[:, 2]
    ax.scatter(X, Y, Z, s=18)

    if edges:
        for p, c in edges:
            ax.plot([J3[p, 0], J3[c, 0]],
                    [J3[p, 1], J3[c, 1]],
                    [J3[p, 2], J3[c, 2]],
                    linewidth=1.8)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)


def main():
    ap = argparse.ArgumentParser(description="Visualize (transformed) joints sequence for sanity check.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--joints_dir", help="Directory containing per-frame joints .npy (each (J,3))")
    src.add_argument("--joints_npy", help="A .npy file of shape (T,J,3) or (J,3)")

    ap.add_argument("--out_dir", required=True, help="Output directory for images (frames/)")
    ap.add_argument("--kin_parent", default=None, help="Optional kin_parent.npy to draw skeleton edges automatically")

    # scene overlay
    ap.add_argument("--scene_ply", default=None, help="Optional scene ply/mesh to visualize together")
    ap.add_argument("--scene_max_points", type=int, default=20000, help="Max points to draw from scene (default 20000)")
    ap.add_argument("--scene_draw_stride", type=int, default=1, help="Extra stride when drawing scene points (default 1)")

    # axis fix (NEW)
    ap.add_argument("--swap_scene_axes", default=None,
                    help="Reorder scene axes, e.g. xyz/xzy/yxz/yzx/zxy/zyx")
    ap.add_argument("--swap_joints_axes", default=None,
                    help="Reorder joints axes, e.g. xyz/xzy/yxz/yzx/zxy/zyx")
    ap.add_argument("--flip_scene", default=None,
                    help="Flip scene axes signs, e.g. -x or -y or -z or '-x,-z'")
    ap.add_argument("--flip_joints", default=None,
                    help="Flip joints axes signs, e.g. -x or -y or -z or '-x,-z'")

    ap.add_argument("--start", type=int, default=0, help="Start frame index (default 0)")
    ap.add_argument("--end", type=int, default=-1, help="End frame index inclusive (default -1 means last)")
    ap.add_argument("--step", type=int, default=1, help="Step/stride (default 1)")
    ap.add_argument("--view", type=str, default="30,45", help="Matplotlib view: elev,azim (default 30,45)")
    ap.add_argument("--dpi", type=int, default=150, help="PNG dpi (default 150)")
    ap.add_argument("--make_gif", action="store_true", help="Also create an animated gif (requires imageio)")
    ap.add_argument("--gif_fps", type=int, default=10, help="GIF fps (default 10)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    frames_dir = os.path.join(args.out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    # load joints sequence
    if args.joints_dir:
        joints_seq, files = load_joints_seq_from_dir(args.joints_dir)
        src_name = args.joints_dir
    else:
        joints_seq = load_joints_seq_from_npy(args.joints_npy)
        files = None
        src_name = args.joints_npy

    # axis fix for joints
    if args.swap_joints_axes:
        joints_seq = np.stack([_apply_axis_order(f, args.swap_joints_axes) for f in joints_seq], axis=0)
    if args.flip_joints:
        joints_seq = np.stack([_apply_flip(f.copy(), args.flip_joints) for f in joints_seq], axis=0)

    T, J, _ = joints_seq.shape
    print(f"[Load] joints_seq: shape={joints_seq.shape} from {src_name}")
    print(f"[Stats] min={joints_seq.min():.4f} max={joints_seq.max():.4f}")
    centers = joints_seq.mean(axis=1)  # (T,3)
    print(f"[Center] first={centers[0]} last={centers[-1]} delta={centers[-1]-centers[0]}")

    # edges from kin_parent
    edges = None
    if args.kin_parent:
        kp = np.load(args.kin_parent).astype(np.int32)
        if kp.shape[0] != J:
            raise ValueError(f"kin_parent length {kp.shape[0]} != joints J {J}. (joint order mismatch)")
        edges = edges_from_kin_parent(kp)
        print(f"[Skeleton] edges={len(edges)} from kin_parent")

    # scene points
    scene_pts = None
    if args.scene_ply:
        scene_pts = load_scene_points(args.scene_ply, max_points=args.scene_max_points)
        # axis fix for scene
        if args.swap_scene_axes:
            scene_pts = _apply_axis_order(scene_pts, args.swap_scene_axes)
        if args.flip_scene:
            scene_pts = _apply_flip(scene_pts, args.flip_scene)
        print(f"[Scene] loaded points: {scene_pts.shape} from {args.scene_ply}")

    # frame range
    start = max(0, args.start)
    end = (T - 1) if args.end < 0 else min(T - 1, args.end)
    step = max(1, args.step)

    elev, azim = (30.0, 45.0)
    try:
        elev, azim = map(float, args.view.split(","))
    except Exception:
        pass

    # global bounds for consistent camera (include scene)
    allX = joints_seq[:, :, 0].reshape(-1)
    allY = joints_seq[:, :, 1].reshape(-1)
    allZ = joints_seq[:, :, 2].reshape(-1)
    if scene_pts is not None:
        allX = np.concatenate([allX, scene_pts[:, 0]])
        allY = np.concatenate([allY, scene_pts[:, 1]])
        allZ = np.concatenate([allZ, scene_pts[:, 2]])

    saved = []
    for t in range(start, end + 1, step):
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(elev=elev, azim=azim)

        title = f"frame {t:06d}  J={J}"
        if files:
            title += f"  ({os.path.basename(files[t])})"

        draw_frame(
            ax,
            joints_seq[t],
            edges=edges,
            title=title,
            scene_pts=scene_pts,
            scene_stride=args.scene_draw_stride
        )

        set_axes_equal(ax, allX, allY, allZ)

        out_path = os.path.join(frames_dir, f"{t:06d}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=args.dpi)
        plt.close(fig)
        saved.append(out_path)

    print(f"[Done] saved {len(saved)} frames to: {frames_dir}")

    # optional gif
    if args.make_gif:
        try:
            import imageio.v2 as imageio
            gif_path = os.path.join(args.out_dir, "preview.gif")
            imgs = [imageio.imread(p) for p in saved]
            imageio.mimsave(gif_path, imgs, fps=args.gif_fps)
            print(f"[GIF] saved: {gif_path}")
        except Exception as e:
            print("[GIF] failed (need imageio):", e)
            print("Tip: use ffmpeg to make mp4/gif from frames.")


if __name__ == "__main__":
    main()
