#!/usr/bin/env python3
import os
import re
import json
import glob
import argparse
import numpy as np


def parse_action_from_filename(path: str):
    """
    Default rule for your filenames:
      101010_brooming_djr_p1_all_transformed_joints.npz
      -> action = brooming  (2nd token after subject)
    If your naming differs, adjust here.
    """
    name = os.path.basename(path)
    name = name.replace("_all_transformed_joints.npz", "")
    parts = name.split("_")
    if len(parts) < 2:
        raise ValueError(f"Cannot parse action from filename: {name}")
    return parts[1]


def pad_or_trunc(j: np.ndarray, T: int):
    """j: (t,22,3) -> (T,22,3)"""
    t = j.shape[0]
    if t == T:
        return j
    if t > T:
        return j[:T]
    # pad with last frame
    pad_n = T - t
    last = j[-1:, :, :]
    pad = np.repeat(last, pad_n, axis=0)
    return np.concatenate([j, pad], axis=0)


def one_hot(labels, num_classes):
    y = np.zeros((len(labels), num_classes), dtype=np.float32)
    y[np.arange(len(labels)), labels] = 1.0
    return y


def main():
    ap = argparse.ArgumentParser("Pack many *_all_transformed_joints.npz into SkateFormer-style dataset_22j.npz")
    ap.add_argument("--input_glob", required=True,
                    help="Glob for joints npz files, e.g. '/mnt/.../**/joint/npz/*_all_transformed_joints.npz'")
    ap.add_argument("--out_npz", required=True, help="Output dataset npz path")
    ap.add_argument("--T", type=int, default=100, help="Fixed temporal length (pad/truncate). Default=100")
    ap.add_argument("--test_ratio", type=float, default=0.2, help="Random split ratio for test")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--require_key", default="joints_3d", help="Which key to read as joints array (default joints_3d)")
    ap.add_argument("--save_label_map", action="store_true",
                    help="Also save label_map.json next to out_npz")
    args = ap.parse_args()

    files = sorted(glob.glob(args.input_glob, recursive=True))
    if not files:
        raise SystemExit(f"No files matched input_glob: {args.input_glob}")

    # Build label map
    actions = []
    for f in files:
        act = parse_action_from_filename(f)
        actions.append(act)

    uniq_actions = sorted(set(actions))
    label_map = {a: i for i, a in enumerate(uniq_actions)}
    num_classes = len(uniq_actions)

    print("[Info] Found files:", len(files))
    print("[Info] Num classes:", num_classes)
    print("[Info] Label map:")
    for a, i in label_map.items():
        print(f"  {i:2d} : {a}")

    # Load all data
    X = []
    Y = []
    bad = 0
    for f in files:
        z = np.load(f, allow_pickle=True)
        if args.require_key not in z:
            print("[Skip] missing key", args.require_key, "in", f, "keys=", z.files)
            bad += 1
            continue
        j = z[args.require_key]  # (t,22,3)
        if j.ndim != 3 or j.shape[1] != 22 or j.shape[2] != 3:
            print("[Skip] bad joints shape", j.shape, "in", f)
            bad += 1
            continue

        j = pad_or_trunc(j.astype(np.float32), args.T)
        act = parse_action_from_filename(f)
        lab = label_map[act]

        X.append(j)
        Y.append(lab)

    if not X:
        raise SystemExit("No valid samples loaded.")

    X = np.stack(X, axis=0)  # (N,T,22,3)
    Y = np.array(Y, dtype=np.int64)

    print("[Info] Loaded samples:", X.shape[0], "bad skipped:", bad)
    print("[Info] X shape:", X.shape, "dtype:", X.dtype)

    # Shuffle + split
    rng = np.random.default_rng(args.seed)
    idx = np.arange(X.shape[0])
    rng.shuffle(idx)

    n_test = int(round(X.shape[0] * args.test_ratio))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]

    x_train = X[train_idx]
    x_test = X[test_idx]

    y_train = one_hot(Y[train_idx], num_classes)
    y_test = one_hot(Y[test_idx], num_classes)

    # Save dataset
    os.makedirs(os.path.dirname(args.out_npz) or ".", exist_ok=True)
    np.savez_compressed(
        args.out_npz,
        x_train=x_train, y_train=y_train,
        x_test=x_test, y_test=y_test,
        class_names=np.array(uniq_actions, dtype=object),
    )

    print("[Done] Saved:", args.out_npz)
    print("[Done] x_train:", x_train.shape, "y_train:", y_train.shape)
    print("[Done] x_test :", x_test.shape, "y_test :", y_test.shape)

    if args.save_label_map:
        map_path = os.path.splitext(args.out_npz)[0] + "_label_map.json"
        with open(map_path, "w") as f:
            json.dump(label_map, f, indent=2)
        print("[Done] Saved label map:", map_path)


if __name__ == "__main__":
    main()
