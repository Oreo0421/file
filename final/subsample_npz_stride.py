#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Subsample motion npz by temporal stride.

Example:
  input  : 01_newformat_rooted.npz (N frames)
  output : 01_newformat_rooted_stride5.npz (ceil(N/5) frames)

Rule:
  - Detect frame count automatically
  - Take every k-th frame (default k=5)
  - Apply to all array-like keys with first dim == N
  - Leave other keys unchanged
"""

import argparse
import numpy as np
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="input motion npz")
    ap.add_argument("--stride", type=int, default=5, help="temporal stride (default=5)")
    ap.add_argument("--out", default=None, help="output npz path")
    args = ap.parse_args()

    data = np.load(args.input, allow_pickle=True)

    # -----------------------------
    # infer number of frames
    # -----------------------------
    N = None
    for k in data.files:
        v = data[k]
        if isinstance(v, np.ndarray) and v.ndim >= 1:
            N = v.shape[0]
            break

    if N is None:
        raise RuntimeError("Cannot infer frame count from npz")

    idx = np.arange(0, N, args.stride)

    print(f"Input frames : {N}")
    print(f"Stride       : {args.stride}")
    print(f"Output frames: {len(idx)}")

    out_data = {}

    for k in data.files:
        v = data[k]

        # subsample arrays that are frame-aligned
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == N:
            out_data[k] = v[idx]
        else:
            # keep metadata as-is
            out_data[k] = v

    out_path = args.out
    if out_path is None:
        p = Path(args.input)
        out_path = p.with_name(p.stem + f"_stride{args.stride}.npz")

    np.savez(out_path, **out_data)

    print("Saved:", out_path)


if __name__ == "__main__":
    main()
