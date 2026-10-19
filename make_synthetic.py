"""Make fake microscopy-like 16-bit TIFFs (filaments + puncta) to smoke-test the pipeline before real data.
Usage: python make_synthetic.py --out data/synthetic --n 40 --size 256
Also writes data/synthetic_lr_stack.tif: a 5-frame low-res stack for testing sample.py.
"""
import argparse
import os

import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter


def fake_cell(size, rng):
    img = np.zeros((size, size), np.float32)
    for _ in range(rng.integers(8, 20)):                      # filaments: smooth random walks
        pos, ang = rng.uniform(0, size, 2), rng.uniform(0, 2 * np.pi)
        for _ in range(rng.integers(100, 400)):
            ang += rng.normal(0, 0.08)
            pos = pos + np.array([np.cos(ang), np.sin(ang)])
            i, j = int(pos[0]) % size, int(pos[1]) % size
            img[i, j] += 1.0
    for _ in range(rng.integers(20, 80)):                     # puncta (vesicles)
        i, j = rng.integers(0, size, 2)
        img[i, j] += rng.uniform(3, 8)
    img = gaussian_filter(img, 1.0) + 0.02 * gaussian_filter(rng.random((size, size)), 8)
    return (img / img.max() * 60000).astype(np.uint16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--scale", type=int, default=4)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    os.makedirs(args.out, exist_ok=True)
    for k in range(args.n):
        tifffile.imwrite(os.path.join(args.out, f"cell_{k:03d}.tif"), fake_cell(args.size, rng))
    s = args.scale
    stack = np.stack([fake_cell(args.size, rng).reshape(args.size // s, s, args.size // s, s).mean((1, 3))
                      for _ in range(5)]).astype(np.uint16)
    tifffile.imwrite(os.path.join(os.path.dirname(args.out.rstrip("/")) or ".", "synthetic_lr_stack.tif"), stack)
    print(f"wrote {args.n} HR images to {args.out} and a 5-frame LR stack {stack.shape}")


if __name__ == "__main__":
    main()
