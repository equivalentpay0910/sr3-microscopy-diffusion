"""Super-resolve real low-res images or video stacks with a trained checkpoint (tiled, any image size).
Usage:
  python sample.py --ckpt runs/exp1/last.pt --input path/to/lr.tif --out results/
  python sample.py --ckpt runs/exp1/last.pt --input path/to/lr_folder --out results/ --steps 50 --method ddim
  add --n_samples 8 to also save a per-pixel uncertainty (std) map and use the mean (DDPM-avg)
"""
import argparse
import os
import time

import numpy as np
import tifffile
import torch

from dataset import (frames_of, list_images, percentile_normalize, read_image, to_model_range,
                     from_model_range, upsample_bicubic)
from diffusion import inference_betas, sample
from utils import get_device, load_checkpoint


def tile_weights(t, overlap, device):
    """2D blending window: 1 in the middle, linear ramp to ~0 over the overlap at the edges."""
    ramp = torch.ones(t, device=device)
    if overlap > 0:
        r = torch.linspace(0, 1, overlap + 2, device=device)[1:-1]
        ramp[:overlap], ramp[-overlap:] = r, r.flip(0)
    return ramp[:, None] * ramp[None, :]


@torch.no_grad()
def super_resolve(model, cfg, lr01, args, device):
    """lr01: (C, h, w) in [0,1]. Returns SR mean and std, each (C, h*scale, w*scale) in [0,1]."""
    s = cfg.data.scale
    x = torch.from_numpy(lr01)[None].to(device)
    if not cfg.data.lr_upsampled:          # LR smaller than HR -> bicubic to the HR grid first
        x = upsample_bicubic(x, scale=s).clamp(0, 1)
    x = to_model_range(x)
    _, C, H, W = x.shape
    t, ov = args.tile, args.overlap
    stride = t - ov
    # pad so tiles cover the image exactly (reflect padding avoids edge artefacts)
    Hp = max(t, int(np.ceil((H - ov) / stride)) * stride + ov)
    Wp = max(t, int(np.ceil((W - ov) / stride)) * stride + ov)
    xp = torch.nn.functional.pad(x, (0, Wp - W, 0, Hp - H), mode="reflect" if (Wp - W < W and Hp - H < H) else "replicate")
    coords = [(i, j) for i in range(0, Hp - t + 1, stride) for j in range(0, Wp - t + 1, stride)]
    w = tile_weights(t, ov, device)
    acc = torch.zeros(args.n_samples, C, Hp, Wp, device=device)
    norm = torch.zeros(1, 1, Hp, Wp, device=device)
    sc = cfg.sample
    steps = args.steps or sc.steps
    b0, b1 = inference_betas(sc, steps)
    for b in range(0, len(coords), args.batch):
        chunk = coords[b:b + args.batch]
        tiles = torch.cat([xp[:, :, i:i + t, j:j + t] for i, j in chunk])
        for k in range(args.n_samples):
            sr = from_model_range(sample(model, tiles, steps=steps, beta_start=b0,
                                         beta_end=b1, method=args.method or sc.method,
                                         residual=cfg.diffusion.residual))
            for (i, j), tile in zip(chunk, sr):
                acc[k, :, i:i + t, j:j + t] += tile * w
        for i, j in chunk:
            norm[0, 0, i:i + t, j:j + t] += w
    out = (acc / norm.clamp_min(1e-8))[:, :, :H, :W]
    return out.mean(0).cpu().numpy(), out.std(0).cpu().numpy() if args.n_samples > 1 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input", required=True, help="LR image, TIFF stack (T,H,W), or folder of them")
    ap.add_argument("--out", default="results")
    ap.add_argument("--steps", type=int, default=0, help="0 = value from the checkpoint config")
    ap.add_argument("--method", default="", choices=["", "ddpm", "ddim"])
    ap.add_argument("--tile", type=int, default=0, help="HR tile side; 0 = training patch_size")
    ap.add_argument("--overlap", type=int, default=32)
    ap.add_argument("--batch", type=int, default=8, help="tiles per forward pass")
    ap.add_argument("--n_samples", type=int, default=1)
    ap.add_argument("--beta_start", type=float, default=0, help="override (as for 100 steps), e.g. from tune_schedule.py")
    ap.add_argument("--beta_end", type=float, default=0, help="override (as for 100 steps)")
    ap.add_argument("--no_ema", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()
    model, cfg = load_checkpoint(args.ckpt, device, use_ema=not args.no_ema)
    cfg.sample.beta_start = args.beta_start or cfg.sample.beta_start
    cfg.sample.beta_end = args.beta_end or cfg.sample.beta_end
    args.tile = args.tile or cfg.data.patch_size
    assert args.tile % model.factor == 0, f"--tile must be divisible by {model.factor}"
    os.makedirs(args.out, exist_ok=True)

    paths = ([os.path.join(args.input, f) for f in list_images(args.input)]
             if os.path.isdir(args.input) else [args.input])
    for path in paths:
        frames = frames_of(read_image(path), cfg.data.channels)
        means, stds, t0 = [], [], time.time()
        for f in frames:
            lr01 = percentile_normalize(f, cfg.data.p_low, cfg.data.p_high).astype(np.float32)
            m, s = super_resolve(model, cfg, lr01, args, device)
            means.append(m); stds.append(s)
        dt = time.time() - t0
        name = os.path.splitext(os.path.basename(path))[0]
        stack = np.stack(means).squeeze()                      # (T,[C,]H,W) or (H,W)
        tifffile.imwrite(os.path.join(args.out, f"{name}_sr.tif"), stack.astype(np.float32))
        if args.n_samples > 1:
            tifffile.imwrite(os.path.join(args.out, f"{name}_sr_std.tif"), np.stack(stds).squeeze().astype(np.float32))
        print(f"{name}: {len(frames)} frame(s) in {dt:.1f}s ({len(frames) / dt:.2f} frames/s) -> {args.out}")


if __name__ == "__main__":
    main()
