"""Evaluate a checkpoint: the model input (bicubic or widefield) vs SR3, on the validation set or any paired folder.
Metrics: PSNR, SSIM, NRMSE (vs HR) and consistency (MSE between downsampled output and LR input, SR3 Table 1).
Usage: python evaluate.py --ckpt runs/exp1/last.pt [--n 64] [--steps 100] [--method ddpm] [--n_samples 1]
"""
import argparse
import csv
import os

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from dataset import SRDataset, downsample, from_model_range, list_images, make_datasets
from diffusion import inference_betas, sample
from utils import get_device, load_checkpoint, save_grid


def metrics(pred, hr, lr_up, scale):
    """pred, hr, lr_up: (C,H,W) numpy in [0,1]."""
    p, h = pred[0], hr[0]
    psnr = peak_signal_noise_ratio(h, p, data_range=1.0)
    ssim = structural_similarity(h, p, data_range=1.0)
    nrmse = np.sqrt(np.mean((p - h) ** 2)) / (h.max() - h.min() + 1e-8)
    # consistency: downsample prediction and compare with the (downsampled) conditioning image
    pd = downsample(torch.from_numpy(pred)[None], scale)[0].numpy()
    ld = downsample(torch.from_numpy(lr_up)[None], scale)[0].numpy()
    consistency = np.mean((pd - ld) ** 2)
    return psnr, ssim, nrmse, consistency


@torch.no_grad()
def run_eval(model, cfg, device, steps, b0, b1, method, n=64, n_samples=1, batch=8, hr_dir="", lr_dir=""):
    """Returns (rows, preview). rows: per-patch [input x4 metrics, sr3 x4 metrics].
    Default: the validation set. hr_dir/lr_dir: evaluate on another paired folder (e.g. a test noise level)."""
    if hr_dir:
        ds = SRDataset(cfg.data, list_images(hr_dir)[:n], train=False, hr_dir=hr_dir, lr_dir=lr_dir)
    else:
        _, ds = make_datasets(cfg.data, cfg.train.seed)
    n = min(n, len(ds))
    rows, preview = [], []
    for b in range(0, n, batch):
        xs, ys = zip(*[ds[i] for i in range(b, min(n, b + batch))])
        x, y = torch.stack(xs).to(device), torch.stack(ys).to(device)
        srs = torch.stack([sample(model, x, steps=steps, beta_start=b0, beta_end=b1, method=method,
                                  residual=cfg.diffusion.residual)
                           for _ in range(n_samples)]).mean(0)
        for xi, yi, si in zip(x, y, srs):
            lr_up, hr, sr = (from_model_range(t).cpu().numpy() for t in (xi, yi, si))
            rows.append(list(metrics(lr_up, hr, lr_up, cfg.data.scale)) + list(metrics(sr, hr, lr_up, cfg.data.scale)))
            if len(preview) < 8:
                preview.append([torch.from_numpy(lr_up), torch.from_numpy(sr), torch.from_numpy(hr)])
    return rows, preview


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=64, help="number of validation patches")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--method", default="")
    ap.add_argument("--beta_start", type=float, default=0, help="override (as for 100 steps)")
    ap.add_argument("--beta_end", type=float, default=0, help="override (as for 100 steps)")
    ap.add_argument("--n_samples", type=int, default=1, help=">1 averages several samples (DDPM-avg)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--hr_dir", default="", help="evaluate on this HR folder instead of the validation set")
    ap.add_argument("--lr_dir", default="", help="matching LR folder (same file names)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    torch.manual_seed(0)
    device = get_device()
    model, cfg = load_checkpoint(args.ckpt, device)
    sc = cfg.sample
    sc.beta_start = args.beta_start or sc.beta_start
    sc.beta_end = args.beta_end or sc.beta_end
    steps = args.steps or sc.steps
    b0, b1 = inference_betas(sc, steps)
    out = args.out or os.path.join(os.path.dirname(args.ckpt), "eval")
    os.makedirs(out, exist_ok=True)

    rows, preview = run_eval(model, cfg, device, steps, b0, b1, args.method or sc.method,
                             args.n, args.n_samples, args.batch, args.hr_dir, args.lr_dir)
    header = ["input_psnr", "input_ssim", "input_nrmse", "input_consistency",
              "sr3_psnr", "sr3_ssim", "sr3_nrmse", "sr3_consistency"]
    with open(os.path.join(out, "metrics.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    save_grid(preview, os.path.join(out, "preview.png"))   # LR-upsampled | SR3 | HR
    m = np.array(rows).mean(0)
    print(f"\n{'':10}{'PSNR':>8}{'SSIM':>8}{'NRMSE':>8}{'Consist.':>12}")
    print(f"{'Input':10}{m[0]:8.2f}{m[1]:8.3f}{m[2]:8.3f}{m[3]:12.2e}")
    print(f"{'SR3':10}{m[4]:8.2f}{m[5]:8.3f}{m[6]:8.3f}{m[7]:12.2e}")
    print(f"per-patch results: {out}/metrics.csv, preview: {out}/preview.png")


if __name__ == "__main__":
    main()
