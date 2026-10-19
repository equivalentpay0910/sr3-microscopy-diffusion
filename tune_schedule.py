"""Search the inference noise schedule on validation data (SR3 trains once, then tunes the sampler).
Tries each (steps, beta_end) pair and ranks by the chosen metric. beta values are "as for 100 steps".
Usage: python tune_schedule.py --ckpt runs/exp1/last.pt --steps 25 50 100 --beta_end 0.01 0.02 0.05 0.1 0.2
Then pass the winner to sample.py / evaluate.py with --steps and --beta_end.
"""
import argparse

import numpy as np
import torch

from config import SampleConfig
from diffusion import inference_betas
from evaluate import run_eval
from utils import get_device, load_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--steps", type=int, nargs="+", default=[25, 50, 100])
    ap.add_argument("--beta_end", type=float, nargs="+", default=[0.01, 0.02, 0.05, 0.1, 0.2])
    ap.add_argument("--beta_start", type=float, default=1e-4)
    ap.add_argument("--method", default="ddpm")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--metric", default="psnr", choices=["psnr", "ssim", "consistency"])
    args = ap.parse_args()

    device = get_device()
    model, cfg = load_checkpoint(args.ckpt, device)
    col = {"psnr": 4, "ssim": 5, "consistency": 7}[args.metric]
    results = []
    for st in args.steps:
        for be in args.beta_end:
            torch.manual_seed(0)
            b0, b1 = inference_betas(SampleConfig(beta_start=args.beta_start, beta_end=be), st)
            rows, _ = run_eval(model, cfg, device, st, b0, b1, args.method, n=args.n)
            m = np.array(rows).mean(0)
            results.append((st, be, m))
            print(f"steps {st:4d} beta_end {be:<6} | PSNR {m[4]:6.2f} SSIM {m[5]:.3f} consistency {m[7]:.2e} "
                  f"(input PSNR {m[0]:.2f})", flush=True)
    best = sorted(results, key=lambda r: r[2][col], reverse=args.metric != "consistency")[0]
    print(f"\nbest by {args.metric}: --steps {best[0]} --beta_end {best[1]}")


if __name__ == "__main__":
    main()
