"""Train SR3 on microscopy images.
Usage: python train.py --config configs/default.json [--resume runs/exp1/last.pt]
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config
from dataset import from_model_range, make_datasets
from diffusion import EMA, Schedule, inference_betas, sample, training_loss
from model import build_model
from utils import get_device, save_grid


def infinite(loader):
    while True:
        for batch in loader:
            yield batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="")
    ap.add_argument("--resume", default="")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    tc = cfg.train
    random.seed(tc.seed); np.random.seed(tc.seed); torch.manual_seed(tc.seed)
    os.makedirs(os.path.join(tc.out_dir, "samples"), exist_ok=True)
    with open(os.path.join(tc.out_dir, "config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    device = get_device()
    use_amp = tc.amp and device.type == "cuda"
    print(f"device={device} amp={use_amp}")

    # ---- data (split by file so validation cells never appear in training)
    train_ds, val_ds = make_datasets(cfg.data, tc.seed)
    print(f"train frames={len(train_ds.hr)} | val frames={len(val_ds.hr)}")
    loader = DataLoader(train_ds, batch_size=tc.batch_size, shuffle=True, drop_last=True,
                        num_workers=tc.num_workers, pin_memory=device.type == "cuda",
                        persistent_workers=tc.num_workers > 0)
    val_x, val_y = zip(*[val_ds[i] for i in range(min(4, len(val_ds)))])
    val_x, val_y = torch.stack(val_x).to(device), torch.stack(val_y).to(device)

    # ---- model, optimizer, EMA, schedule
    model = build_model(cfg).to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    opt = torch.optim.Adam(model.parameters(), lr=tc.lr)
    warmup = lambda step: min(1.0, (step + 1) / max(1, tc.warmup_steps))   # linear warmup, then constant (SR3)
    sched_lr = torch.optim.lr_scheduler.LambdaLR(opt, warmup)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ema = EMA(model, tc.ema_decay)
    sched = Schedule(cfg.diffusion.timesteps, cfg.diffusion.beta_start, cfg.diffusion.beta_end, device=device)

    step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); ema.model.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"]); sched_lr.load_state_dict(ck["sched"]); step = ck["step"]
        print(f"resumed at step {step}")

    def save(name):
        torch.save({"model": model.state_dict(), "ema": ema.model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched_lr.state_dict(), "step": step, "config": cfg.to_dict()},
                   os.path.join(tc.out_dir, name))

    # ---- loop
    batches, losses, t0 = infinite(loader), [], time.time()
    model.train()
    while step < tc.total_steps:
        x, y = next(batches)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            loss = training_loss(model, sched, x, y, cfg.diffusion.loss, cfg.diffusion.residual)
        if not math.isfinite(loss.item()):
            raise RuntimeError(f"loss became {loss.item()} at step {step}; lower lr or disable amp")
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        scaler.step(opt); scaler.update(); sched_lr.step()
        ema.update(model, step)
        step += 1
        losses.append(loss.item())

        if step % tc.log_every == 0:
            dt = time.time() - t0
            print(f"step {step:>7} | loss {np.mean(losses):.4f} | lr {sched_lr.get_last_lr()[0]:.2e} | "
                  f"{tc.log_every / dt:.1f} it/s", flush=True)
            losses, t0 = [], time.time()

        if step % tc.sample_every == 0 or step == tc.total_steps:
            sc = cfg.sample
            b0, b1 = inference_betas(sc, sc.steps)
            sr = sample(ema.model, val_x, steps=sc.steps, beta_start=b0, beta_end=b1,
                        method=sc.method, residual=cfg.diffusion.residual)
            rows = [[from_model_range(a), from_model_range(b), from_model_range(c)]
                    for a, b, c in zip(val_x, sr, val_y)]
            save_grid(rows, os.path.join(tc.out_dir, "samples", f"step_{step:07d}.png"))   # LR | SR | HR
            print(f"saved preview samples/step_{step:07d}.png")

        if step % tc.ckpt_every == 0 or step == tc.total_steps:
            save("last.pt")
            save(f"step_{step:07d}.pt")
    print("done")


if __name__ == "__main__":
    main()
