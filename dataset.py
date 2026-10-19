"""Loading, normalizing and degrading microscopy images into (LR-upsampled, HR) training pairs."""
import os
import random

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

IMG_EXT = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


# ---------------------------------------------------------------- I/O + normalization
def list_images(folder):
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(IMG_EXT))


def read_image(path):
    """Returns float32 array: (H, W), (H, W, C) or, for TIFF stacks, (T, H, W)."""
    if path.lower().endswith((".tif", ".tiff")):
        arr = tifffile.imread(path)
    else:
        arr = np.array(Image.open(path))
    return arr.astype(np.float32)


def frames_of(arr, channels):
    """Split an array into a list of (C, H, W) frames."""
    if arr.ndim == 2:
        return [arr[None]]
    if arr.ndim == 3 and arr.shape[-1] == 1:                           # (H, W, 1) single-channel image
        return [arr[..., 0][None]]
    if arr.ndim == 3 and channels > 1 and arr.shape[-1] == channels:   # (H, W, C) colour image
        return [arr.transpose(2, 0, 1)]
    if arr.ndim == 3 and arr.shape[-1] in (3, 4) and channels == 1:   # RGB image used as 1 channel
        return [arr[..., :3].mean(-1)[None]]
    if arr.ndim == 3:                                                  # (T, H, W) stack of frames
        return [f[None] for f in arr]
    if arr.ndim == 4:                                                  # (T, C, H, W)
        return [f for f in arr]
    raise ValueError(f"Unsupported image shape {arr.shape}")


def percentile_normalize(x, p_low=1.0, p_high=99.8, eps=1e-8):
    """Map intensities so p_low -> 0 and p_high -> 1, then clip to [0, 1]. Works on numpy or torch."""
    if isinstance(x, torch.Tensor):
        flat = x.flatten().float()
        k = min(flat.numel(), 10_000_000)
        lo, hi = torch.quantile(flat[:k], torch.tensor([p_low / 100, p_high / 100], device=x.device))
        return ((x - lo) / (hi - lo + eps)).clamp(0, 1)
    lo, hi = np.percentile(x, [p_low, p_high])
    return np.clip((x - lo) / (hi - lo + eps), 0, 1)


def to_model_range(x01):
    return x01 * 2 - 1          # [0,1] -> [-1,1]


def from_model_range(xm):
    return ((xm + 1) / 2).clamp(0, 1)


# ---------------------------------------------------------------- degradation
def gaussian_blur(x, sigma):
    """x: (B, C, H, W). Separable Gaussian blur."""
    if sigma <= 0:
        return x
    radius = max(1, int(3 * sigma + 0.5))
    k = torch.arange(-radius, radius + 1, dtype=x.dtype, device=x.device)
    k = torch.exp(-(k ** 2) / (2 * sigma ** 2))
    k = (k / k.sum())
    c = x.shape[1]
    kx = k.view(1, 1, 1, -1).repeat(c, 1, 1, 1)
    ky = k.view(1, 1, -1, 1).repeat(c, 1, 1, 1)
    x = F.pad(x, (radius, radius, radius, radius), mode="reflect")
    x = F.conv2d(x, kx, groups=c)
    return F.conv2d(x, ky, groups=c)


def downsample(x, scale):
    """Area (box) downsampling = anti-aliased, matches camera pixel binning."""
    return F.avg_pool2d(x, scale)


def upsample_bicubic(x, scale=None, size=None):
    return F.interpolate(x, scale_factor=scale, size=size, mode="bicubic", align_corners=False)


def simulate_lr(hr01, cfg, gen=None):
    """hr01: (1, C, H, W) in [0,1]. Returns LR (1, C, H/scale, W/scale) in [0,1] with blur + Poisson + read noise."""
    rnd = gen.random if gen is not None else random.random
    sigma = rnd() * cfg.blur_sigma_max
    x = gaussian_blur(hr01, sigma)
    x = downsample(x, cfg.scale).clamp(0, 1)
    photons = cfg.photons_min + rnd() * (cfg.photons_max - cfg.photons_min)
    tgen = torch.Generator().manual_seed(int(rnd() * 2**31)) if gen is not None else None
    x = torch.poisson(x * photons, generator=tgen) / photons
    read = rnd() * cfg.read_noise_max
    x = x + read * torch.randn(x.shape, generator=tgen)
    return x.clamp(0, 1)


# ---------------------------------------------------------------- dataset
class SRDataset(Dataset):
    """
    Returns (x, y), both (C, P, P) in [-1, 1]:
      x = low-res patch, bicubic-upsampled back to P x P (the SR3 conditioning image)
      y = high-res target patch
    Paired mode (cfg.lr_dir set): LR files must have the same names and be exactly 1/scale the HR size.
    Simulated mode: LR is made from HR with blur + area downsampling + Poisson/read noise.
    """

    def __init__(self, cfg, files, train=True, hr_dir=None, lr_dir=None):
        self.cfg, self.train = cfg, train
        hr_dir = hr_dir or cfg.hr_dir
        lr_dir = cfg.lr_dir if lr_dir is None else lr_dir
        # LR already on the HR pixel grid (e.g. BioSR widefield): crop with step 1 and skip bicubic upsampling
        self.step = 1 if cfg.lr_upsampled else cfg.scale
        self.hr, self.lr = [], []
        for name in files:
            hr_frames = frames_of(read_image(os.path.join(hr_dir, name)), cfg.channels)
            lr_frames = (frames_of(read_image(os.path.join(lr_dir, name)), cfg.channels)
                         if lr_dir else [None] * len(hr_frames))
            for h, l in zip(hr_frames, lr_frames):
                if h.shape[-1] < cfg.patch_size or h.shape[-2] < cfg.patch_size:
                    continue
                h = percentile_normalize(h, cfg.p_low, cfg.p_high).astype(np.float32)
                if l is not None:
                    k = 1 if cfg.lr_upsampled else cfg.scale
                    assert l.shape[-2] * k == h.shape[-2] and l.shape[-1] * k == h.shape[-1], \
                        f"{name}: LR {l.shape} vs HR {h.shape}: expected HR / {k} (check scale / lr_upsampled)"
                    l = percentile_normalize(l, cfg.p_low, cfg.p_high).astype(np.float32)
                self.hr.append(h)
                self.lr.append(l)
        if not self.hr:
            raise RuntimeError("No usable images found (check hr_dir, file types and patch_size).")
        # validation: fixed patches, fixed noise -> comparable numbers between runs
        self.n = cfg.samples_per_epoch if train else max(len(self.hr), 16)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        cfg, P, s = self.cfg, self.cfg.patch_size, self.step
        gen = None if self.train else random.Random(idx)
        rnd = random if self.train else gen
        i = rnd.randrange(len(self.hr)) if self.train else idx % len(self.hr)
        hr, lr = self.hr[i], self.lr[i]
        H, W = hr.shape[-2:]
        # crop at LR-grid-aligned coordinates so paired LR/HR stay registered
        top = rnd.randrange(0, (H - P) // s + 1) * s
        left = rnd.randrange(0, (W - P) // s + 1) * s
        hr_p = torch.from_numpy(np.ascontiguousarray(hr[:, top:top + P, left:left + P]))[None]
        if lr is not None:
            lr_p = torch.from_numpy(np.ascontiguousarray(
                lr[:, top // s:(top + P) // s, left // s:(left + P) // s]))[None]
        else:
            lr_p = simulate_lr(hr_p, cfg, gen)
            if s == 1:  # simulated LR is always smaller; bring it back to the HR grid
                lr_p = upsample_bicubic(lr_p, scale=cfg.scale).clamp(0, 1)
        if self.train:  # flips + 90 degree rotations: safe for cells (no preferred orientation)
            k = random.randrange(4)
            flip = random.random() < 0.5
            hr_p, lr_p = torch.rot90(hr_p, k, (2, 3)), torch.rot90(lr_p, k, (2, 3))
            if flip:
                hr_p, lr_p = hr_p.flip(3), lr_p.flip(3)
        x = lr_p if s == 1 else upsample_bicubic(lr_p, scale=s).clamp(0, 1)
        return to_model_range(x[0]), to_model_range(hr_p[0])


def split_files(cfg, seed=0):
    files = list_images(cfg.hr_dir)
    if cfg.max_files:
        files = files[:cfg.max_files]
    if len(files) < 2:
        return files, files
    rng = random.Random(seed)
    rng.shuffle(files)
    n_val = max(1, int(round(len(files) * cfg.val_fraction)))
    return files[n_val:], files[:n_val]


def make_datasets(cfg, seed=0):
    """Train/val datasets. Uses val_hr_dir/val_lr_dir if given, else holds out val_fraction of the files."""
    train_files, val_files = split_files(cfg, seed)
    if cfg.val_hr_dir:
        train_files = list_images(cfg.hr_dir)[:cfg.max_files or None]
        val_files = list_images(cfg.val_hr_dir)
        val = SRDataset(cfg, val_files, train=False, hr_dir=cfg.val_hr_dir, lr_dir=cfg.val_lr_dir)
    else:
        val = SRDataset(cfg, val_files, train=False)
    return SRDataset(cfg, train_files, train=True), val
