"""Checkpoint loading and image-saving helpers."""
import numpy as np
import torch
from PIL import Image

from config import Config
from model import build_model


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(path, device, use_ema=True):
    """Returns (model in eval mode, Config). Uses the EMA weights by default."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = Config.from_dict(ckpt["config"])
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["ema"] if use_ema and "ema" in ckpt else ckpt["model"])
    return model.eval(), cfg


def save_grid(rows, path):
    """rows: list of (C,H,W) tensors in [0,1] per column, one list per row -> PNG (first channel shown)."""
    tiles = [np.concatenate([t[0].cpu().numpy() for t in row], axis=1) for row in rows]
    img = (np.clip(np.concatenate(tiles, axis=0), 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img).save(path)
