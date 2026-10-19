"""Configuration: every setting lives here. Override with a JSON file (see configs/)."""
import json
from dataclasses import dataclass, field, asdict


@dataclass
class DataConfig:
    hr_dir: str = "data/hr"            # folder of high-res images (.tif/.tiff/.png/.jpg); 3D TIFFs = stacks of frames
    lr_dir: str = ""                   # folder of real low-res images with the SAME file names; "" = simulate LR from HR
    lr_upsampled: bool = False         # True if LR images are already the same size as HR (e.g. BioSR widefield)
    val_hr_dir: str = ""               # optional separate validation folders; "" = hold out val_fraction of files
    val_lr_dir: str = ""
    max_files: int = 0                 # 0 = use all; >0 = only the first N files (quick tests, low RAM)
    channels: int = 1                  # 1 for a single fluorescence channel
    scale: int = 4                     # magnification factor (HR side / LR side)
    patch_size: int = 128              # HR training patch side; must be divisible by scale and by 2**(len(channel_mults)-1)
    val_fraction: float = 0.1          # fraction of FILES held out for validation (split by file = by sample, no leakage)
    p_low: float = 1.0                 # percentile normalization, lower bound
    p_high: float = 99.8               # percentile normalization, upper bound
    # Simulated degradation (used only when lr_dir == "")
    blur_sigma_max: float = 1.0        # random Gaussian blur on HR before downsampling (SR3's blur augmentation, PSF mismatch)
    photons_min: float = 50.0          # Poisson shot noise: fewer photons = noisier
    photons_max: float = 1000.0
    read_noise_max: float = 0.02       # Gaussian camera read noise (in [0,1] intensity units)
    samples_per_epoch: int = 2000      # random patches drawn per "epoch" of the training loader


@dataclass
class ModelConfig:
    base_channels: int = 64
    channel_mults: list = field(default_factory=lambda: [1, 2, 4, 8])
    num_res_blocks: int = 2
    attn_levels: list = field(default_factory=lambda: [3])   # which resolution levels get self-attention (0 = full res)
    dropout: float = 0.0


@dataclass
class DiffusionConfig:
    timesteps: int = 2000              # training T (SR3: 2000)
    beta_start: float = 1e-6           # training schedule (linear)
    beta_end: float = 1e-2
    loss: str = "l1"                   # "l1" (SR3 ablation winner) or "l2"
    residual: bool = True              # diffuse HR - bicubic(LR) instead of HR (SRDiff); False = plain SR3


@dataclass
class TrainConfig:
    out_dir: str = "runs/exp1"
    batch_size: int = 16
    lr: float = 1e-4
    warmup_steps: int = 10000
    total_steps: int = 500000
    ema_decay: float = 0.9999
    grad_clip: float = 1.0
    num_workers: int = 4
    log_every: int = 100
    sample_every: int = 5000           # save a preview grid (LR | SR | HR)
    ckpt_every: int = 10000
    seed: int = 0
    amp: bool = True                   # mixed precision on GPU


@dataclass
class SampleConfig:
    steps: int = 100                   # inference steps (SR3: <= 100)
    beta_start: float = 1e-4           # inference schedule AS FOR 100 STEPS; auto-scaled by 100/steps
    beta_end: float = 0.1              # (tune on validation data; SR3 picked its schedule by FID)
    method: str = "ddpm"               # "ddpm" (stochastic, Algorithm 2) or "ddim" (deterministic, fewer steps)


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    sample: SampleConfig = field(default_factory=SampleConfig)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(
            data=DataConfig(**d.get("data", {})),
            model=ModelConfig(**d.get("model", {})),
            diffusion=DiffusionConfig(**d.get("diffusion", {})),
            train=TrainConfig(**d.get("train", {})),
            sample=SampleConfig(**d.get("sample", {})),
        )

    @classmethod
    def load(cls, path):
        if not path:
            return cls()
        with open(path) as f:
            return cls.from_dict(json.load(f))
