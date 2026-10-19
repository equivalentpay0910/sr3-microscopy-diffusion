"""SR3 diffusion: noise schedule, continuous-gamma training loss, DDPM/DDIM sampling with any schedule, EMA."""
import copy

import torch
import torch.nn.functional as F


_warned = False


class Schedule:
    """Linear beta schedule and the derived alpha, gamma (= alpha-bar) quantities, as float64 for accuracy."""

    def __init__(self, n_steps, beta_start, beta_end, device="cpu"):
        self.n = n_steps
        self.betas = torch.linspace(beta_start, beta_end, n_steps, dtype=torch.float64, device=device)
        self.alphas = 1 - self.betas
        self.gammas = torch.cumprod(self.alphas, 0)                                   # gamma_1 .. gamma_T
        self.gammas_prev = torch.cat([torch.ones(1, dtype=torch.float64, device=device), self.gammas[:-1]])


def training_loss(model, sched, x, y0, loss="l1", residual=True):
    """
    SR3 Algorithm 1 with continuous gamma: t ~ U{1..T}, gamma ~ U(gamma_t, gamma_{t-1}).
    x: upsampled LR (B,C,H,W) in [-1,1]; y0: HR target in [-1,1].
    residual=True diffuses (HR - bicubic) instead of HR (SRDiff trick): the target is near zero-mean,
    so the model only has to generate missing detail. Much faster to learn on dark microscopy images.
    """
    if residual:
        y0 = y0 - x
    B = y0.shape[0]
    t = torch.randint(0, sched.n, (B,), device=y0.device)
    lo, hi = sched.gammas[t], sched.gammas_prev[t]
    gamma = (lo + (hi - lo) * torch.rand(B, dtype=torch.float64, device=y0.device)).float()
    noise = torch.randn_like(y0)
    g = gamma.view(-1, 1, 1, 1)
    y_noisy = g.sqrt() * y0 + (1 - g).sqrt() * noise
    pred = model(torch.cat([x, y_noisy], dim=1), gamma)
    return F.l1_loss(pred, noise) if loss == "l1" else F.mse_loss(pred, noise)


@torch.no_grad()
def sample(model, x, steps=100, beta_start=1e-4, beta_end=0.1, method="ddpm", clip=True, generator=None,
           residual=True):
    """
    SR3 Algorithm 2 with an inference schedule that can differ from training (possible because the model
    is conditioned on gamma, not t). x: upsampled LR in [-1,1]. Returns SR in [-1,1].
    method="ddpm": stochastic posterior sampling; "ddim": deterministic (eta=0), works with fewer steps.
    clip=True clamps the predicted clean image to its valid range each step (standard trick for stability).
    residual=True must match training: the chain generates (HR - bicubic), added to x at the end.
    """
    lo, hi = (-2.0, 2.0) if residual else (-1.0, 1.0)
    dev = x.device
    s = Schedule(steps, beta_start, beta_end, device=dev)
    global _warned
    if float(s.gammas[-1]) > 0.02 and not _warned:
        _warned = True
        print(f"warning: inference schedule ends at gamma_T={float(s.gammas[-1]):.3f} (should be ~0); "
              f"fine if tune_schedule.py chose it, otherwise raise beta_end")
    y = torch.randn(x.shape, device=dev, generator=generator)
    for t in reversed(range(steps)):
        g, g_prev, a, b = s.gammas[t], s.gammas_prev[t], s.alphas[t], s.betas[t]
        gamma = torch.full((x.shape[0],), float(g), device=dev)
        eps = model(torch.cat([x, y], dim=1), gamma).double()
        yd = y.double()
        y0_hat = (yd - (1 - g).sqrt() * eps) / g.sqrt()             # SR3 eq. (10)
        if clip:
            y0_hat = y0_hat.clamp(lo, hi)
        if method == "ddim":
            eps = (yd - g.sqrt() * y0_hat) / (1 - g).sqrt()          # re-derive noise after clipping
            y = (g_prev.sqrt() * y0_hat + (1 - g_prev).sqrt() * eps).float()
        else:
            # posterior q(y_{t-1} | y0_hat, y_t), SR3 eq. (4)
            mean = (g_prev.sqrt() * b / (1 - g)) * y0_hat + (a.sqrt() * (1 - g_prev) / (1 - g)) * yd
            var = b * (1 - g_prev) / (1 - g)
            y = mean.float()
            if t > 0:
                y = y + var.sqrt().float() * torch.randn(y.shape, device=dev, generator=generator)
    return (x + y if residual else y).clamp(-1, 1)


def inference_betas(sc, steps):
    """Beta range for a given step count. The configured beta_start/beta_end are for a 100-step schedule;
    they are scaled by 100/steps so the total noise (and gamma_T ~ 0) stays the same for 25, 50, 200 steps."""
    k = 100 / steps
    return sc.beta_start * k, min(sc.beta_end * k, 0.999)


class EMA:
    """Exponential moving average of weights; sample with the EMA copy (DDPM used 0.9999)."""

    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model, step):
        d = min(self.decay, (1 + step) / (10 + step))   # warm up the decay early in training
        for pe, p in zip(self.model.parameters(), model.parameters()):
            pe.mul_(d).add_(p.detach(), alpha=1 - d)
        for be, b in zip(self.model.buffers(), model.buffers()):
            be.copy_(b)
