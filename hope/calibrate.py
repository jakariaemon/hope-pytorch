"""One-time statistics calibration for networks without BN. Paper App E box."""

import numpy as np
import torch


@torch.no_grad()
def calibrate_vit(model, batches, device="cpu"):
    """Per-unit pre-GELU statistics and residual stream energy for every encoder MLP."""
    model = model.eval().to(device)
    blocks = list(model.encoder.layers)
    acc = [{"n": 0, "sum": 0.0, "sumsq": 0.0, "stream_n": 0, "stream_sq": 0.0} for _ in blocks]
    hooks = []

    def unit_hook(idx):
        def fn(module, inp, out):
            y = out.detach().cpu().double().reshape(-1, out.shape[-1])
            acc[idx]["n"] += y.shape[0]
            acc[idx]["sum"] += y.sum(0).numpy()
            acc[idx]["sumsq"] += (y * y).sum(0).numpy()

        return fn

    def stream_hook(idx):
        def fn(module, inp):
            x = inp[0].detach().cpu().double().reshape(-1, inp[0].shape[-1])
            acc[idx]["stream_n"] += x.shape[0]
            acc[idx]["stream_sq"] += (x * x).sum(0).numpy()

        return fn

    for idx, block in enumerate(blocks):
        hooks.append(block.mlp[0].register_forward_hook(unit_hook(idx)))
        hooks.append(block.ln_2.register_forward_pre_hook(stream_hook(idx)))

    for x in batches:
        model(x.to(device))
    for h in hooks:
        h.remove()
    model.cpu()

    stats = []
    for a in acc:
        mu = a["sum"] / a["n"]
        var = np.maximum(a["sumsq"] / a["n"] - mu * mu, 1e-12)
        stats.append(
            {
                "mu": mu,
                "sigma": np.sqrt(var),
                "stream_rms": np.sqrt(a["stream_sq"] / a["stream_n"]),
            }
        )
    return stats


def save_stats(stats, path):
    flat = {}
    for i, s in enumerate(stats):
        for key, val in s.items():
            flat[f"{key}_{i}"] = val
    np.savez(path, n_blocks=len(stats), **flat)


def load_stats(path):
    data = np.load(path)
    n = int(data["n_blocks"])
    return [
        {key: data[f"{key}_{i}"] for key in ("mu", "sigma", "stream_rms")} for i in range(n)
    ]
