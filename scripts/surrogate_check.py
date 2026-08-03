"""Compare kernel-predicted E[relu(y)^2] against a real batch on pretrained ResNet-50."""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hope.device import auto_device
from hope.kernels import self_kernel


def collect_bn_layers(model):
    """BN layers whose output feeds a ReLU: stem bn1 and bottleneck bn1, bn2."""
    layers = {"bn1": model.bn1}
    for stage in ["layer1", "layer2", "layer3", "layer4"]:
        for i, block in enumerate(getattr(model, stage)):
            layers[f"{stage}.{i}.bn1"] = block.bn1
            layers[f"{stage}.{i}.bn2"] = block.bn2
    return layers


def run_check(imagenet_dir, batch_size=64, device="auto", seed=0):
    from torchvision import datasets, models, transforms

    dev = auto_device(device)
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).eval().to(dev)

    tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    ds = datasets.ImageFolder(imagenet_dir, transform=tf)
    idx = np.random.default_rng(seed).choice(len(ds), size=batch_size, replace=False)
    batch = torch.stack([ds[int(i)][0] for i in idx]).to(dev)

    layers = collect_bn_layers(model)
    captured = {}
    hooks = [
        bn.register_forward_hook(lambda m, inp, out, name=name: captured.__setitem__(name, out.detach()))
        for name, bn in layers.items()
    ]
    with torch.no_grad():
        model(batch)
    for h in hooks:
        h.remove()

    rel_errs, flags = [], []
    for name, bn in layers.items():
        gamma = bn.weight.detach().cpu().double().numpy()
        beta = bn.bias.detach().cpu().double().numpy()
        predicted = self_kernel(gamma, beta)
        y = captured[name].cpu().double().numpy()
        empirical = np.mean(np.maximum(y, 0.0) ** 2, axis=(0, 2, 3))
        scale = np.maximum(np.abs(empirical), 1e-6)
        err = np.abs(predicted - empirical) / scale
        rel_errs.append(err)
        flags.append(np.abs(beta) / np.maximum(np.abs(gamma), 1e-12) > 2.0)

    err = np.concatenate(rel_errs)
    flagged = np.concatenate(flags)
    return {
        "channels": int(err.size),
        "median": float(np.median(err)),
        "p90": float(np.quantile(err, 0.9)),
        "mean": float(err.mean()),
        "flagged_large_bias": int(flagged.sum()),
        "median_flagged": float(np.median(err[flagged])) if flagged.any() else 0.0,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="ImageNet val directory (ImageFolder layout)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    stats = run_check(args.data, args.batch_size, args.device)
    for k, v in stats.items():
        print(f"{k}: {v}")
