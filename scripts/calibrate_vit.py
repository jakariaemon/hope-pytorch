"""Calibrate pretrained ViT-B/16 on real ImageNet val images and save the statistics."""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hope.calibrate import calibrate_vit, save_stats
from hope.device import auto_device


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="ImageNet val directory (ImageFolder layout)")
    ap.add_argument("--samples", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from torchvision import datasets, models

    weights = models.ViT_B_16_Weights.IMAGENET1K_V1
    model = models.vit_b_16(weights=weights).eval()
    tf = weights.transforms()

    ds = datasets.ImageFolder(args.data, transform=tf)
    idx = np.random.default_rng(0).choice(len(ds), size=args.samples, replace=False)
    batches = []
    for lo in range(0, args.samples, args.batch_size):
        batches.append(torch.stack([ds[int(i)][0] for i in idx[lo : lo + args.batch_size]]))

    stats = calibrate_vit(model, batches, device=auto_device(args.device))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_stats(stats, args.out)

    for i, s in enumerate(stats):
        ratio = np.abs(s["mu"]) / s["sigma"]
        print(
            f"block {i:2d} sigma median {np.median(s['sigma']):.3f} "
            f"|mu/sigma| median {np.median(ratio):.3f} p90 {np.quantile(ratio, 0.9):.3f} "
            f"e_identity {s['stream_rms'].sum():.1f}"
        )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
