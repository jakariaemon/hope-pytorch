"""Compress pretrained ViT-B/16 MLPs and record accuracy vs density on ImageNet."""

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hope.adapters.vit import build_vit_encoder
from hope.calibrate import load_stats
from hope.device import auto_device


def build_loader(data_dir, tf, batch_size, workers, subset, seed=0):
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets

    ds = datasets.ImageFolder(data_dir, transform=tf)
    if subset:
        idx = np.random.default_rng(seed).choice(len(ds), size=subset, replace=False)
        ds = Subset(ds, sorted(int(i) for i in idx))
    return DataLoader(ds, batch_size=batch_size, num_workers=workers)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval().to(device)
    correct = total = 0
    for x, y in loader:
        pred = model(x.to(device)).argmax(1).cpu()
        correct += (pred == y).sum().item()
        total += y.numel()
    model.cpu()
    return correct / total


def baseline_scores(enc, method):
    scores = []
    for cache in enc.caches:
        surr = cache.surrogate
        if method == "l1_in":
            s = np.abs(surr.w_eff).sum(axis=1)
        elif method == "l1_joint":
            s = np.abs(surr.w_eff).sum(axis=1) + np.abs(surr.w_out).sum(axis=1)
        else:
            raise ValueError(method)
        scores.append(s)
    return scores


def baseline_step(enc, scores):
    best = None
    for g, cache in enumerate(enc.caches):
        if cache.n_live <= 1:
            continue
        act = np.where(cache.active)[0]
        k = act[np.argmin(scores[g][act])]
        if best is None or scores[g][k] < best[0]:
            best = (scores[g][k], g, int(k))
    if best is None:
        return False
    _, g, k = best
    enc.executor.prune(g, k)
    enc.caches[g].apply_prune(k)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="ImageNet val directory")
    ap.add_argument("--calib", required=True, help="statistics file from calibrate_vit.py")
    ap.add_argument("--method", default="hope", choices=["hope", "l1_in", "l1_joint"])
    ap.add_argument("--kernel", default="zero_bias", choices=["zero_bias", "exact"])
    ap.add_argument("--target-density", type=float, default=0.3)
    ap.add_argument("--eval-every", type=float, default=0.05)
    ap.add_argument("--subset", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None)
    ap.add_argument("--audit", action="store_true")
    args = ap.parse_args()

    from torchvision.models import ViT_B_16_Weights, vit_b_16

    weights = ViT_B_16_Weights.IMAGENET1K_V1
    model = vit_b_16(weights=weights).eval()
    device = auto_device(args.device)
    loader = build_loader(args.data, weights.transforms(), args.batch_size, args.workers, args.subset)
    p0 = sum(p.numel() for p in model.parameters())

    t0 = time.time()
    enc = build_vit_encoder(model, load_stats(args.calib), kernel_mode=args.kernel, audit=args.audit, check_forward=False)
    print(f"cache init {time.time() - t0:.1f}s", flush=True)
    if args.method != "hope":
        scores = baseline_scores(enc, args.method)

    out_path = args.out or f"results/vit_{args.method}.csv"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    rows = []
    encode_time = 0.0

    def record():
        with torch.no_grad():
            check = model(torch.randn(1, 3, 224, 224))
        assert torch.isfinite(check).all()
        acc = evaluate(model, loader, device)
        p = sum(x.numel() for x in model.parameters())
        rows.append(
            {
                "method": args.method,
                "density": round(enc.density, 6),
                "top1": acc,
                "params": p,
                "param_ratio": round(p / p0, 6),
                "encode_seconds": round(encode_time, 1),
            }
        )
        print(f"density={enc.density:.3f} top1={acc:.4f} params={p / p0:.3f}", flush=True)

    record()
    for cp in np.arange(1.0 - args.eval_every, args.target_density - 1e-9, -args.eval_every):
        t0 = time.time()
        if args.method == "hope":
            enc.run(target_density=float(cp))
        else:
            while enc.density > cp and baseline_step(enc, scores):
                pass
        encode_time += time.time() - t0
        record()

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_path}, encode {encode_time:.1f}s")

    if args.audit and enc.audit_reports:
        viol = sum(r["violations"] > 0 for r in enc.audit_reports)
        rhos = [r["rho_ij"] for r in enc.audit_reports]
        print(
            f"lemma c3: merges={len(enc.audit_reports)} violated={viol} "
            f"rho_q10/50/90={np.quantile(rhos, [0.1, 0.5, 0.9]).round(3)}"
        )


if __name__ == "__main__":
    main()
