"""Compress pretrained ResNet-50 and record accuracy vs density. Paper Sec 11.1."""

import argparse
import csv
import json
import os
import resource
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hope.adapters.tp import build_resnet_encoder
from hope.device import auto_device


def build_loader(data_dir, batch_size, workers, subset, seed=0):
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms

    tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    ds = datasets.ImageFolder(data_dir, transform=tf)
    if subset:
        idx = np.random.default_rng(seed).choice(len(ds), size=subset, replace=False)
        ds = Subset(ds, sorted(int(i) for i in idx))
    return DataLoader(ds, batch_size=batch_size, num_workers=workers, pin_memory=True)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval().to(device)
    correct = total = 0
    for x, y in loader:
        pred = model(x.to(device)).argmax(1).cpu()
        correct += (pred == y).sum().item()
        total += y.numel()
    model.cpu()
    return correct / max(total, 1)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def baseline_scores(enc, method):
    """Static importance scores for the pruning baselines, Sec 11.1."""
    scores = []
    for g, cache in enumerate(enc.caches):
        surr = cache.surrogate
        grp = enc.executor.groups[g]
        w_raw = grp.conv.weight.detach().cpu().double().numpy().reshape(surr.n, -1)
        if method == "l1_in":
            s = np.abs(w_raw).sum(axis=1)
        elif method == "l1_joint":
            s = np.abs(w_raw).sum(axis=1) + np.abs(surr.w_out).sum(axis=1)
        elif method == "bn_scale":
            s = np.abs(surr.gamma)
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
        val = scores[g][k]
        if best is None or val < best[0]:
            best = (val, g, int(k))
    if best is None:
        return False
    _, g, k = best
    enc.executor.prune(g, k)
    enc.caches[g].apply_prune(k)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="resnet50", choices=["resnet50"])
    ap.add_argument("--method", default="hope", choices=["hope", "l1_in", "l1_joint", "bn_scale"])
    ap.add_argument("--kernel", default="zero_bias", choices=["zero_bias", "exact"])
    ap.add_argument("--target-density", type=float, default=0.3)
    ap.add_argument("--eval-every", type=float, default=0.05)
    ap.add_argument("--data", default=None, help="ImageNet val directory (ImageFolder layout)")
    ap.add_argument("--subset", type=int, default=0, help="fixed image subset size, seed 0")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None)
    ap.add_argument("--audit", action="store_true", help="run the Lemma C.3 merge path audit")
    ap.add_argument("--no-check", action="store_true", help="skip per-action shape checks")
    args = ap.parse_args()

    from torchvision.models import ResNet50_Weights, resnet50

    device = auto_device(args.device)
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).eval()
    p0 = n_params(model)

    loader = None
    if args.data:
        loader = build_loader(args.data, args.batch_size, args.workers, args.subset)
        if args.subset:
            print(f"eval on fixed {args.subset}-image subset, seed 0")

    enc = build_resnet_encoder(
        model, kernel_mode=args.kernel, audit=args.audit, check_forward=not args.no_check
    )
    if args.method != "hope":
        scores = baseline_scores(enc, args.method)

    out_path = args.out or f"results/{args.method}.csv"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    rows = []
    encode_time = 0.0
    label = args.method if args.kernel == "zero_bias" else f"{args.method}_{args.kernel}"

    def record():
        acc = evaluate(model, loader, device) if loader else float("nan")
        rows.append(
            {
                "method": label,
                "density": round(enc.density, 6),
                "top1": acc,
                "params": n_params(model),
                "param_ratio": n_params(model) / p0,
                "encode_seconds": round(encode_time, 2),
            }
        )
        print(f"density={enc.density:.3f} top1={acc:.4f} params={rows[-1]['param_ratio']:.3f}")

    record()
    checkpoints = np.arange(1.0 - args.eval_every, args.target_density - 1e-9, -args.eval_every)
    for cp in checkpoints:
        t0 = time.time()
        if args.method == "hope":
            enc.run(target_density=float(cp))
        else:
            while enc.density > cp and baseline_step(enc, scores):
                pass
        encode_time += time.time() - t0
        record()
        if args.method == "hope" and enc.best_action() is None:
            break

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_path}")

    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2 if sys.platform == "darwin" else 1024)
    print(f"encode wall clock: {encode_time:.1f}s, peak rss: {peak_mb:.0f} MB")

    if args.audit and enc.audit_reports:
        summary = {
            "merges": len(enc.audit_reports),
            "violations": int(sum(r["violations"] > 0 for r in enc.audit_reports)),
            "min_margin": float(min(r["min_margin"] for r in enc.audit_reports)),
            "rho_quantiles": [
                float(q) for q in np.quantile([r["rho_ij"] for r in enc.audit_reports], [0.1, 0.5, 0.9])
            ],
        }
        audit_path = os.path.splitext(out_path)[0] + "_audit.json"
        with open(audit_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"lemma c3 audit: {summary}")


if __name__ == "__main__":
    main()
