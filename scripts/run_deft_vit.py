"""DEFT on pretrained ViT-B/16: ImageNet source, CIFAR-100 target. Paper Algorithm 3."""

import argparse
import copy
import csv
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hope.calibrate import load_stats
from hope.deft import configure_trainable, gate_gradients, mask_slack_drift, partition_vit
from hope.device import auto_device


def target_loaders(root, batch_size):
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms

    tf = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    train = datasets.CIFAR100(root, train=True, download=True, transform=tf)
    idx = np.random.default_rng(0).permutation(len(train))
    tr = Subset(train, idx[:45000].tolist())
    va = Subset(train, idx[45000:].tolist())
    return (
        DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True),
        DataLoader(va, batch_size=256, num_workers=4),
    )


def source_loader(imagenet_dir, tf, subset=5000):
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets

    ds = datasets.ImageFolder(imagenet_dir, transform=tf)
    idx = np.random.default_rng(0).choice(len(ds), subset, replace=False)
    return DataLoader(Subset(ds, sorted(int(i) for i in idx)), batch_size=256, num_workers=4)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        pred = model(x.to(device)).argmax(1).cpu()
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", default="deft", choices=["deft", "head_only", "full_ft"])
    ap.add_argument("--percentile", type=float, default=40.0)
    ap.add_argument("--merge-rho", type=float, default=0.9)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=0.003)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--imagenet", required=True)
    ap.add_argument("--target-data", default="data")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from torchvision.models import ViT_B_16_Weights, vit_b_16

    weights = ViT_B_16_Weights.IMAGENET1K_V1
    device = auto_device(args.device)
    model = vit_b_16(weights=weights).eval()
    source_head = copy.deepcopy(model.heads)

    elasticity = None
    if args.method == "deft":
        t0 = time.time()
        elasticity, info = partition_vit(
            model, load_stats(args.calib), args.percentile, merge_rho=args.merge_rho
        )
        slack_frac = float(np.mean(np.concatenate(elasticity)))
        print(
            f"partition {time.time() - t0:.1f}s tau={info['tau']:.4f} "
            f"merges={len(info['merges'])} slack_frac={slack_frac:.3f}",
            flush=True,
        )

    target_head = nn.Sequential(nn.Linear(768, 100))
    model.heads = target_head
    model.to(device)

    if args.method == "deft":
        configure_trainable(model, elasticity, target_head)
    elif args.method == "head_only":
        for p in model.parameters():
            p.requires_grad = False
        for p in target_head.parameters():
            p.requires_grad = True
    else:
        for p in model.parameters():
            p.requires_grad = True

    train_loader, val_loader = target_loaders(args.target_data, args.batch_size)
    src_loader = source_loader(args.imagenet, weights.transforms())
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    out_path = args.out or f"results/deft_{args.method}.csv"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    rows, best = [], -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        for x, y in train_loader:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x.to(device)), y.to(device))
            loss.backward()
            if args.method == "deft":
                gate_gradients(model, elasticity)
            opt.step()
        train_s = time.time() - t0

        acc_tgt = evaluate(model, val_loader, device)
        model.heads = source_head.to(device)
        if args.method == "deft":
            with mask_slack_drift(model, elasticity):
                acc_src = evaluate(model, src_loader, device)
        else:
            acc_src = evaluate(model, src_loader, device)
        model.heads = target_head
        h = 2 * acc_tgt * acc_src / max(acc_tgt + acc_src, 1e-9)
        best = max(best, h)
        rows.append(
            {
                "method": args.method,
                "epoch": epoch,
                "target_acc": round(acc_tgt, 4),
                "source_acc": round(acc_src, 4),
                "h_score": round(h, 4),
                "train_seconds": round(train_s, 1),
            }
        )
        print(
            f"epoch {epoch} target {acc_tgt:.4f} source {acc_src:.4f} h {h:.4f} ({train_s:.0f}s)",
            flush=True,
        )

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_path}, best h {best:.4f}")


if __name__ == "__main__":
    main()
