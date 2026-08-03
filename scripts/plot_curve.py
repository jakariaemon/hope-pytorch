"""Plot accuracy vs density curves from run_compress CSV files."""

import argparse
import csv

import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csvs", nargs="+", help="CSV files from run_compress.py")
    ap.add_argument("--out", default="results/curve.png")
    args = ap.parse_args()

    styles = ["-", "--", "-.", ":"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k, path in enumerate(args.csvs):
        with open(path) as f:
            rows = list(csv.DictReader(f))
        density = [float(r["density"]) for r in rows]
        top1 = [float(r["top1"]) for r in rows]
        # cycling line styles keep identical curves distinguishable
        ax.plot(
            density, top1, marker="o", markersize=3, linestyle=styles[k % len(styles)], label=rows[0]["method"]
        )
    ax.set_xlabel("model density")
    ax.set_ylabel("top-1 accuracy")
    ax.invert_xaxis()
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
