# hope-pytorch

PyTorch implementation of **HOPE: Hilbert Operator for Progressive Encoding** ([arXiv:2607.21366](https://arxiv.org/abs/2607.21366)): data-free structured compression of trained networks, built from the paper's equations with equation numbers cited throughout the source. There is no official code release.

Beyond the paper's ReLU + BatchNorm scope, this repo adds GELU + LayerNorm support (new closed-form kernels) and the paper's DEFT elastic fine-tuning, both validated on real pretrained models.

Headline results, real pretrained checkpoints, no fine-tuning unless the method is fine-tuning:

- **ResNet-50**: 68.4 percent ImageNet top-1 with 17.5 percent of parameters removed; magnitude baselines collapse to chance. Encoding takes 9 seconds on a laptop CPU.
- **ViT-B/16**: 80.4 percent top-1 (from 82.3) with 15 percent of MLP units removed, using a 512 image calibration pass instead of BatchNorm statistics.
- **DEFT on ViT**: first place on both transfer targets (H-Score 0.815 easy, 0.848 hard), with the frozen core bitwise identical to the source weights after fine-tuning.

## Install

```bash
git clone https://github.com/jakariaemon/hope-pytorch
cd hope-pytorch
pip install -e .
```

Python 3.10+. Pinned development versions in `requirements.txt`. Runs on CUDA, Apple Silicon, and CPU; the compression math is float64 NumPy on CPU, the GPU only accelerates evaluation.

## Usage

```bash
# ResNet-50 compression and baselines
python scripts/run_compress.py --data /path/to/imagenet/val --target-density 0.3 --audit
python scripts/run_compress.py --data /path/to/imagenet/val --method l1_in

# ViT-B/16: calibrate once, then compress
python scripts/calibrate_vit.py --data /path/to/imagenet/val --out vit_calib.npz
python scripts/run_compress_vit.py --data /path/to/imagenet/val --calib vit_calib.npz

# DEFT transfer, ImageNet source to SVHN target
python scripts/run_deft_vit.py --method deft --target svhn --percentile 20 \
  --calib vit_calib.npz --imagenet /path/to/imagenet/val
```

## Results

Evaluation uses a fixed 5000 image ImageNet val subset (seed 0). Full CSVs in `assets/`.

### ResNet-50 compression

![accuracy vs density](assets/curve.png)

| density | HOPE | L1 in | L1 joint | BN scale |
|---|---|---|---|---|
| 1.00 | 0.774 | 0.774 | 0.774 | 0.774 |
| 0.86 | 0.684 | 0.002 | 0.002 | 0.003 |
| 0.59 | 0.391 | 0.002 | 0.001 | 0.001 |
| 0.30 | 0.059 | 0.001 | 0.001 | 0.001 |

Magnitude baselines lose 5 to 10 percent of filters and collapse, the scale-symmetry failure the paper predicts. Exact and zero-bias kernels produce identical trajectories on these weights.

### ViT-B/16 compression (GELU + LayerNorm)

![vit accuracy vs density](assets/vit_curve.png)

| density | HOPE | L1 in | L1 joint |
|---|---|---|---|
| 1.00 | 0.823 | 0.823 | 0.823 |
| 0.90 | 0.806 | 0.244 | 0.226 |
| 0.85 | 0.804 | 0.095 | 0.147 |
| 0.80 | 0.632 | 0.045 | 0.034 |

Compresses the MLP hidden units, about two thirds of parameters. The extension contributes a closed-form GELU self-kernel (Stein identities plus Owen's T), quadrature cross-kernels, and a 2D coefficient search replacing the paper's eq (15) scale split, which requires positive homogeneity that GELU lacks.

### DEFT transfer (ImageNet source, 3 epochs)

Best H-Score, the harmonic mean of target accuracy and source retention:

| method | CIFAR-100 target | SVHN target |
|---|---|---|
| DEFT P=20 | **0.815** (0.861 / 0.774) | **0.848** (0.939 / 0.774) |
| head only | 0.809 (0.795 / 0.823) | 0.654 (0.542 / 0.823) |
| full FT | 0.727 (0.862 / 0.629) | 0.518 (0.946 / 0.357) |
| DEFT P=40 | 0.487 (0.869 / 0.338) | 0.498 (0.945 / 0.338) |

DEFT trains only the low-capacity slack plus a new head. On the hard target it matches full fine-tuning's accuracy while full FT destroys its source knowledge. The slack fraction P trades plasticity against retention; P=20 wins both settings.

## Layout

```
hope/
  surrogate.py    BN-derived Gaussian surrogate (Sec 4, eq 1)
  kernels.py      ReLU kernels, warped correlation (App E, eq 79-85)
  activations.py  GELU kernels and the activation registry
  calibrate.py    statistics pass for LayerNorm models (App E)
  capacity.py     neuron and layer capacity (eq 75)
  parent.py       parent synthesis, BN recovery, 2D search (Sec 7, App D)
  costs.py        prune, merge, evict costs and footprints (eq 6, 20, App B.2)
  cache.py        decoupled pair cache, slim mode for wide layers (App B.4)
  audit.py        Lemma C.3 path audit
  encoder.py      greedy loop (Sec 10, Algorithms 1-2)
  deft.py         DEFT partition, gradient gating, slack masking (Sec 11.2, Alg 3)
  adapters/       ResNet (Torch-Pruning) and ViT (direct surgery) executors
```

## Tests

```bash
pytest -q
```

Kernels are gated against 40M sample Monte Carlo at 0.1 percent; parent deployment is exact through a real BatchNorm; cache scalars match direct synthesis; DEFT's frozen core is asserted bitwise on the real ViT. Slow tests need `HOPE_IMAGENET_DIR` and `HOPE_VIT_CALIB` set and skip otherwise.

## Notes and limitations

- The Gaussian surrogate has real error on real activations: median 18.5 percent per channel on ResNet-50 (Test D). The method works because rankings, not absolute values, drive decisions.
- The zero-bias cross-kernel degrades with bias (7 percent normalized error at |beta/gamma| 0.25, 48 percent at 1.5). ViT and Whisper units are far more biased than ResNet's, so the scan is a heuristic there; executed merges are always repriced with exact kernels.
- Block eviction is unsound on pre-norm transformers and off by default for ViT: one eviction collapsed accuracy from 0.80 to 0.03, and the static footprint of App B.2 overprices eviction of already thinned blocks.
- All 2310 ViT merges satisfied the Lemma C.3 capacity bound; merges are rare on ResNet-50 (1 in 686 actions).
- DEFT numbers are single seed and P was selected after observing P=40; a multi-seed grid is the natural hardening step.
- DEFT's structural mask is a no-op on ViT (no direct slack-to-core weights across blocks); protection comes from exact freezing, and the shared fc2 bias must stay frozen or target drift leaks into the source path.

## Extending

The core is architecture agnostic; only adapters know model structure. A new architecture needs a `LayerSurrogate` per compressible layer, static footprints, and an executor with `prune`, `merge`, `evict`. Networks without BatchNorm need one calibration pass. ReLU and GELU kernels exist; other activations need their kernels derived or tabulated.

## License

MIT. If you use this code, cite the original paper (see `CITATION.cff`).
