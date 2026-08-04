# hope-pytorch

PyTorch reproduction of **HOPE: Hilbert Operator for Progressive Encoding** ([arXiv:2607.21366](https://arxiv.org/abs/2607.21366)), a data-free framework for compressing trained networks by pruning, merging, and evicting entire residual blocks under one unified cost.

Each neuron is lifted into a Hilbert space through a Gaussian surrogate built from the network's BatchNorm statistics; compression greedily executes the action with the lowest distortion per removed parameter. There is no official code release; this implementation is built from the paper's equations, with equation numbers cited throughout the source.

## Install

```bash
git clone https://github.com/jakariaemon/hope-pytorch
cd hope-pytorch
pip install -e .
```

Requires Python 3.10+. Pinned versions used for development are in `requirements.txt`. Runs on CUDA, Apple Silicon (MPS), and CPU. The compression math itself is float64 NumPy on CPU; the GPU only accelerates accuracy evaluation.

## Usage

Compress a pretrained ResNet-50 and record the accuracy vs density curve:

```bash
python scripts/run_compress.py --data /path/to/imagenet/val --target-density 0.3 --audit
python scripts/run_compress.py --data /path/to/imagenet/val --method l1_in
python scripts/run_compress.py --data /path/to/imagenet/val --method bn_scale
python scripts/plot_curve.py results/hope.csv results/l1_in.csv results/bn_scale.csv
```

Without `--data` the script still compresses and reports densities, parameter counts, and the Lemma C.3 audit. `--subset 5000` evaluates on a fixed 5000 image subset (seed 0) if the full 50k validation pass is too slow.

Check the Gaussian surrogate against real activations:

```bash
python scripts/surrogate_check.py --data /path/to/imagenet/val
```

## Layout

```
hope/
  surrogate.py   BN-derived Gaussian surrogate (Sec 4, eq 1)
  kernels.py     self-kernel, warped correlation, cross-kernels (App E, eq 79-85)
  capacity.py    neuron and layer capacity (eq 75), conv adaptation (App B.1)
  parent.py      optimal parent synthesis and BN recovery (Sec 7, App D, B.5)
  costs.py       prune, merge, evict costs and static footprints (eq 6, 20, App B.2)
  cache.py       decoupled O(1) cache, scalars a and b only (App B.4)
  audit.py       Lemma C.3 straight-line path audit
  encoder.py     greedy loop (Sec 10, Algorithms 1-2)
  adapters/tp.py Torch-Pruning executor for bottleneck ResNets
tests/           one file per phase, pytest
scripts/         run_compress.py, plot_curve.py, surrogate_check.py
```

## Tests

```bash
pytest -q
```

The suite verifies the implementation against the paper: kernels against 40M sample Monte Carlo, parent deployment through a real BatchNorm to float tolerance, cache scalars against direct synthesis. Test D needs `HOPE_IMAGENET_DIR` pointing at ImageNet val and skips otherwise.

## Results

ResNet-50, torchvision `IMAGENET1K_V1` weights, ImageNet val top-1 on a fixed 5000 image subset (seed 0), without fine-tuning. HOPE runs the full prune, merge, evict loop; baselines prune the same filter set globally by their score.

![accuracy vs density](assets/curve.png)

| density | HOPE | L1 in | L1 joint | BN scale |
|---|---|---|---|---|
| 1.00 | 0.774 | 0.774 | 0.774 | 0.774 |
| ~0.86 | 0.684 | 0.002 | 0.002 | 0.003 |
| ~0.73 | 0.572 | 0.000 | 0.001 | 0.002 |
| ~0.59 | 0.391 | 0.002 | 0.001 | 0.001 |
| ~0.45 | 0.228 | 0.001 | 0.002 | 0.001 |
| 0.30 | 0.059 | 0.001 | 0.001 | 0.001 |

HOPE checkpoints land on action boundaries, so densities differ slightly from the baseline grid; each row reports the nearest recorded point (full data in `assets/*.csv`). The magnitude baselines collapse to chance after losing 5 to 10 percent of filters, the scale-symmetry failure the paper predicts. HOPE stays above every baseline at every density. Encoding to density 0.3 took 9 seconds on an Apple Silicon CPU, peak rss 1.8 GB.

## Measured findings

Numbers from this implementation, torchvision `IMAGENET1K_V1` weights unless noted.

**Gaussian surrogate accuracy** (Test D, one real batch of 64 val images): median per-channel relative error between predicted `E[relu(y)^2]` and empirical is 0.185, p90 0.472, over 7616 channels. The 59 channels flagged with |beta/gamma| above 2 are the most accurate (median 0.026).

**Lemma C.3 audit** on the real sweep: merges are rare on these weights (1 of 686 actions to density 0.3); the executed merge held the capacity bound at all 20 path steps, minimum margin 0.013.

**Exact vs zero-bias kernels.** `--kernel exact` evaluates the full biased cross-kernel (eq 83) for every pair, vectorized through Owen's T, at the same 9 second encode cost. The action sequence and accuracy are identical at every density: prune and evict costs use only the self-kernel, exact in both modes, and the one borderline merge from zero-bias (rho 0.14) is repriced out. Kernel mode may still matter for models with genuinely duplicated features.

**Zero-bias cross-kernel validity.** The paper approximates the cross-kernel by dropping biases (eq 5). Worst error against the exact biased kernel (eq 83), normalized by sqrt(Kii*Kjj):

| beta/gamma | 0.0 | 0.1 | 0.25 | 0.45 | 0.75 | 1.0 | 1.5 |
|---|---|---|---|---|---|---|---|
| worst err | 0.000 | 0.029 | 0.074 | 0.135 | 0.231 | 0.320 | 0.481 |

The approximation is exact at zero bias and as correlation approaches 1, which is where the greedy optimizer operates.

**Encoding ResNet-50 to density 0.3**: 686 actions, 678 prunes, 7 block evictions, 1 merge. Block eviction dominates early because its static parameter yield is large relative to its distortion under the DR criterion (eq 23).

## GELU and LayerNorm: pretrained ViT-B/16

This branch extends HOPE beyond the paper's PH-1 activations: closed-form GELU self-kernel via Stein identities and Owen's T, quadrature cross-kernels, and a 2D coefficient search replacing the eq (15) scale split, which relies on positive homogeneity that GELU lacks. LayerNorm models use a one-time calibration pass (App E): 512 unlabeled images recover per-unit statistics, then compression runs on weights alone.

```bash
python scripts/calibrate_vit.py --data /path/to/imagenet/val --out vit_calib.npz
python scripts/run_compress_vit.py --data /path/to/imagenet/val --calib vit_calib.npz
```

![vit accuracy vs density](assets/vit_curve.png)

Results on torchvision `IMAGENET1K_V1` ViT-B/16, compressing the MLP hidden units (about two thirds of parameters), fixed 5000 image subset, no fine-tuning:

| density | HOPE | L1 in | L1 joint |
|---|---|---|---|
| 1.00 | 0.823 | 0.823 | 0.823 |
| 0.95 | 0.819 | 0.158 | 0.823 |
| 0.90 | 0.806 | 0.244 | 0.226 |
| 0.85 | 0.804 | 0.095 | 0.147 |
| 0.80 | 0.632 | 0.045 | 0.034 |
| 0.70 | 0.271 | 0.003 | 0.004 |

Findings:

- Calibrated ViT pre-GELU units are heavily biased (|mu/sigma| median 1.7 to 2.9 per block), so the zero-bias scan is a rough ranking heuristic here; synthesis always reprices with exact kernels before deploying a merge.
- MLP block eviction is off by default (`evictable=False`). A mature pre-norm ViT is not identity robust: a single eviction at density 0.85 collapsed accuracy from 0.80 to 0.03, traced action by action. The static footprint of App B.2 also overprices eviction of an already thinned block, which is what selected it.
- 2310 merges executed with zero Lemma C.3 violations (rho median 0.69).
- At density 0.85 the model retains 0.804 of its 0.823 baseline, a better retention ratio than the BN ResNet-50 reproduction at the same density. The curve declines smoothly below 0.80 as the redundancy budget runs out.

## DEFT on pretrained ViT-B/16

Algorithm 3 adapted to the real pretrained ViT: partition MLP hidden units into a frozen core and plastic slack by global capacity percentile P (703 merges at rho above 0.9 free extra vessels), sever nothing across blocks (no direct slack-to-core weights exist in a stream-mediated transformer, so protection comes from exact freezing), fine-tune only the slack and a new head, and evaluate source retention with slack outputs masked and the source head grafted back.

```bash
python scripts/run_deft_vit.py --method deft --target svhn --percentile 20 --calib vit_calib.npz --imagenet /path/to/imagenet/val
```

ImageNet source, two targets, 3 epochs, no source data during adaptation. Best H-Score (harmonic mean of target accuracy and source retention) per method:

| method | CIFAR-100 target | SVHN target |
|---|---|---|
| DEFT P=20 | **0.815** (0.861 / 0.774) | **0.848** (0.939 / 0.774) |
| head only | 0.809 (0.795 / 0.823) | 0.654 (0.542 / 0.823) |
| full FT | 0.727 (0.862 / 0.629) | 0.518 (0.946 / 0.357, decays each epoch) |
| DEFT P=40 | 0.487 (0.869 / 0.338) | 0.498 (0.945 / 0.338) |

Findings:

- Theorem H.2 holds bitwise on the real model: after fine-tuning, every core fc1 row and fc2 column is identical to the source weights, and the measured source path is constant to four decimals across every epoch and both targets.
- The shared fc2 bias is a stream parameter, not a per-unit weight; training it leaks target drift into the source path. It stays frozen, mirroring the paper's own bias handling (App G.2).
- DEFT at P=20 takes first place on both targets: a narrow edge over head only on the easy transfer (0.815 vs 0.809, within single-seed noise) and a wide margin on the hard one, where linear probing collapses and full FT forgets catastrophically. The masked source path is identical across targets by construction, measured at 0.7736 in every epoch of both runs.
- The slack fraction P is a retention dial: masking 41 percent of units caps retention at 0.338, masking 21 percent keeps 0.774 at almost no target cost.

## Scope and extending

The core (`hope/`) is architecture agnostic: it operates on per-neuron effective weights, BN parameters, and outgoing weight vectors. Only the adapter is model specific; the included one covers torchvision bottleneck ResNets (resnet50, resnet101, resnet152). A new architecture needs a `LayerSurrogate` per compressible layer, static parameter footprints, and an executor with `prune`, `merge`, and `evict`. Networks without BN can use `surrogate.calibrated_params` after a one-time statistics pass (App E). Natural next targets: VGG-style nets with the non-residual eviction rule (App F.3), BasicBlock ResNets, and Transformer MLP blocks.

## Interpretation choices

Points where the paper leaves room and this implementation commits:

- `E_identity` for block eviction uses the terminal BN of the previous block on the trunk (eq 96 says "the preceding BN layer" without naming one).
- Density is the ratio of active to initial neurons over the compressible set, the internal W1 and W2 filters of every bottleneck.
- The paper's experiment used a Keras ResNet-50 checkpoint; this reproduction uses torchvision weights, so curves are comparable in shape, not point for point.
- Per Algorithm 1, layer caches are static except for the acted-on layer, even though a merge physically rewrites the next conv's input slices.

## License

MIT. If you use this code, cite the original paper (see `CITATION.cff`).
