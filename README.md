# hope-pytorch

PyTorch reproduction of **HOPE: Hilbert Operator for Progressive Encoding** ([arXiv:2607.21366](https://arxiv.org/abs/2607.21366)), a data-free framework for compressing trained networks by pruning, merging, and evicting entire residual blocks under one unified cost.

HOPE never touches a dataset. It lifts each neuron into a Hilbert space using a Gaussian surrogate built from the network's own BatchNorm statistics, then greedily executes the compression action with the lowest distortion per removed parameter. There is no official code release; this implementation is built from the paper's equations, with equation numbers cited throughout the source.

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

The suite gates each phase of the reproduction:

- **Kernels**: closed forms match Monte Carlo (40M samples) to 0.1 percent; Cauchy-Schwarz, diagonal consistency, monotonicity.
- **Capacity**: exact invariance under PH-1 rescaling and BN weight rescaling, where L1 magnitude scores break.
- **Parent synthesis**: the deployed parent reproduces `y_p = c1*y_i + c2*y_j` through a real torch BatchNorm to float tolerance, forward pass invariant to the sign of the recovered gamma, and merging identical twins is lossless.
- **Cache**: batched pair scalars agree with direct per-pair synthesis to 1e-9, and incremental updates after a merge match a cache rebuilt from scratch.
- **Encoder**: full loop on ResNet-50 with a shape-check forward after every action.

`tests/test_capacity.py::TestD` compares kernel predictions against real activations and needs `HOPE_IMAGENET_DIR` set; it is skipped otherwise.

## Measured findings

Numbers from this implementation, torchvision `IMAGENET1K_V1` weights unless noted.

**Zero-bias cross-kernel validity.** The paper approximates the cross-kernel by dropping biases (eq 5). Worst error against the exact biased kernel (eq 83), normalized by sqrt(Kii*Kjj):

| beta/gamma | 0.0 | 0.1 | 0.25 | 0.45 | 0.75 | 1.0 | 1.5 |
|---|---|---|---|---|---|---|---|
| worst err | 0.000 | 0.029 | 0.074 | 0.135 | 0.231 | 0.320 | 0.481 |

The approximation is exact at zero bias and as correlation approaches 1, which is where the greedy optimizer operates.

**Encoding ResNet-50 to density 0.3** (no data touched): 686 actions in about 7 seconds on CPU, 678 prunes, 7 block evictions, 1 merge. Block eviction dominates early because its static parameter yield is huge relative to its distortion under the DR criterion (eq 23). The single executed merge passed the Lemma C.3 audit with zero violations along the 20-step path.

Accuracy vs density curves require an ImageNet validation set and will be added once run.

## Scope and extending

The core (`hope/`) is architecture agnostic: it operates on per-neuron effective weights, BN parameters, and outgoing weight vectors. Only the adapter is model specific. The included adapter covers torchvision bottleneck ResNets (resnet50, resnet101, resnet152). To add an architecture, provide what `adapters/tp.py` provides: a `LayerSurrogate` per compressible layer, static parameter footprints, and an executor with `prune`, `merge`, and `evict`. Networks without BN can use `surrogate.calibrated_params` after a one-time statistics pass (App E). Natural next targets: VGG-style nets with the non-residual eviction rule (App F.3), BasicBlock ResNets, and Transformer MLP blocks.

## Interpretation choices

Points where the paper leaves room and this implementation commits:

- `E_identity` for block eviction uses the terminal BN of the previous block on the trunk (eq 96 says "the preceding BN layer" without naming one).
- Density is the ratio of active to initial neurons over the compressible set, the internal W1 and W2 filters of every bottleneck.
- The paper's experiment used a Keras ResNet-50 checkpoint; this reproduction uses torchvision weights, so curves are comparable in shape, not point for point.
- Per Algorithm 1, layer caches are static except for the acted-on layer, even though a merge physically rewrites the next conv's input slices.

## License

MIT. If you use this code, cite the original paper (see `CITATION.cff`).
