"""Dispersed Elastic Fine-Tuning on ViT encoder MLPs. Paper Sec 11.2, Algorithm 3, App G, H."""

from contextlib import contextmanager

import numpy as np
import torch

from .adapters.vit import MlpGroup, _group_surrogate
from .cache import LayerCache


def partition_vit(model, calib_stats, percentile, merge_rho=0.9, kernel_mode="zero_bias"):
    """Algorithm 3 Phase 1: merge redundancies into vessels, then global capacity percentile.

    Returns per-block boolean elasticity arrays (True = plastic slack) and a merge log.
    Merged children are zeroed in place and kept as freed slack, not removed.
    """
    model = model.eval()
    groups = [
        MlpGroup(fc1=b.mlp[0], fc2=b.mlp[3], block_idx=i) for i, b in enumerate(model.encoder.layers)
    ]
    caches = [
        LayerCache(_group_surrogate(g, s), kernel_mode=kernel_mode, pair_dtype=np.float32)
        for g, s in zip(groups, calib_stats)
    ]

    merge_log = []
    freed = [np.zeros(c.surrogate.n, dtype=bool) for c in caches]
    for g, cache in enumerate(caches):
        while True:
            ii, jj, costs, _ = cache.merge_costs()
            if not len(costs):
                break
            rhos = cache.rho_hat[ii, jj]
            eligible = np.where(rhos >= merge_rho)[0]
            if not eligible.size:
                break
            k = eligible[np.argmin(costs[eligible])]
            i, j = int(ii[k]), int(jj[k])
            parent = cache.synthesize(i, j)
            grp = groups[g]
            with torch.no_grad():
                grp.fc1.weight.data[i] = torch.from_numpy(parent.w_eff).to(grp.fc1.weight.dtype)
                grp.fc1.bias.data[i] = float(parent.b)
                grp.fc2.weight.data[:, i] = torch.from_numpy(parent.w_out).to(grp.fc2.weight.dtype)
                grp.fc1.weight.data[j] = 0.0
                grp.fc1.bias.data[j] = 0.0
                grp.fc2.weight.data[:, j] = 0.0
            cache.apply_merge(i, j, parent)
            freed[g][j] = True
            merge_log.append({"block": g, "vessel": i, "freed": j, "rho": float(parent.rho_hat)})

    all_caps = np.concatenate([c.caps[c.active] for c in caches])
    tau = float(np.percentile(all_caps, percentile))

    elasticity = []
    for cache, freed_g in zip(caches, freed):
        slack = np.where(cache.active, cache.caps <= tau, True)
        slack[freed_g] = True
        elasticity.append(slack)
    return elasticity, {"tau": tau, "merges": merge_log}


def configure_trainable(model, elasticity, head):
    """Freeze everything except MLP fc1/fc2 (gated per unit at gradient time) and the new head."""
    for p in model.parameters():
        p.requires_grad = False
    for block in model.encoder.layers:
        for m in (block.mlp[0], block.mlp[3]):
            for p in m.parameters():
                p.requires_grad = True
    for p in head.parameters():
        p.requires_grad = True


def gate_gradients(model, elasticity):
    """eq (29): nullify gradients of frozen core units after backward, before the step."""
    with torch.no_grad():
        for block, slack in zip(model.encoder.layers, elasticity):
            core = torch.from_numpy(~slack).to(block.mlp[0].weight.device)
            fc1, fc2 = block.mlp[0], block.mlp[3]
            if fc1.weight.grad is not None:
                fc1.weight.grad[core] = 0.0
                fc1.bias.grad[core] = 0.0
            if fc2.weight.grad is not None:
                fc2.weight.grad[:, core] = 0.0


@contextmanager
def mask_slack_drift(model, elasticity):
    """MaskSlackDrift for ViT: zero fc2 columns of all slack units so only the core feeds the stream."""
    saved = []
    with torch.no_grad():
        for block, slack in zip(model.encoder.layers, elasticity):
            fc2 = block.mlp[3]
            idx = torch.from_numpy(slack).to(fc2.weight.device)
            saved.append(fc2.weight.data[:, idx].clone())
            fc2.weight.data[:, idx] = 0.0
    try:
        yield
    finally:
        with torch.no_grad():
            for block, slack, keep in zip(model.encoder.layers, elasticity, saved):
                fc2 = block.mlp[3]
                idx = torch.from_numpy(slack).to(fc2.weight.device)
                fc2.weight.data[:, idx] = keep
