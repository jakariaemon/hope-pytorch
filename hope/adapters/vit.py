"""Torch-Pruning execution adapter for torchvision ViT encoder MLPs. Paper App B.2.1, F.2."""

from dataclasses import dataclass

import numpy as np
import torch

from ..cache import LayerCache
from ..encoder import BlockInfo, Encoder
from ..surrogate import LayerSurrogate


class ZeroMLP(torch.nn.Module):
    """Evicted MLP block: the residual branch contributes nothing."""

    def forward(self, x):
        return x * 0.0


@dataclass
class MlpGroup:
    fc1: torch.nn.Linear
    fc2: torch.nn.Linear
    block_idx: int


def _group_surrogate(group, stats):
    return LayerSurrogate(
        w_eff=group.fc1.weight.detach().cpu().double().numpy(),
        b=group.fc1.bias.detach().cpu().double().numpy(),
        gamma=stats["sigma"],
        beta=stats["mu"],
        w_out=group.fc2.weight.detach().cpu().double().numpy().T.copy(),
        activation="gelu",
    )


class VitExecutor:
    """Applies prune, merge, and MLP eviction on a torchvision VisionTransformer."""

    def __init__(self, model, groups, image_size=224, check_forward=True):
        self.model = model
        self.groups = groups
        self.check_forward = check_forward
        self.example = torch.randn(1, 3, image_size, image_size)
        self.live = [list(range(g.fc1.out_features)) for g in groups]

    def _check(self):
        if not self.check_forward:
            return
        with torch.no_grad():
            self.model(self.example)

    def _prune_channel(self, g, logical):
        """Direct surgery: an MLP hidden unit is one fc1 row and one fc2 column."""
        grp = self.groups[g]
        phys = self.live[g].index(logical)
        keep = [k for k in range(grp.fc1.out_features) if k != phys]
        with torch.no_grad():
            grp.fc1.weight = torch.nn.Parameter(grp.fc1.weight.data[keep].clone())
            grp.fc1.bias = torch.nn.Parameter(grp.fc1.bias.data[keep].clone())
            grp.fc1.out_features -= 1
            grp.fc2.weight = torch.nn.Parameter(grp.fc2.weight.data[:, keep].clone())
            grp.fc2.in_features -= 1
        self.live[g].remove(logical)
        self._check()

    def prune(self, g, i):
        self._prune_channel(g, i)

    def merge(self, g, vessel, purge, parent):
        """LayerNorm deployment: effective parameters are the physical parameters."""
        grp = self.groups[g]
        phys = self.live[g].index(vessel)
        with torch.no_grad():
            grp.fc1.weight.data[phys] = torch.from_numpy(parent.w_eff).to(grp.fc1.weight.dtype)
            grp.fc1.bias.data[phys] = float(parent.b)
            grp.fc2.weight.data[:, phys] = torch.from_numpy(parent.w_out).to(grp.fc2.weight.dtype)
        self._prune_channel(g, purge)

    def evict(self, b):
        block = self.model.encoder.layers[self.groups[b].block_idx]
        block.mlp = ZeroMLP()
        self.live[b] = []
        self._check()


def build_vit_encoder(
    model, calib_stats, kernel_mode="zero_bias", audit=False, check_forward=True, image_size=224, evictable=False
):
    """HOPE encoder over the MLP hidden units of every encoder block.

    Eviction is off by default: mature pre-norm ViTs are not identity-robust, so removing
    a whole MLP collapses the network, unlike the ResNet blocks of paper Sec 8.
    """
    model = model.eval()
    groups = []
    for idx, block in enumerate(model.encoder.layers):
        groups.append(MlpGroup(fc1=block.mlp[0], fc2=block.mlp[3], block_idx=idx))

    caches = []
    blocks = []
    dp_prune = []
    for g, (group, stats) in enumerate(zip(groups, calib_stats)):
        caches.append(
            LayerCache(_group_surrogate(group, stats), kernel_mode=kernel_mode, pair_dtype=np.float32)
        )
        d_model = group.fc1.in_features
        dp_prune.append(2 * d_model + 1)
        if evictable:
            ln_params = 2 * d_model
            dp_evict = group.fc1.weight.numel() + group.fc1.bias.numel() + ln_params
            blocks.append(
                BlockInfo(layers=[g], e_identity=float(np.sum(stats["stream_rms"])), dp=dp_evict)
            )

    executor = VitExecutor(model, groups, image_size=image_size, check_forward=check_forward)
    return Encoder(caches, dp_prune, blocks=blocks, executor=executor, audit=audit)
