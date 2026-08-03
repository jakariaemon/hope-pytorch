"""Torch-Pruning execution adapter for torchvision ResNet bottleneck models. Paper App B.1, Sec 8."""

from dataclasses import dataclass

import numpy as np
import torch
import torch_pruning as tp

from ..cache import LayerCache
from ..capacity import conv_output_vectors
from ..costs import dp_conv_filter, dp_evict, e_identity
from ..encoder import BlockInfo, Encoder
from ..surrogate import LayerSurrogate, effective_params


class IdentityBlock(torch.nn.Module):
    """Evicted residual block: Y = X, valid because block inputs are post-ReLU."""

    def forward(self, x):
        return x


@dataclass
class Group:
    conv: torch.nn.Conv2d
    bn: torch.nn.BatchNorm2d
    next_conv: torch.nn.Conv2d
    upstream: int  # group index feeding this conv's input channels, -1 for ambient
    downstream: int  # group index owning next_conv, -1 if next_conv is locked


def _group_surrogate(group):
    w = group.conv.weight.detach().cpu().double().numpy()
    w_raw = w.reshape(w.shape[0], -1)
    gamma = group.bn.weight.detach().cpu().double().numpy()
    beta = group.bn.bias.detach().cpu().double().numpy()
    mu = group.bn.running_mean.detach().cpu().double().numpy()
    var = group.bn.running_var.detach().cpu().double().numpy()
    w_eff, b = effective_params(w_raw, gamma, beta, mu, var, group.bn.eps)
    w_out = conv_output_vectors(group.next_conv.weight.detach().cpu().double().numpy())
    return LayerSurrogate(w_eff=w_eff, b=b, gamma=gamma, beta=beta, w_out=w_out)


class ResNetExecutor:
    """Physically applies prune, merge, and evict actions via a DependencyGraph."""

    def __init__(self, model, groups, blocks, block_modules, check_forward=True, input_size=64):
        self.model = model
        self.groups = groups
        self.blocks = blocks
        self.block_modules = block_modules
        self.check_forward = check_forward
        self.example = torch.randn(1, 3, input_size, input_size)
        self.live = [list(range(g.conv.out_channels)) for g in groups]
        self._init_shapes = [tuple(g.conv.weight.shape) for g in groups]
        self._init_out = [
            (g.next_conv.weight.shape[0],) + tuple(g.next_conv.weight.shape[2:])
            for g in groups
        ]
        self._rebuild()

    def _rebuild(self):
        self.dg = tp.DependencyGraph().build_dependency(self.model, example_inputs=self.example)

    def _check(self):
        if not self.check_forward:
            return
        with torch.no_grad():
            self.model(self.example)

    def _prune_channel(self, g, logical):
        phys = self.live[g].index(logical)
        group = self.dg.get_pruning_group(
            self.groups[g].conv, tp.prune_conv_out_channels, idxs=[phys]
        )
        group.prune()
        self.live[g].remove(logical)
        self._rebuild()
        self._check()

    def prune(self, g, i):
        self._prune_channel(g, i)

    def merge(self, g, vessel, purge, parent):
        """Write the parent into the vessel channel, then prune the purged channel."""
        grp = self.groups[g]
        phys = self.live[g].index(vessel)
        _, c_in_init, kh, kw = self._init_shapes[g]
        in_keep = (
            self.live[grp.upstream]
            if grp.upstream >= 0
            else list(range(c_in_init))
        )
        w_raw = parent.w_raw.reshape(c_in_init, kh, kw)[in_keep]
        out_keep = (
            self.live[grp.downstream]
            if grp.downstream >= 0
            else list(range(self._init_out[g][0]))
        )
        c_out_next, nkh, nkw = self._init_out[g]
        w_out = parent.w_out.reshape(c_out_next, nkh, nkw)[out_keep]
        with torch.no_grad():
            grp.conv.weight.data[phys] = torch.from_numpy(w_raw).to(grp.conv.weight.dtype)
            grp.bn.weight.data[phys] = float(parent.gamma)
            grp.bn.bias.data[phys] = float(parent.beta)
            grp.bn.running_mean.data[phys] = float(parent.mu)
            grp.bn.running_var.data[phys] = float(parent.sigma2)
            grp.next_conv.weight.data[:, phys] = torch.from_numpy(w_out).to(
                grp.next_conv.weight.dtype
            )
        self._prune_channel(g, purge)

    def evict(self, b):
        stage, idx = self.block_modules[b]
        getattr(self.model, stage)[idx] = IdentityBlock()
        for g in self.blocks[b].layers:
            self.live[g] = []
        self._rebuild()
        self._check()

def build_resnet_encoder(model, kernel_mode="zero_bias", audit=False, check_forward=True):
    """Build a HOPE encoder over the internal W1, W2 layers of every bottleneck block."""
    model = model.eval()
    groups, blocks, block_modules = [], [], []
    for stage_name in ["layer1", "layer2", "layer3", "layer4"]:
        stage = getattr(model, stage_name)
        for bi, block in enumerate(stage):
            g1 = len(groups)
            g2 = g1 + 1
            groups.append(Group(block.conv1, block.bn1, block.conv2, upstream=-1, downstream=g2))
            groups.append(Group(block.conv2, block.bn2, block.conv3, upstream=g1, downstream=-1))
            if block.downsample is None:
                prev = stage[bi - 1]
                e_id = e_identity(
                    prev.bn3.weight.detach().cpu().double().numpy(),
                    prev.bn3.bias.detach().cpu().double().numpy(),
                )
                n_bn = block.conv1.out_channels + block.conv2.out_channels + block.conv3.out_channels
                dp = dp_evict(block.conv1.weight.numel(), block.conv2.weight.numel(), n_bn)
                blocks.append(BlockInfo(layers=[g1, g2], e_identity=e_id, dp=dp))
                block_modules.append((stage_name, bi))

    caches = [LayerCache(_group_surrogate(g), kernel_mode=kernel_mode) for g in groups]
    dp_prune = [
        dp_conv_filter(
            g.conv.in_channels,
            g.conv.kernel_size[0],
            g.conv.kernel_size[1],
            g.next_conv.weight[:, 0].numel(),
        )
        for g in groups
    ]
    executor = ResNetExecutor(model, groups, blocks, block_modules, check_forward=check_forward)
    return Encoder(caches, dp_prune, blocks=blocks, executor=executor, audit=audit)
