"""DEFT gates on the real pretrained ViT-B/16: Theorem H.2 bitwise core freeze and slack masking. Needs HOPE_VIT_CALIB; CIFAR-100 downloads on first run."""

import os

import numpy as np
import pytest
import torch

CALIB = os.environ.get("HOPE_VIT_CALIB")

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not CALIB or not os.path.exists(CALIB), reason="set HOPE_VIT_CALIB"),
]


@pytest.fixture(scope="module")
def setup():
    from torchvision.models import ViT_B_16_Weights, vit_b_16

    from hope.calibrate import load_stats
    from hope.deft import partition_vit

    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1).eval()
    elasticity, info = partition_vit(model, load_stats(CALIB), percentile=40.0, merge_rho=2.0)
    return model, elasticity, info


class TestPartition:
    def test_global_percentile_split(self, setup):
        _, elasticity, info = setup
        assert len(elasticity) == 12
        frac = float(np.mean(np.concatenate(elasticity)))
        assert 0.35 <= frac <= 0.45
        assert info["tau"] > 0


class TestTheoremH2:
    def test_core_bitwise_frozen_after_real_steps(self, setup):
        import torch.nn as nn

        from hope.deft import configure_trainable, gate_gradients

        model, elasticity, _ = setup
        snap = {}
        for g, block in enumerate(model.encoder.layers):
            snap[g] = (block.mlp[0].weight.detach().clone(), block.mlp[3].weight.detach().clone())
        attn_snap = model.encoder.layers[0].self_attention.in_proj_weight.detach().clone()

        target_head = nn.Sequential(nn.Linear(768, 100))
        model.heads = target_head
        configure_trainable(model, elasticity, target_head)
        opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.05, momentum=0.9)
        loss_fn = nn.CrossEntropyLoss()

        from torchvision import datasets, transforms

        tf = transforms.Compose(
            [
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        ds = datasets.CIFAR100("data", train=True, download=True, transform=tf)
        x = torch.stack([ds[i][0] for i in range(16)])
        y = torch.tensor([ds[i][1] for i in range(16)])

        model.train()
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            loss_fn(model(x), y).backward()
            gate_gradients(model, elasticity)
            opt.step()

        changed = 0
        for g, block in enumerate(model.encoder.layers):
            core = ~elasticity[g]
            fc1, fc2 = block.mlp[0].weight.detach(), block.mlp[3].weight.detach()
            assert torch.equal(fc1[core], snap[g][0][core])
            assert torch.equal(fc2[:, core], snap[g][1][:, core])
            slack = elasticity[g]
            changed += int(not torch.equal(fc1[slack], snap[g][0][slack]))
        assert changed >= 10
        assert torch.equal(model.encoder.layers[0].self_attention.in_proj_weight.detach(), attn_snap)


class TestMaskSlackDrift:
    def test_mask_equals_core_only_and_restores(self, setup):
        from hope.deft import mask_slack_drift

        model, elasticity, _ = setup
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        fc2 = model.encoder.layers[3].mlp[3]
        before = fc2.weight.detach().clone()

        with torch.no_grad(), mask_slack_drift(model, elasticity):
            slack_cols = fc2.weight[:, torch.from_numpy(elasticity[3])]
            assert torch.all(slack_cols == 0)
            masked_out = model(x)
        assert torch.equal(fc2.weight.detach(), before)
        assert torch.isfinite(masked_out).all()
