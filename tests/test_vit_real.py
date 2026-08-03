"""GELU stage 3 gates on the real pretrained ViT-B/16: calibration, execution, and merge fidelity. Needs HOPE_VIT_CALIB and HOPE_IMAGENET_DIR."""

import os

import numpy as np
import pytest
import torch

CALIB = os.environ.get("HOPE_VIT_CALIB")
IMAGENET = os.environ.get("HOPE_IMAGENET_DIR")

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not CALIB or not os.path.exists(CALIB), reason="set HOPE_VIT_CALIB"),
]


@pytest.fixture(scope="module")
def stats():
    from hope.calibrate import load_stats

    return load_stats(CALIB)


@pytest.fixture()
def vit():
    from torchvision.models import ViT_B_16_Weights, vit_b_16

    return vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1).eval()


def build(model, stats, mode="zero_bias", check=True):
    from hope.adapters.vit import build_vit_encoder

    return build_vit_encoder(model, stats, kernel_mode=mode, check_forward=check)


class TestRealCalibration:
    def test_stats_cover_real_model(self, stats):
        assert len(stats) == 12
        for s in stats:
            assert s["mu"].shape == (3072,)
            assert s["sigma"].shape == (3072,)
            assert s["stream_rms"].shape == (768,)
            assert np.all(s["sigma"] > 0)
            assert np.all(np.isfinite(s["mu"]))

    def test_kernels_finite_on_real_statistics(self, stats):
        from hope.activations import gelu_cross_kernel_exact, gelu_self_kernel

        rng = np.random.default_rng(0)
        for s in stats:
            k = gelu_self_kernel(s["sigma"], s["mu"])
            assert np.all(np.isfinite(k))
            assert np.all(k >= 0)
            idx = rng.choice(3072, 200, replace=False)
            jdx = rng.choice(3072, 200, replace=False)
            rho = rng.uniform(-0.99, 0.99, 200)
            kc = gelu_cross_kernel_exact(
                s["sigma"][idx], s["mu"][idx], s["sigma"][jdx], s["mu"][jdx], rho
            )
            bound = np.sqrt(k[idx] * k[jdx])
            assert np.all(np.isfinite(kc))
            assert np.all(np.abs(kc) <= bound * (1 + 1e-8))


class TestRealExecution:
    def test_actions_execute_on_real_vit(self, vit, stats):
        enc = build(vit, stats, check=True)
        d0 = enc.density
        history = enc.run(target_density=0.999, max_steps=10)
        assert history
        assert enc.density < d0
        with torch.no_grad():
            out = vit(torch.randn(1, 3, 224, 224))
        assert out.shape == (1, 1000)
        assert torch.isfinite(out).all()
        for g, cache in enumerate(enc.caches):
            assert len(enc.executor.live[g]) == cache.n_live

    def test_best_merge_function_drift(self, vit, stats):
        if not IMAGENET:
            pytest.skip("set HOPE_IMAGENET_DIR")
        from torchvision import datasets
        from torchvision.models import ViT_B_16_Weights

        tf = ViT_B_16_Weights.IMAGENET1K_V1.transforms()
        ds = datasets.ImageFolder(IMAGENET, transform=tf)
        idx = np.random.default_rng(1).choice(len(ds), 32, replace=False)
        x = torch.stack([ds[int(i)][0] for i in idx])

        enc = build(vit, stats, mode="zero_bias", check=False)
        with torch.no_grad():
            before = vit(x)
        g = 0
        ii, jj, costs, _ = enc.caches[g].merge_costs()
        k = int(np.argmin(costs))
        parent = enc.caches[g].synthesize(int(ii[k]), int(jj[k]))
        enc.executor.merge(g, int(ii[k]), int(jj[k]), parent)
        enc.caches[g].apply_merge(int(ii[k]), int(jj[k]), parent)
        with torch.no_grad():
            after = vit(x)
        agree = (before.argmax(1) == after.argmax(1)).float().mean().item()
        drift = (before - after).norm() / before.norm()
        print(f"\nbest merge on block 0: rho={parent.rho_hat:.3f} logit drift {drift:.4f} top1 agree {agree:.3f}")
        assert agree >= 0.9

    def test_evict_one_block(self, vit, stats):
        from hope.encoder import Action

        enc = build(vit, stats, check=True)
        enc.best_action = lambda: Action("evict", block=3, j_cost=1.0, dp=enc.blocks[3].dp)
        enc.step()
        assert enc.blocks[3].evicted
        assert enc.caches[3].n_live == 0
        with torch.no_grad():
            out = vit(torch.randn(1, 3, 224, 224))
        assert torch.isfinite(out).all()
