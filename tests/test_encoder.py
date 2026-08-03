"""Phase 5 gates: greedy loop behavior and physical execution on ResNet-50."""

import numpy as np
import pytest
import torch

from hope.cache import LayerCache
from hope.encoder import Action, Encoder
from hope.surrogate import LayerSurrogate


def math_layer(n, d_in, d_out, seed):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((n, d_in))
    w[1] = w[0] + 0.02 * rng.standard_normal(d_in)
    b = rng.standard_normal(n) * 0.2
    gamma = rng.uniform(0.5, 1.5, n)
    beta = rng.standard_normal(n) * 0.2
    w_out = rng.standard_normal((n, d_out))
    b[1], gamma[1], beta[1] = b[0], gamma[0], beta[0]
    w_out[1] = w_out[0] + 0.02 * rng.standard_normal(d_out)
    return LayerSurrogate(w_eff=w, b=b, gamma=gamma, beta=beta, w_out=w_out)


class TestPureMathLoop:
    def test_run_to_density(self):
        caches = [LayerCache(math_layer(16, 20, 10, seed=s)) for s in range(3)]
        enc = Encoder(caches, dp_prune=[100.0, 100.0, 100.0], audit=True)
        history = enc.run(target_density=0.5)
        assert enc.density <= 0.5
        densities = [row.density for row in history]
        assert all(a > b for a, b in zip(densities, densities[1:]))
        kinds = {row.kind for row in history}
        assert "merge" in kinds or "prune" in kinds
        for cache in caches:
            live = cache.active.sum()
            assert live >= 1
            assert cache.e_rem == pytest.approx(cache.caps[cache.active].sum(), rel=1e-6)
        assert len(enc.audit_reports) == sum(1 for r in history if r.kind == "merge")

    def test_merge_of_near_twins_wins_first(self):
        cache = LayerCache(math_layer(16, 20, 10, seed=7))
        enc = Encoder([cache], dp_prune=[100.0])
        action = enc.best_action()
        assert action.kind == "merge"
        assert {action.i, action.j} == {0, 1}


@pytest.fixture()
def resnet50():
    from torchvision.models import resnet50 as make

    torch.manual_seed(0)
    model = make(weights=None).eval()
    # plant near-duplicate filters so a merge is selected early
    with torch.no_grad():
        b = model.layer1[1]
        b.conv1.weight[1] = b.conv1.weight[0] * 1.01
        b.bn1.weight[1] = b.bn1.weight[0]
        b.bn1.bias[1] = b.bn1.bias[0]
        b.bn1.running_mean[1] = b.bn1.running_mean[0]
        b.bn1.running_var[1] = b.bn1.running_var[0]
        b.conv2.weight[:, 1] = b.conv2.weight[:, 0]
    return model


class TestResNetAdapter:
    def test_encoder_runs_with_shape_checks(self, resnet50):
        from hope.adapters.tp import build_resnet_encoder

        enc = build_resnet_encoder(resnet50, check_forward=True)
        d0 = enc.density
        history = enc.run(target_density=0.9, max_steps=25)
        assert 1 <= len(history) <= 25
        assert enc.density <= 0.9 or len(history) == 25
        assert enc.density < d0
        assert {r.kind for r in history} & {"prune", "merge", "evict"}
        with torch.no_grad():
            out = resnet50(torch.randn(2, 3, 64, 64))
        assert out.shape == (2, 1000)
        assert torch.isfinite(out).all()
        for g, cache in enumerate(enc.caches):
            assert len(enc.executor.live[g]) == cache.n_live

    def test_merge_selected_for_planted_twins(self, resnet50):
        from hope.adapters.tp import build_resnet_encoder

        enc = build_resnet_encoder(resnet50, check_forward=False)
        merges = [r for r in enc.run(target_density=0.9, max_steps=25) if r.kind == "merge"]
        assert merges, "planted twin filters should trigger at least one merge"

    def test_evict_path(self, resnet50):
        from hope.adapters.tp import build_resnet_encoder

        enc = build_resnet_encoder(resnet50, check_forward=True)
        block = enc.blocks[0]
        n_before = enc.n_live
        enc.best_action = lambda: Action("evict", block=0, j_cost=1.0, dp=block.dp)
        enc.step()
        assert block.evicted
        assert enc.n_live < n_before
        for l in block.layers:
            assert enc.caches[l].n_live == 0
        with torch.no_grad():
            out = resnet50(torch.randn(1, 3, 64, 64))
        assert out.shape == (1, 1000)
