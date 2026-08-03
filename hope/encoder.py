"""Greedy progressive encoding loop. Paper Sec 10, Algorithms 1 and 2."""

from dataclasses import dataclass, field

import numpy as np

from .audit import audit_merge_path
from .costs import distortion_rate, j_evict


@dataclass
class Action:
    kind: str
    layer: int = -1
    i: int = -1
    j: int = -1
    block: int = -1
    j_cost: float = 0.0
    dp: float = 1.0

    @property
    def dr(self):
        return distortion_rate(self.j_cost, self.dp)


@dataclass
class BlockInfo:
    layers: list
    e_identity: float
    dp: float
    evicted: bool = False


@dataclass
class HistoryRow:
    step: int
    kind: str
    layer: int
    block: int
    j_cost: float
    dr: float
    density: float


class Encoder:
    """Scans all cached actions by DR = J / dP_init, executes one, updates locally."""

    def __init__(self, caches, dp_prune, blocks=None, executor=None, audit=False):
        self.caches = caches
        self.dp_prune = dp_prune
        self.blocks = blocks or []
        self.executor = executor
        self.audit = audit
        self.total_init = sum(c.surrogate.n for c in caches)
        self.history = []
        self.audit_reports = []
        self._step = 0

    @property
    def n_live(self):
        return sum(c.n_live for c in self.caches)

    @property
    def density(self):
        return self.n_live / self.total_init

    def best_action(self):
        """Alg 1 greedy scan: O(1) cost queries against live residual capacities."""
        best = None
        for li, cache in enumerate(self.caches):
            if cache.n_live > 1:
                act, costs = cache.prune_costs()
                k = int(np.argmin(costs))
                cand = Action("prune", layer=li, i=int(act[k]), j_cost=float(costs[k]), dp=self.dp_prune[li])
                if best is None or cand.dr < best.dr:
                    best = cand
            if cache.n_live >= 2:
                ii, jj, costs, _ = cache.merge_costs()
                if len(costs):
                    k = int(np.argmin(costs))
                    cand = Action(
                        "merge", layer=li, i=int(ii[k]), j=int(jj[k]), j_cost=float(costs[k]), dp=self.dp_prune[li]
                    )
                    if best is None or cand.dr < best.dr:
                        best = cand
        for bi, block in enumerate(self.blocks):
            if block.evicted:
                continue
            stats = [(self.caches[l].n_live, self.caches[l].e_rem) for l in block.layers]
            if all(n == 0 for n, _ in stats):
                continue
            cand = Action("evict", block=bi, j_cost=float(j_evict(stats, block.e_identity)), dp=block.dp)
            if best is None or cand.dr < best.dr:
                best = cand
        return best

    def step(self):
        action = self.best_action()
        if action is None:
            return None
        if action.kind == "prune":
            cache = self.caches[action.layer]
            if self.executor is not None:
                self.executor.prune(action.layer, action.i)
            cache.apply_prune(action.i)
        elif action.kind == "merge":
            cache = self.caches[action.layer]
            parent = cache.synthesize(action.i, action.j)
            if self.audit:
                self.audit_reports.append(audit_merge_path(cache, action.i, action.j, parent))
            if self.executor is not None:
                self.executor.merge(action.layer, action.i, action.j, parent)
            cache.apply_merge(action.i, action.j, parent)
        else:
            block = self.blocks[action.block]
            if self.executor is not None:
                self.executor.evict(action.block)
            for l in block.layers:
                self.caches[l].clear()
            block.evicted = True
        self._step += 1
        self.history.append(
            HistoryRow(self._step, action.kind, action.layer, action.block, action.j_cost, action.dr, self.density)
        )
        return action

    def run(self, target_density, callback=None, max_steps=None):
        while self.density > target_density:
            if max_steps is not None and self._step >= max_steps:
                break
            action = self.step()
            if action is None:
                break
            if callback is not None:
                callback(self, action)
        return self.history
