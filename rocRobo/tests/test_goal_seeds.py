"""seed_cfg places a caller's IK solution in seed row 0 and leaves the rest of the seed set unchanged."""

import numpy as np
import pytest

from rocrobo import pipeline as v0


def _capture(monkeypatch):
    seen = []

    def fake(p, q, seeds, world, rcoll):
        seen.append(np.array(seeds))
        return seeds

    monkeypatch.setattr(v0, "_multiseed_jit", fake)
    return seen


def test_seed_cfg_replaces_row_zero_only(monkeypatch):
    seen = _capture(monkeypatch)
    p, q = np.array([0.4, 0.0, 0.4]), np.array([0.0, 1.0, 0.0, 0.0])
    seed = (v0.lo + v0.hi) / 2
    monkeypatch.setattr(v0, "rng", np.random.default_rng(1))
    v0.multiseed_ik(p, q, n=4)
    monkeypatch.setattr(v0, "rng", np.random.default_rng(1))
    v0.multiseed_ik(p, q, n=4, seed_cfg=seed)
    plain, seeded = seen
    assert np.allclose(seeded[0], seed)
    assert np.allclose(plain[1:], seeded[1:])


def test_seed_cfg_with_the_wrong_length_raises(monkeypatch):
    _capture(monkeypatch)
    with pytest.raises(ValueError):
        v0.multiseed_ik(np.array([0.4, 0.0, 0.4]), np.array([0.0, 1.0, 0.0, 0.0]), n=4, seed_cfg=np.zeros(3))
