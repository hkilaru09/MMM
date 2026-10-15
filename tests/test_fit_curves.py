"""Slow smoke tests: the response curves rebuilt from posterior draws must match PyMC-Marketing's own contributions."""
import numpy as np
import pytest

from mmmcal.fit import curve_draws, get_model
from mmmcal.world import CHANNELS, N_TRAIN, random_world

TINY = dict(chains=1, cores=1, draws=60, tune=60, target_accept=0.9)


@pytest.mark.slow
@pytest.mark.parametrize("saturation", ["logistic", "hill"])
def test_rebuilt_curves_match_the_models_own_channel_contributions(saturation, tmp_path):
    w = random_world(1)
    S = get_model(f"smoke_{saturation}", 1, w, saturation=saturation, cache_dir=tmp_path, sampler=TINY)
    # Weekly contribution at the channel's mean (adstock-smoothed ~ mean) spend must be close to the model's average weekly
    # contribution; exact equality is impossible (variable spend), so check the same order of magnitude per channel.
    train = w["df"].iloc[:N_TRAIN]
    avg_contrib = S["roi"].mean() * train[CHANNELS].sum() / N_TRAIN       # model's average weekly incremental sales
    for c in CHANNELS:
        rebuilt = curve_draws(S["params"], c, train[c].mean()).mean()
        assert 0.5 < rebuilt / avg_contrib[c] < 2.0, (saturation, c, rebuilt, avg_contrib[c])


@pytest.mark.slow
def test_roi_prior_pulls_roi_estimates_toward_its_median(tmp_path):
    w = random_world(1)
    free = get_model("free", 1, w, cache_dir=tmp_path, sampler=TINY)
    tight = get_model("tight", 1, w, roi_prior=(2.0, 0.25), cache_dir=tmp_path, sampler=TINY)
    dist = lambda S: np.abs(np.log(S["roi"].mean() / 2.0)).mean()
    assert dist(tight) < dist(free)
