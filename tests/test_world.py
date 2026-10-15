import numpy as np

from mmmcal.world import (CHANNELS, N_TRAIN, PARAM_RANGES, REFERENCE_PARAMS, adstock, random_world, sample_params,
                          simulate, true_roi, true_weekly_response)


def test_true_roi_matches_the_roi_that_was_set():
    for seed in (1, 2, 3):
        w = simulate(seed)
        roi = true_roi(w)
        for c in CHANNELS:
            assert np.isclose(roi[c], w["params"][c]["roi"], rtol=1e-9), (seed, c)


def test_simulation_is_reproducible_and_seed_dependent():
    a, b, c = simulate(5), simulate(5), simulate(6)
    assert a["df"].equals(b["df"])
    assert not a["df"]["sales"].equals(c["df"]["sales"])


def test_normalised_adstock_has_unit_steady_state_gain():
    out = adstock(np.full(200, 10.0), alpha=0.6)
    assert np.isclose(out[-1], 10.0, rtol=1e-6)
    assert np.isclose(adstock(np.array([10.0]), alpha=0.6)[0], 4.0)   # (1 - alpha) * x on the first week


def test_spend_is_confounded_with_baseline_demand():
    d = simulate(1)["df"].iloc[:N_TRAIN]
    for c in ("TV", "Search", "Display"):
        assert np.corrcoef(d[c], d["baseline"])[0, 1] > 0.3, c


def test_true_response_is_increasing_and_saturating():
    w = simulate(1)
    for c in CHANNELS:
        grid = np.linspace(1_000, 60_000, 40)
        y = np.array([true_weekly_response(w, c, s) for s in grid])
        assert np.all(np.diff(y) > 0)
        assert y[-1] < w["beta"][c]        # never exceeds the channel's ceiling


def test_reference_params_cover_all_channels():
    assert set(REFERENCE_PARAMS) == set(CHANNELS)


def test_sampled_params_stay_inside_their_ranges_and_differ_across_worlds():
    draws = [sample_params(s) for s in range(1, 9)]
    for d in draws:
        for c, r in PARAM_RANGES.items():
            for k in ("alpha", "k_mult", "shape", "roi"):
                assert r[k][0] <= d[c][k] <= r[k][1], (c, k)
    assert len({round(d["TV"]["roi"], 6) for d in draws}) == len(draws)      # every world has its own truth
    assert len({round(d["Display"]["alpha"], 6) for d in draws}) == len(draws)


def test_random_world_true_roi_matches_its_own_parameters():
    w = random_world(4)
    roi = true_roi(w)
    for c in CHANNELS:
        assert np.isclose(roi[c], w["params"][c]["roi"], rtol=1e-9)
