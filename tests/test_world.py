import numpy as np

from mmmcal.world import CHANNELS, N_TRAIN, REFERENCE_PARAMS, adstock, simulate, true_roi, true_weekly_response


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
