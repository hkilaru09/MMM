import numpy as np

from mmmcal.geotest import run_geo_test
from mmmcal.world import simulate


def test_with_no_noise_the_estimate_equals_the_true_effect():
    w = simulate(1)
    for c in ("Social", "TV", "Email"):
        g = run_geo_test(w, c, seed=3, noise_sd=0.0)
        assert np.isclose(g["effect"], g["truth_effect_in_window"], rtol=1e-9), c
        assert g["effect"] < 0 and not g["wrong_sign"]


def test_halving_spend_costs_less_than_switching_off():
    w = simulate(1)
    off = run_geo_test(w, "Search", seed=3, noise_sd=0.0)["effect"]
    half = run_geo_test(w, "Search", seed=3, change=-0.5, noise_sd=0.0)["effect"]
    assert off < half < 0


def test_wrong_sign_measurements_are_passed_as_indistinguishable_from_zero():
    w = simulate(1)
    seen_wrong = seen_right = False
    for s in range(40):
        g = run_geo_test(w, "Display", seed=s, noise_sd=0.15)
        row = g["lift_test"].iloc[0]
        assert row["delta_y"] * row["delta_x"] > 0          # always sign-consistent, as PyMC-Marketing requires
        if g["wrong_sign"]:
            seen_wrong = True
            assert np.isclose(abs(row["delta_y"]), 0.01 * g["se"])
        else:
            seen_right = True
            assert row["delta_y"] == g["effect"]
    assert seen_wrong and seen_right


def test_spend_level_reported_is_the_pre_change_level():
    w = simulate(1)
    g = run_geo_test(w, "TV", seed=3)
    assert np.isclose(g["lift_test"]["delta_x"].iloc[0], -g["x"])


def test_washout_reduces_the_carryover_bias_for_slow_channels():
    # A slow-decaying channel (TV, decay ~0.6) is still ramping down early in the test, so a window that includes those
    # weeks under-measures it. Benchmark: what a fully developed cut would remove = the channel's contribution in those weeks.
    from mmmcal.world import N_EXTRA, N_TRAIN, random_world
    w = random_world(2)
    share_measured = {}
    for washout in (0, 4):
        g = run_geo_test(w, "TV", seed=3, noise_sd=0.0, washout=washout)
        full_cut = w["df"]["contrib_TV"].iloc[N_TRAIN + washout:N_TRAIN + N_EXTRA].mean()
        share_measured[washout] = -g["effect"] / full_cut
    assert share_measured[0] < 0.7                        # without a washout the effect is badly under-measured
    assert share_measured[4] > share_measured[0] + 0.2    # the washout recovers most of it
    assert 0.85 < share_measured[4] <= 1.0


def test_test_window_is_eight_weeks_with_four_analysed():
    from mmmcal.geotest import DEFAULT_WASHOUT
    from mmmcal.world import N_EXTRA
    assert N_EXTRA == 8 and DEFAULT_WASHOUT == 4
    g = run_geo_test(simulate(1), "Social", seed=3)
    assert g["panel"].week.max() - g["panel"].week.min() + 1 == 8 + N_EXTRA   # 8 pre-weeks + 8 test weeks
