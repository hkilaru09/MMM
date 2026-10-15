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
