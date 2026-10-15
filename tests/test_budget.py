import numpy as np

from mmmcal.budget import HI_MULT, LO_MULT, WEEKLY_BUDGET, current_mix_weekly, headroom, linear_fn, optimise, true_fn
from mmmcal.world import CHANNELS, N_TRAIN, random_world, simulate, true_quarter_sales


def _setup(seed=1):
    w = simulate(seed)
    train = w["df"].iloc[:N_TRAIN]
    return w, train, current_mix_weekly(train)


def test_current_mix_spends_exactly_the_budget():
    _, _, cur = _setup()
    assert np.isclose(sum(cur.values()), WEEKLY_BUDGET)


def test_optimiser_respects_budget_and_bounds():
    w, _, cur = _setup()
    best = optimise(true_fn(w), cur)
    assert np.isclose(sum(best.values()), WEEKLY_BUDGET, rtol=1e-6)
    for c in CHANNELS:
        assert LO_MULT * cur[c] - 1e-6 <= best[c] <= HI_MULT * cur[c] + 1e-6


def test_oracle_is_never_worse_than_the_current_mix():
    w, train, cur = _setup()
    best = optimise(true_fn(w), cur)
    assert true_quarter_sales(w, best) >= true_quarter_sales(w, cur) - 1e-6
    assert headroom(w, train) >= 0


def test_a_linear_model_goes_to_the_corners():
    _, _, cur = _setup()
    roi = {"TV": 1.0, "Search": 5.0, "Social": 3.0, "Display": 0.5, "Email": 4.0}
    best = optimise(linear_fn(roi), cur)
    assert np.isclose(best["Search"], HI_MULT * cur["Search"], rtol=1e-4)
    assert np.isclose(best["Display"], LO_MULT * cur["Display"], rtol=1e-4)


def test_random_worlds_have_a_misallocated_status_quo_with_real_headroom():
    for seed in range(1, 9):
        w = random_world(seed)
        h = headroom(w, w["df"].iloc[:N_TRAIN])
        assert 0.05 <= h <= 0.35, (seed, h)     # the optimiser has room to win, but not an absurd amount


def test_status_quo_overspends_the_low_roi_channel():
    w = random_world(3)
    cur = current_mix_weekly(w["df"].iloc[:N_TRAIN])
    assert cur["Display"] > 2 * cur["Email"] and cur["Display"] > cur["Search"]
