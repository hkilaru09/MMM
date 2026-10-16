"""The study design: which worlds, which tests, which models. Shared by the notebook and the fitting script."""
from __future__ import annotations

import pandas as pd

from .fit import get_model, hdi
from .geotest import CHANNEL_SEED_OFFSET, run_geo_test
from .world import CHANNELS, N_TRAIN, random_world

MAIN_SEED = 1                          # the worked example; the pipeline was developed on it
ROBUST_SEEDS = [11, 12, 13, 14, 15, 16]  # fresh worlds, never used while developing
ROI_PRIOR = (2.0, 0.75)                # weakly informative: median ROI 2, 95% range about 0.5-8.7 (same for every channel)


def all_tests(seed, world):
    """Off-tests and halve-tests for every channel, each with its own independent measurement noise."""
    off = {c: run_geo_test(world, c, seed=seed + 1000 + CHANNEL_SEED_OFFSET[c]) for c in CHANNELS}
    half = {c: run_geo_test(world, c, seed=seed + 2000 + CHANNEL_SEED_OFFSET[c], change=-0.5) for c in CHANNELS}
    return off, half


def concat_tests(*channels, tests):
    return pd.concat([tests[c]["lift_test"] for c in channels], ignore_index=True)


def next_test_channel(summary, train_df, exclude=("Social",)):
    """Rule fixed before looking at results: widest 94% interval of attributed incremental sales ($) among untested channels."""
    spend = train_df[CHANNELS].sum()
    width = {c: (lambda lo, hi: hi - lo)(*hdi(summary["roi"][c].to_numpy() * spend[c])) for c in CHANNELS if c not in exclude}
    return max(width, key=width.get), width


def fit_core(seed, world=None, **kw):
    """The four models every world gets: B (uncalibrated), C (Social test), D (C + next-most-uncertain channel), E (all channels)."""
    w = world or random_world(seed)
    train = w["df"].iloc[:N_TRAIN]
    off, _ = all_tests(seed, w)
    out = {"B": get_model("B", seed, w, **kw),
           "C": get_model("C", seed, w, lift_tests=off["Social"]["lift_test"], **kw)}
    nxt, widths = next_test_channel(out["C"], train)
    out["D"] = get_model(f"D_{nxt}", seed, w, lift_tests=concat_tests("Social", nxt, tests=off), **kw)
    out["E"] = get_model("E", seed, w, lift_tests=concat_tests(*CHANNELS, tests=off), **kw)
    out["_next"], out["_widths"] = nxt, widths
    return out


VARIANT_SPECS = {
    "H": dict(saturation="hill"),                                   # Hill curves (the family the truth uses)
    "R": dict(roi_prior=ROI_PRIOR),                                 # ROI-style prior against over-attribution
    "HR": dict(saturation="hill", roi_prior=ROI_PRIOR),
    "HE": dict(saturation="hill", tests="all"),                     # Hill + off-tests on all channels
    "HRE": dict(saturation="hill", roi_prior=ROI_PRIOR, tests="all"),
    "OC": dict(extra_controls=("u",)),                              # oracle: hand the model the unobserved demand shock
}


def fit_variants(seed, world=None, only=None, **kw):
    """Diagnostic variants (see VARIANT_SPECS). `only` restricts to a subset of keys."""
    w = world or random_world(seed)
    off, _ = all_tests(seed, w)
    all5 = concat_tests(*CHANNELS, tests=off)
    out = {}
    for key in (only or VARIANT_SPECS):
        spec = dict(VARIANT_SPECS[key])
        if spec.pop("tests", None) == "all":
            spec["lift_tests"] = all5
        out[key] = get_model(key, seed, w, **spec, **kw)
    return out
