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


def fit_variants(seed, world=None, **kw):
    """Diagnostic variants: Hill curves, ROI prior, both, plus calibration on top, and an oracle confounder control."""
    w = world or random_world(seed)
    off, _ = all_tests(seed, w)
    all5 = concat_tests(*CHANNELS, tests=off)
    return {
        "H": get_model("H", seed, w, saturation="hill", **kw),
        "R": get_model("R", seed, w, roi_prior=ROI_PRIOR, **kw),
        "HR": get_model("HR", seed, w, saturation="hill", roi_prior=ROI_PRIOR, **kw),
        "HE": get_model("HE", seed, w, saturation="hill", lift_tests=all5, **kw),
        "HRE": get_model("HRE", seed, w, saturation="hill", roi_prior=ROI_PRIOR, lift_tests=all5, **kw),
        "OC": get_model("OC", seed, w, extra_controls=("u",), **kw),
    }
