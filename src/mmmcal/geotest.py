"""Simulated geo-lift experiments: one channel's spend changed in half the regions."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .world import CHANNELS, N_EXTRA, N_TRAIN, adstock, hill

REGION_SHARES = np.array([0.22, 0.16, 0.13, 0.11, 0.09, 0.08, 0.07, 0.06, 0.05, 0.03])
REGION_NOISE_SD = 0.04      # weekly multiplicative noise per region
PRE_WEEKS = 8

# Each channel's test is run in different weeks/regions in practice, so its measurement error is
# independent of the others. (Social's offset is 0.)
CHANNEL_SEED_OFFSET = {"TV": 101, "Search": 202, "Social": 0, "Display": 303, "Email": 404}


def run_geo_test(world, channel="Social", seed=7, change=-1.0, washout=0, noise_sd=REGION_NOISE_SD):
    """Simulate a geo-lift test on `channel` in the weeks after the training data.

    Regions are paired by size and randomised within pairs; treated regions get their spend multiplied
    by (1 + change) for the test window (-1.0 = switch off, -0.5 = halve). The analyst estimates the
    weekly sales change with a difference-in-differences across regions (pre-period = last 8 training
    weeks). The first `washout` test weeks are dropped because carry-over (adstock) means the effect
    has not fully developed yet.
    """
    rng = np.random.default_rng(seed)
    df = world["df"]
    t0 = N_TRAIN
    weeks = np.arange(t0 - PRE_WEEKS, t0 + N_EXTRA)
    p = world["params"][channel]

    spend_on = df[channel].to_numpy().copy()
    spend_test = spend_on.copy()
    spend_test[t0:] = spend_on[t0:] * (1.0 + change)

    def contrib(spend):
        a = adstock(spend, p["alpha"])
        return world["beta"][channel] * hill(a, world["kappa"][channel], p["shape"])

    c_on, c_test = contrib(spend_on), contrib(spend_test)
    expected_other = (df["baseline"].to_numpy()
                      + sum(df[f"contrib_{c}"].to_numpy() for c in CHANNELS if c != channel))

    order = np.argsort(-REGION_SHARES)
    treated = np.zeros(len(REGION_SHARES), dtype=bool)
    for i in range(0, len(order), 2):
        treated[rng.choice(order[i:i + 2])] = True

    rows = []
    for r in range(len(REGION_SHARES)):
        eps = rng.normal(0, noise_sd, len(weeks))
        for j, t in enumerate(weeks):
            ch = c_test[t] if (treated[r] and t >= t0) else c_on[t]
            rows.append(dict(region=r, week=t, treated=bool(treated[r]), post=bool(t >= t0),
                             y=(expected_other[t] + ch) * np.exp(eps[j])))   # share-normalised sales
    panel = pd.DataFrame(rows)
    analysed = panel[~((panel.week >= t0) & (panel.week < t0 + washout))]

    d = analysed.groupby(["region", "treated", "post"]).y.mean().unstack("post")
    d["did"] = d[True] - d[False]
    d = d.reset_index()
    t_d, c_d = d[d.treated].did.to_numpy(), d[~d.treated].did.to_numpy()
    effect = t_d.mean() - c_d.mean()               # weekly sales change caused by the spend change
    se = np.sqrt(t_d.var(ddof=1) / len(t_d) + c_d.var(ddof=1) / len(c_d))
    x = df[channel].iloc[t0 + washout:t0 + N_EXTRA].mean()   # weekly spend level before the change

    # PyMC-Marketing requires measured lift and spend change to share a sign. A wrong-sign measurement is pure
    # noise (the true effect is never of the opposite sign), so it is passed as "indistinguishable from zero":
    # a tiny same-sign value with its full standard error, rather than being dropped (which would bias the evidence).
    dx = change * x
    wrong_sign = bool(effect * dx <= 0)
    effect_used = np.sign(dx) * 0.01 * se if wrong_sign else effect

    return dict(panel=panel, treated=treated, effect=effect, se=se, x=x, channel=channel, wrong_sign=wrong_sign,
                truth_effect_in_window=float((c_test[t0 + washout:] - c_on[t0 + washout:]).mean()),
                lift_test=pd.DataFrame({"channel": [channel], "x": [x], "delta_x": [dx],
                                        "delta_y": [effect_used], "sigma": [se]}))
