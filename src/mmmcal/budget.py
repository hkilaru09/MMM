"""Budget optimisation under a realistic planning rule, scored against the true response."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .fit import curve_draws
from .world import CHANNELS, true_quarter_sales, true_weekly_response

WEEKLY_BUDGET = 1_000_000 / 13
LO_MULT, HI_MULT = 0.5, 2.0     # a channel may be halved or doubled vs. the current mix, no more


def current_mix_weekly(train_df):
    m = train_df[CHANNELS].mean()
    return (m / m.sum() * WEEKLY_BUDGET).to_dict()


def optimise(response_fn, current):
    x0 = np.array([current[c] for c in CHANNELS])
    res = minimize(lambda x: -sum(response_fn(c, x[i]) for i, c in enumerate(CHANNELS)), x0, method="SLSQP",
                   bounds=[(LO_MULT * v, HI_MULT * v) for v in x0],
                   constraints=[{"type": "eq", "fun": lambda x: x.sum() - WEEKLY_BUDGET}],
                   options={"maxiter": 500, "ftol": 1e-9})
    return dict(zip(CHANNELS, res.x))


bayes_fn = lambda S: (lambda c, x: curve_draws(S["params"], c, x).mean())
linear_fn = lambda roi: (lambda c, x: roi[c] * x)
true_fn = lambda w: (lambda c, x: true_weekly_response(w, c, x))


def headroom(world, train_df):
    """Fractional gain in true quarterly sales from the oracle allocation over the current mix."""
    cur = current_mix_weekly(train_df)
    best = optimise(true_fn(world), cur)
    return true_quarter_sales(world, best) / true_quarter_sales(world, cur) - 1


def budget_table(world, train_df, naive, summaries):
    cur = current_mix_weekly(train_df)
    allocs = {"historical mix": cur, "naive": optimise(linear_fn(naive), cur)}
    for k, S in summaries.items():
        allocs[k] = optimise(bayes_fn(S), cur)
    allocs["oracle"] = optimise(true_fn(world), cur)
    t = pd.DataFrame({k: {**{c: a[c] * 13 for c in CHANNELS}, "true quarterly incremental sales": true_quarter_sales(world, a)}
                      for k, a in allocs.items()}).T
    h, o, n = (t.loc[k, "true quarterly incremental sales"] for k in ["historical mix", "oracle", "naive"])
    t["vs naive"] = t["true quarterly incremental sales"] / n - 1
    t["vs historical"] = t["true quarterly incremental sales"] / h - 1
    t["share of achievable gain captured"] = (t["true quarterly incremental sales"] - h) / (o - h)
    return t, allocs
