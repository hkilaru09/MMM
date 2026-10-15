"""A simulated marketing world with built-in confounding and known ground truth."""
from __future__ import annotations

import numpy as np
import pandas as pd

CHANNELS = ["TV", "Search", "Social", "Display", "Email"]

# True channel parameters. `roi` is the TRUE average incremental revenue per $ over the
# training years; beta is solved from it, so it holds exactly.
REFERENCE_PARAMS = {
    "TV":      dict(alpha=0.60, k_mult=1.2, shape=1.5, roi=1.6, base=28_000, gamma=1.2, theta=1.5, flight=0.0),
    "Search":  dict(alpha=0.10, k_mult=0.8, shape=1.0, roi=3.2, base=14_000, gamma=0.5, theta=3.0, flight=0.0),
    "Social":  dict(alpha=0.30, k_mult=1.5, shape=1.0, roi=2.0, base=14_000, gamma=0.7, theta=2.0, flight=1.0),
    "Display": dict(alpha=0.40, k_mult=1.3, shape=1.0, roi=0.8, base=12_000, gamma=1.0, theta=1.5, flight=1.0),
    "Email":   dict(alpha=0.10, k_mult=0.6, shape=1.0, roi=6.0, base=5_000, gamma=0.1, theta=0.3, flight=0.0),
}

# --- Randomised worlds -------------------------------------------------------------------------------
# Each world draws its channels' true adstock decay, saturation and ROI from these ranges, so "robust across
# worlds" means robust to different markets, not just different noise. The status quo is deliberately
# misallocated: planners over-weight Display (low ROI) and under-weight Search and Email (high ROI), which
# leaves roughly 12-21% of achievable incremental sales on the table under the halve/double planning rule.
# These ranges and base spends were chosen by looking only at the TRUE response curves (the resulting
# headroom), before any model was fitted.
PARAM_RANGES = {
    "TV":      dict(alpha=(0.45, 0.75), k_mult=(1.0, 1.6), shape=(1.2, 2.0), roi=(1.0, 2.0), gamma=1.2, theta=1.5, flight=0.0),
    "Search":  dict(alpha=(0.00, 0.20), k_mult=(0.8, 1.4), shape=(1.0, 1.0), roi=(3.0, 5.0), gamma=0.5, theta=3.0, flight=0.0),
    "Social":  dict(alpha=(0.20, 0.40), k_mult=(1.2, 1.8), shape=(1.0, 1.0), roi=(1.5, 2.5), gamma=0.7, theta=2.0, flight=1.0),
    "Display": dict(alpha=(0.30, 0.50), k_mult=(1.0, 1.6), shape=(1.0, 1.0), roi=(0.3, 0.8), gamma=1.0, theta=1.5, flight=1.0),
    "Email":   dict(alpha=(0.00, 0.20), k_mult=(0.8, 1.4), shape=(1.0, 1.0), roi=(5.0, 10.0), gamma=0.1, theta=0.3, flight=0.0),
}
STATUS_QUO_BASE_SPEND = {"TV": 20_000, "Search": 5_000, "Social": 11_000, "Display": 34_000, "Email": 3_000}


def sample_params(seed):
    """Draw one world's true channel parameters (independent of the seed that drives the noise)."""
    rng = np.random.default_rng(10_000 + seed)
    out = {}
    for c, r in PARAM_RANGES.items():
        u = lambda ab: float(rng.uniform(*ab))
        out[c] = dict(alpha=u(r["alpha"]), k_mult=u(r["k_mult"]), shape=u(r["shape"]), roi=u(r["roi"]),
                      base=STATUS_QUO_BASE_SPEND[c] * float(rng.uniform(0.85, 1.15)),
                      gamma=r["gamma"] * float(rng.uniform(0.75, 1.25)),
                      theta=r["theta"] * float(rng.uniform(0.75, 1.25)),
                      flight=r["flight"])
    return out


def random_world(seed):
    return simulate(seed, sample_params(seed))

N_TRAIN = 156
N_EXTRA = 4          # weeks after training reserved for the geo-lift test window
BUFFER = 8           # extra weeks generated so lead/lag shifts don't wrap


def _hol_indicator(dates, month, day):
    out = np.zeros(len(dates))
    for i, d in enumerate(dates):
        target = pd.Timestamp(year=d.year, month=month, day=day)
        if 0 <= (target - d).days < 7:
            out[i] = 1.0
    return out


def _black_friday_week(dates):
    out = np.zeros(len(dates))
    for i, d in enumerate(dates):
        nov1 = pd.Timestamp(year=d.year, month=11, day=1)
        first_thu = nov1 + pd.Timedelta(days=(3 - nov1.weekday()) % 7)
        bf = first_thu + pd.Timedelta(days=21 + 1)  # 4th Thursday + 1
        if 0 <= (bf - d).days < 7:
            out[i] = 1.0
    return out


def adstock(x, alpha):
    """Normalised geometric adstock: steady-state output equals constant input."""
    out = np.zeros_like(x, dtype=float)
    prev = 0.0
    for t, v in enumerate(x):
        prev = (1 - alpha) * v + alpha * prev
        out[t] = prev
    return out


def hill(a, k, s):
    return a**s / (a**s + k**s)


def simulate(seed=42, params=None):
    params = params or REFERENCE_PARAMS
    rng = np.random.default_rng(seed)
    T = N_TRAIN + N_EXTRA + BUFFER
    dates = pd.date_range("2021-01-04", periods=T, freq="W-MON")
    doy = dates.dayofyear.to_numpy()
    phi = 2 * np.pi * doy / 365.25

    # --- seasonality + holidays (baseline demand) ---
    season = (0.60 * np.cos(2 * np.pi * (doy - 340) / 365.25)
              + 0.15 * np.sin(2 * phi * 2)
              + 0.08 * np.cos(3 * phi))
    bf = _black_friday_week(dates)
    xmas = _hol_indicator(dates, 12, 25)
    summer = _hol_indicator(dates, 7, 11)
    hol_effect = 0.30 * bf + 0.20 * xmas + 0.12 * summer

    # --- unobserved demand shock (AR1): hits sales AND is reacted to by marketers ---
    u = np.zeros(T)
    for t in range(1, T):
        u[t] = 0.7 * u[t - 1] + rng.normal(0, 0.057)   # stationary sd ~ 0.08

    trend = 1.0 + 0.001 * np.arange(T)
    baseline = 500_000 * trend * (1 + 0.25 * season + hol_effect + u)

    # --- spend: marketers anticipate peak season (lead 3 weeks) and chase demand ---
    plan = np.roll(season, -3) + 0.6 * np.maximum(hol_effect, np.roll(hol_effect, -1)) / 0.3
    plan = plan - plan.mean()
    flight = np.zeros(T)
    for t in range(1, T):
        flight[t] = 0.8 * flight[t - 1] + rng.normal(0, 0.12)   # shared Social/Display campaign flighting
    u_lag = np.roll(u, 1)

    spend = {}
    for c, p in params.items():
        z = (p["gamma"] * plan + p["theta"] * u_lag + p["flight"] * flight
             + rng.normal(0, 0.12, T))
        s = np.exp(z)
        spend[c] = p["base"] * s / s[:N_TRAIN].mean()
    spend = pd.DataFrame(spend, index=dates)

    # --- true response ---
    contrib, beta, kappa = {}, {}, {}
    sales = baseline.copy()
    for c, p in params.items():
        a = adstock(spend[c].to_numpy(), p["alpha"])
        k = p["k_mult"] * a[:N_TRAIN].mean()
        h = hill(a, k, p["shape"])
        b = p["roi"] * spend[c].to_numpy()[:N_TRAIN].sum() / h[:N_TRAIN].sum()
        contrib[c] = b * h
        beta[c], kappa[c] = b, k
        sales = sales + contrib[c]
    sales = sales + rng.normal(0, 12_000, T)

    df = pd.DataFrame({"date": dates, "sales": sales, "baseline": baseline, "u": u,
                       "black_friday": bf, "christmas": xmas, "summer_sale": summer})
    for c in CHANNELS:
        df[c] = spend[c].to_numpy()
        df[f"contrib_{c}"] = contrib[c]
    return dict(df=df.iloc[: N_TRAIN + N_EXTRA].reset_index(drop=True), beta=beta, kappa=kappa,
                params=params, seed=seed)


def true_roi(world):
    d = world["df"].iloc[:N_TRAIN]
    return {c: d[f"contrib_{c}"].sum() / d[c].sum() for c in CHANNELS}


def true_weekly_response(world, channel, spend_level):
    """Steady-state weekly incremental sales from constant weekly spend."""
    p = world["params"][channel]
    return world["beta"][channel] * hill(spend_level, world["kappa"][channel], p["shape"])


def true_quarter_sales(world, alloc_weekly):
    """True incremental sales over a 13-week quarter for a constant weekly allocation."""
    return 13 * sum(true_weekly_response(world, c, alloc_weekly[c]) for c in CHANNELS)
