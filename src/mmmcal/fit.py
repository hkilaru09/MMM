"""Fitting PyMC-Marketing MMMs (with on-disk caching) and summarising the posteriors."""
from __future__ import annotations

import gc
import hashlib
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

from .world import CHANNELS, N_TRAIN

CTRL = ["trend", "black_friday", "christmas", "summer_sale"]
try:
    from importlib.metadata import version as _v
    PYMC_MARKETING_VERSION = _v("pymc-marketing")
except Exception:  # pragma: no cover
    PYMC_MARKETING_VERSION = "unknown"
# Measured on one world (seed 1): tune 800 / ta 0.95 gave 53 divergences (logistic) and 378 divergences with R-hat 1.25 (Hill);
# tune 2000 / draws 1000 / ta 0.98 gave 8 (R-hat 1.00) and 10 (R-hat 1.02). The stronger setting costs ~3x the sampling time.
SAMPLER = dict(chains=4, cores=2, draws=1000, tune=2000, target_accept=0.98)


def training_frame(world, extra_controls=()):
    d = world["df"].iloc[:N_TRAIN].copy()
    d["trend"] = np.arange(N_TRAIN) / N_TRAIN
    return d, d[["date"] + CHANNELS + CTRL + list(extra_controls)], d["sales"]


def hdi(a, prob=0.94):
    lo, hi = np.percentile(a, [(1 - prob) / 2 * 100, (1 + prob) / 2 * 100])
    return lo, hi


def _logsat(x, lam):
    return (1 - np.exp(-lam * x)) / (1 + np.exp(-lam * x))


def summarize(m, train_df, saturation="logistic"):
    """Everything downstream needs: ROI draws, response-curve parameters, diagnostics. Small, so the model can be dropped."""
    P = m.idata.posterior
    contrib = P["channel_contribution_original_scale"].sum("date").stack(s=("chain", "draw")).transpose("s", "channel").values
    roi = pd.DataFrame(contrib / train_df[CHANNELS].sum().to_numpy(), columns=CHANNELS)
    flat = lambda name: P[name].stack(s=("chain", "draw")).transpose("s", "channel").values
    names = ["saturation_lam", "saturation_beta"] if saturation == "logistic" else ["saturation_slope", "saturation_kappa", "saturation_beta"]
    draws = {n: flat(n) for n in names}
    idx = np.random.default_rng(0).choice(len(roi), size=min(400, len(roi)), replace=False)
    params = dict(kind=saturation, scale=np.asarray(m.channel_scale, dtype=float), target=float(np.asarray(m.target_scale)),
                  **{n.replace("saturation_", ""): v[idx] for n, v in draws.items()})
    diag = az.summary(m.idata, var_names=["adstock_alpha", *names], kind="diagnostics")
    return dict(
        roi=roi, params=params,
        marketing_share=float(contrib.sum(axis=1).mean() / train_df["sales"].sum()),
        max_rhat=float(diag["r_hat"].max()), min_ess=float(diag["ess_bulk"].min()),
        divergences=int(m.idata.sample_stats["diverging"].sum()),
        n_draws=int(P.sizes["chain"] * P.sizes["draw"]),
    )


def curve_draws(params, channel, spend):
    """Steady-state weekly incremental sales at constant weekly `spend` - one value per posterior draw."""
    j = CHANNELS.index(channel)
    x = spend / params["scale"][j]
    if params.get("kind", "logistic") == "hill":
        s_, k_ = params["slope"][:, j], params["kappa"][:, j]
        sat = x**s_ / (k_**s_ + x**s_)
    else:
        sat = _logsat(x, params["lam"][:, j])
    return params["target"] * params["beta"][:, j] * sat


def roi_table(summary):
    r = summary["roi"]
    return pd.DataFrame({"mean": r.mean(), "lo94": r.quantile(0.03), "hi94": r.quantile(0.97)})


def get_model(name, seed, world, lift_tests=None, informed_priors=False, extra_controls=(),
              saturation="logistic", roi_prior=None, cache_dir=Path("cache"), sampler=None, force_refit=False):
    """Fit (or load from cache) one MMM and return its small summary.

    saturation: "logistic" (PyMC-Marketing default) or "hill" (the family the simulator uses for the truth).
    roi_prior: None, or (median, sigma). Approximates an ROI prior (the approach Google's Meridian takes) with a LogNormal
        prior on each channel's saturation scale `beta`, centred so that a channel at typical saturation (0.6) and its mean
        spend implies an average ROI of `median`. A true ROI prior (an extra log-probability term on the model's own ROI)
        does not work with PyMC-Marketing 0.19's xtensor graph: NUTS cannot be assigned and PyMC falls back to Metropolis.
    """
    from pymc_marketing.mmm import GeometricAdstock, HillSaturation, LogisticSaturation, MMM
    from pymc_marketing.prior import Prior

    cache_dir = Path(cache_dir); cache_dir.mkdir(exist_ok=True)
    d, X, y = training_frame(world, extra_controls)
    # The cache key fingerprints everything that determines the posterior, so a stale file can never be loaded for a
    # different world, test, prior or sampler setting that happens to share a name and seed.
    key = hashlib.sha1(repr((
        X.to_csv(), y.to_csv(), None if lift_tests is None else lift_tests.round(6).to_csv(), informed_priors,
        tuple(extra_controls), saturation, roi_prior, sorted((sampler or SAMPLER).items()), seed, PYMC_MARKETING_VERSION,
    )).encode()).hexdigest()[:10]
    path = cache_dir / f"{name}_seed{seed}_{key}.nc"
    cfg = None
    if informed_priors:   # spend-share-scaled HalfNormal on saturation_beta
        share = d[CHANNELS].sum() / d[CHANNELS].sum().sum()
        cfg = {"saturation_beta": Prior("HalfNormal", sigma=len(CHANNELS) * share.to_numpy(), dims="channel")}
    if roi_prior is not None:
        median, sigma = roi_prior
        target_scale = float(np.abs(y).max())            # PyMC-Marketing scales the target by its max
        beta0 = median * d[CHANNELS].mean().to_numpy() / (target_scale * 0.6)
        cfg = {"saturation_beta": Prior("LogNormal", mu=np.log(beta0), sigma=sigma, dims="channel")}

    def build():
        return MMM(date_column="date", channel_columns=CHANNELS, adstock=GeometricAdstock(l_max=8),
                   saturation=HillSaturation() if saturation == "hill" else LogisticSaturation(),
                   control_columns=CTRL + list(extra_controls), yearly_seasonality=2, model_config=cfg)

    if path.exists() and not force_refit:
        m = MMM.load(str(path))
    else:
        m = build()
        if lift_tests is not None:
            m.build_model(X, y)
            m.add_lift_test_measurements(lift_tests)
        m.fit(X, y, random_seed=seed, progressbar=False, **(sampler or SAMPLER))
        m.save(str(path))
    out = summarize(m, d, saturation=saturation)
    del m; gc.collect()
    return out
