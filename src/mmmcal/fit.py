"""Fitting PyMC-Marketing MMMs (with on-disk caching) and summarising the posteriors."""
from __future__ import annotations

import gc
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

from .world import CHANNELS, N_TRAIN

CTRL = ["trend", "black_friday", "christmas", "summer_sale"]
SAMPLER = dict(chains=4, cores=2, draws=800, tune=800, target_accept=0.95)


def training_frame(world, extra_controls=()):
    d = world["df"].iloc[:N_TRAIN].copy()
    d["trend"] = np.arange(N_TRAIN) / N_TRAIN
    return d, d[["date"] + CHANNELS + CTRL + list(extra_controls)], d["sales"]


def hdi(a, prob=0.94):
    lo, hi = np.percentile(a, [(1 - prob) / 2 * 100, (1 + prob) / 2 * 100])
    return lo, hi


def _logsat(x, lam):
    return (1 - np.exp(-lam * x)) / (1 + np.exp(-lam * x))


def summarize(m, train_df):
    """Everything downstream needs: ROI draws, response-curve parameters, diagnostics. Small, so the model can be dropped."""
    P = m.idata.posterior
    contrib = P["channel_contribution_original_scale"].sum("date").stack(s=("chain", "draw")).transpose("s", "channel").values
    roi = pd.DataFrame(contrib / train_df[CHANNELS].sum().to_numpy(), columns=CHANNELS)
    lam = P["saturation_lam"].stack(s=("chain", "draw")).transpose("s", "channel").values
    beta = P["saturation_beta"].stack(s=("chain", "draw")).transpose("s", "channel").values
    idx = np.random.default_rng(0).choice(lam.shape[0], size=min(400, lam.shape[0]), replace=False)
    params = dict(lam=lam[idx], beta=beta[idx], scale=np.asarray(m.channel_scale, dtype=float),
                  target=float(np.asarray(m.target_scale)))
    diag = az.summary(m.idata, var_names=["adstock_alpha", "saturation_lam", "saturation_beta"], kind="diagnostics")
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
    return params["target"] * params["beta"][:, j] * _logsat(spend / params["scale"][j], params["lam"][:, j])


def roi_table(summary):
    r = summary["roi"]
    return pd.DataFrame({"mean": r.mean(), "lo94": r.quantile(0.03), "hi94": r.quantile(0.97)})


def get_model(name, seed, world, lift_tests=None, informed_priors=False, extra_controls=(),
              cache_dir=Path("cache"), sampler=None, force_refit=False):
    """Fit (or load from cache) one MMM and return its small summary."""
    from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation, MMM
    from pymc_marketing.prior import Prior

    cache_dir = Path(cache_dir); cache_dir.mkdir(exist_ok=True)
    path = cache_dir / f"{name}_seed{seed}.nc"
    d, X, y = training_frame(world, extra_controls)
    cfg = None
    if informed_priors:   # spend-share-scaled HalfNormal on saturation_beta
        share = d[CHANNELS].sum() / d[CHANNELS].sum().sum()
        cfg = {"saturation_beta": Prior("HalfNormal", sigma=len(CHANNELS) * share.to_numpy(), dims="channel")}

    def build():
        return MMM(date_column="date", channel_columns=CHANNELS, adstock=GeometricAdstock(l_max=8),
                   saturation=LogisticSaturation(), control_columns=CTRL + list(extra_controls),
                   yearly_seasonality=2, model_config=cfg)

    if path.exists() and not force_refit:
        m = MMM.load(str(path))
    else:
        m = build()
        if lift_tests is not None:
            m.build_model(X, y)
            m.add_lift_test_measurements(lift_tests)
        m.fit(X, y, random_seed=seed, progressbar=False, **(sampler or SAMPLER))
        m.save(str(path))
    out = summarize(m, d)
    del m; gc.collect()
    return out
