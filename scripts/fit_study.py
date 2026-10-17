"""Fit (and cache) every model in the study. Resumable: models already in cache/ are loaded, not refit.

    python scripts/fit_study.py main          # core + variant models on the worked-example world
    python scripts/fit_study.py robust        # core models on the fresh worlds
"""
import sys
import time
import warnings

warnings.filterwarnings("ignore")

from mmmcal import study
from mmmcal.world import random_world


def timed(label, fn):
    t0 = time.time()
    out = fn()
    print(f"[{time.strftime('%H:%M:%S')}] {label}: {time.time() - t0:.0f}s", flush=True)
    return out


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "main"
    if stage == "main":
        w = random_world(study.MAIN_SEED)
        timed("main world: core models", lambda: study.fit_core(study.MAIN_SEED, w))
        timed("main world: variant models", lambda: study.fit_variants(study.MAIN_SEED, w))
    elif stage == "robust":
        for seed in study.ROBUST_SEEDS:
            w = random_world(seed)
            timed(f"world {seed}: core models", lambda s=seed, w=w: study.fit_core(s, w))
            timed(f"world {seed}: variants {study.ROBUST_EXTRA}", lambda s=seed, w=w: study.fit_variants(s, w, only=study.ROBUST_EXTRA))
    else:
        raise SystemExit(f"unknown stage {stage!r}")
    print("done", flush=True)
