# -*- coding: utf-8 -*-
r"""
Freeze the winning hyperparameters of every method into one file.

The fold-seed study (fold_seed_study.py) has to refit every method 30 times.
Re-running each method's nested grid search 30 times is not affordable — the
tree run alone took 77 min and GLS-RF 95 min, so 30 seeds would be ~85 hours.
Instead each method is pinned at the configuration its own nested search
already chose, and only the FOLDS change.

What that measures, and what it does not: pinning the hyperparameters isolates
the variance contributed by the fold partition itself. It does not reproduce
the full nested-CV variance, part of which comes from re-tuning inside each new
partition. The study is therefore a lower bound on total run-to-run spread —
which is the right instrument for the question "is fold 1 a property of this
particular partition, or of the deposit?".

Selection rule (declared here, applied blind): for each method x target, take
the config the CV criterion selected most often across the 5 outer folds; break
ties by the best mean inner-CV score. No test-set information is involved.

Output: best_configs.json
"""

import os
import json
from collections import Counter
from pathlib import Path

import pandas as pd

# Every input and output hangs off one root. Defaults to the repository folder;
# set the SRF_PROJECT_ROOT environment variable to run against data elsewhere.
try:
    _HERE = Path(__file__).resolve().parent
except NameError:                       # notebook cell
    _HERE = Path.cwd()
RF = Path(os.environ.get("SRF_PROJECT_ROOT", _HERE))
# Resolved rather than hard-coded: a stale path here silently freezes a
# deployment config from a superseded run. GLS-RF is pinned to the exported
# REPORTED model (design fixed at ANISO/exp, mean-function prediction), not to
# the raw grid run, so the deployed field matches the paper.
def _newest(root, pattern, marker):
    hits = sorted((q for q in root.glob(pattern)
                   if q.is_dir() and any(q.glob(marker))),
                  key=lambda q: q.stat().st_mtime)
    if not hits:
        raise FileNotFoundError(f"No run matching {pattern}/{marker} under {root}")
    return hits[-1]


TREES_RUN = _newest(RF / "Classic_runs", "TREES_*", "best_params_by_fold.csv")
GLSRF_RUN = RF / "GLSRF_runs" / "GLSRF_reported_ANISOexp"
SRF_RUN = _newest(RF / "SRF_runs", "SRF_run_*",
                  "03_test_evaluation/predictions_long_SRF_LOCAL_ANISO_k3.csv")
OUT = RF / "best_configs.json"

GLSRF_CFG_COLS = ["Mode", "Kernel", "Azimuth", "Dip", "Tilt",
                  "A_major", "A_semi", "A_minor", "nugget", "n_estimators",
                  "max_depth", "min_leaf", "max_features", "max_samples",
                  "ridge", "jitter"]
SRF_CFG_COLS = ["n_estimators", "max_depth", "min_leaf", "max_features",
                "max_samples", "split_quantiles", "bootstrap"]


def modal_config(sub, cfg_key, score_col, higher_is_better=True):
    """Most frequently selected config; ties broken by mean inner score."""
    counts = Counter(sub[cfg_key])
    top = max(counts.values())
    tied = [c for c, n in counts.items() if n == top]
    if len(tied) == 1:
        chosen = tied[0]
    else:
        means = {c: sub.loc[sub[cfg_key] == c, score_col].mean() for c in tied}
        chosen = (max(means, key=means.get) if higher_is_better
                  else min(means, key=means.get))
    return chosen, counts[chosen], len(sub)


def main():
    out = {}

    # ---- tree baselines ----
    bp = pd.read_csv(TREES_RUN / "best_params_by_fold.csv")
    bp = bp[bp.selection == "cv"]
    for (target, method), sub in bp.groupby(["target", "method"]):
        cfg_json, n_sel, n_folds = modal_config(sub, "best_params_json", "SUBCV_R2")
        out.setdefault(method, {})[target] = {
            "params": json.loads(cfg_json),
            "selected_in_folds": f"{n_sel}/{n_folds}",
            "mean_inner_SUBCV_R2": float(sub.SUBCV_R2.mean()),
            "source": TREES_RUN.name}

    # ---- GLS-RF ----
    sc = pd.read_csv(GLSRF_RUN / "selected_configs.csv")
    sc["_key"] = sc[GLSRF_CFG_COLS].astype(str).agg("|".join, axis=1)
    score = "SUBCV_R2" if "SUBCV_R2" in sc.columns else None
    for target, sub in sc.groupby("target"):
        if score is None:
            sub = sub.assign(_s=0.0)
            key, n_sel, n_folds = modal_config(sub, "_key", "_s")
        else:
            key, n_sel, n_folds = modal_config(sub, "_key", score)
        row = sub[sub._key == key].iloc[0]
        out.setdefault("GLSRF", {})[target] = {
            "params": {c: (row[c].item() if hasattr(row[c], "item") else row[c])
                       for c in GLSRF_CFG_COLS},
            "selected_in_folds": f"{n_sel}/{n_folds}",
            "source": GLSRF_RUN.name}

    # ---- SRF ----
    tm = pd.read_csv(SRF_RUN / "03_test_evaluation" / "test_metrics.csv")
    tm["_key"] = tm[SRF_CFG_COLS].astype(str).agg("|".join, axis=1)
    for target, sub in tm.groupby("TARGET"):
        key, n_sel, n_folds = modal_config(sub, "_key", "BEST_SUBCV_R2")
        row = sub[sub._key == key].iloc[0]
        tgt = "DTR" if target == "DTR" else "Magnetic"
        out.setdefault("SRF", {})[tgt] = {
            "params": {c: (row[c].item() if hasattr(row[c], "item") else row[c])
                       for c in SRF_CFG_COLS},
            "selected_in_folds": f"{n_sel}/{n_folds}",
            "mean_inner_SUBCV_R2": float(sub.BEST_SUBCV_R2.mean()),
            "source": SRF_RUN.name}

    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("=" * 78)
    print("FROZEN CONFIGS (most frequently CV-selected across the 5 outer folds)")
    print("=" * 78)
    for method in ["RF", "RF_XYZ", "BAG", "GBM", "XGB", "GLSRF", "SRF"]:
        if method not in out:
            continue
        print(f"\n{method}")
        for target, d in out[method].items():
            p = d["params"]
            if method == "GLSRF":
                brief = (f"{p['Mode']}/{p['Kernel']} nugget={p['nugget']} depth={p['max_depth']} "
                         f"leaf={p['min_leaf']} mf={p['max_features']} n={p['n_estimators']}")
            else:
                brief = json.dumps(p, sort_keys=True)
            print(f"  {target:9s} [{d['selected_in_folds']} folds]  {brief}")
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
