# -*- coding: utf-8 -*-
r"""
TREE-ENSEMBLE BASELINES — the single, final runner.

The study is about trees, so the non-tree baselines (GLM, regression kriging)
are gone. What remains is a family of tree ensembles that differ in exactly one
mechanism at a time:

  RF       random forest on covariates only — bagging PLUS random feature
           selection at each split; coordinates deliberately withheld
  RF_XYZ   the identical forest with (x, y, z) appended as ordinary predictors.
           Same grid, same seeds, so RF_XYZ minus RF is the contribution of
           location as a plain covariate and nothing else
  BAG      bagged unpruned trees, every feature available at every split. The
           contrast with RF isolates random feature selection, since bagging
           is what is left of a random forest once you remove it
  GBM      gradient boosting — trees fitted sequentially on residuals rather
           than independently on bootstrap samples
  XGB      XGBoost — boosting with an explicit regularised objective, shrinkage
           and column subsampling

All five are regressors. All five are tuned by nested spatial cross-validation
on the frozen folds, and the bootstrap ensembles are ALSO tuned by out-of-bag
error, so the two selection criteria can be compared directly.

Two selection criteria
----------------------
  cv    hyperparameters chosen by pooled RMSE over the other frozen spatial
        folds intersected with the outer-train rows. Honest under spatial shift
  oob   hyperparameters chosen by out-of-bag RMSE on the outer-train rows

OOB exists only where bootstrap resampling exists: RF, RF_XYZ and BAG. Boosting
fits every tree on the full (or subsampled) training set in sequence, so there
is no out-of-bag set to score, and GBM/XGB therefore carry `cv` selection only.
That is a property of the algorithms, not an omission. To keep an optimistic
internal reference available for every method, the training-set fit is recorded
for all five as TRAIN_R2 — the gap between TRAIN/OOB and the honest fold score
is the quantity of interest.

Protocol per outer fold f:
  train = roles[:, f] == 0,  test = roles[:, f] == 1
  Every preprocessing step (median imputation) is fitted inside the current
  training split only. Outer-test rows are touched once, after selection.

Outputs (Classic_runs\TREES_<data stem>_<timestamp>\):
  predictions_long_TREES.csv     row,x,y,z,fold,target,method,selection,
                                 y_true,y_pred,y_unc
  fold_metrics.csv               per fold x target x method x selection,
                                 with SUBCV / OOB / TRAIN / TEST side by side
  pooled_summary.csv             pooled R2/RMSE/MAE/CCC
  oob_vs_cv.csv                  what each criterion selected and what it cost
  paired_tests.csv               Wilcoxon on per-sample squared errors
  inner_cv_search_summary.csv    every candidate, both criteria
  inner_cv_search_by_fold.csv    every candidate x inner-fold score
  best_params_by_fold.csv        selected parameters
  run_config.json                includes SHA1 fingerprints of data and folds

Run:  python -u classic_final.py
      --smoke   cut-down grids, to check the plumbing in a minute
Time: about 50 min for the full search on the case-study data, single-threaded.
"""

import hashlib
import json
import os
import sys
import time
import warnings
from datetime import datetime
from itertools import product
from pathlib import Path

try:
    _HERE = Path(__file__).resolve().parent
except NameError:  # notebook cell
    _HERE = Path.cwd()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

SMOKE = "--smoke" in sys.argv

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from sklearn.ensemble import (RandomForestRegressor, BaggingRegressor,
                              GradientBoostingRegressor)
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

# Every input and output hangs off one root. Defaults to the repository folder;
# set the SRF_PROJECT_ROOT environment variable to run against data elsewhere.
PROJECT_ROOT = Path(os.environ.get("SRF_PROJECT_ROOT", _HERE))

FOLDS_ROOT = PROJECT_ROOT / "CV_folds"
FOLDS_DIR = FOLDS_ROOT / "classic"

ROLES_PATH = FOLDS_DIR / "roles_classic.npy"
FOLDS_CSV = FOLDS_DIR / "folds_classic.csv"

# The data table and the fold roles are only meaningful together, and keeping
# them as two independent constants means every edit to one can silently
# contradict the other. make_folds.py already records which table it read, so
# the table is resolved FROM the folds rather than declared again here. Set
# DATA_PATH_OVERRIDE to pin a specific file (the row-count guard still applies).
DATA_PATH_OVERRIDE = None


def resolve_data_path():
    """The table the current folds were actually built from."""
    if DATA_PATH_OVERRIDE is not None:
        return Path(DATA_PATH_OVERRIDE)
    cfg_path = FOLDS_ROOT / "fold_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"{cfg_path} not found — run make_folds.py, or set "
            "DATA_PATH_OVERRIDE explicitly.")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    try:
        ds = cfg["datasets"]["classic"]
    except KeyError:
        raise KeyError(f"{cfg_path} has no 'classic' dataset — the folds were "
                       "built without one.")
    src = Path(ds["source"])
    if not src.exists():
        # fold_config.json records an absolute path; fall back to the same file
        # name under PROJECT_ROOT if the project has been moved or cloned.
        local = PROJECT_ROOT / src.name
        if not local.exists():
            raise FileNotFoundError(
                f"The folds were built from {src}, which no longer exists, and "
                f"there is no {src.name} under {PROJECT_ROOT}.")
        src = local
    print(f"[data resolved from fold_config.json] {src.name} "
          f"({ds['n_samples']} samples, built {cfg['created_at'][:19]})")
    return src


DATA_PATH = resolve_data_path()

COORD_COLS = ["XC", "YC", "ZC"]
FEATURE_COLS = ["FE", "FEO", "Magsus", "S"]
TARGET_COLS = ["DTR", "Magnetic"]

RUNS_BASE = PROJECT_ROOT / "Classic_runs"

METHODS = ["RF", "RF_XYZ", "BAG", "GBM", "XGB"]

# RF and RF_XYZ share this grid on purpose: the pair is only interpretable as
# an isolated test of "does location help?" if nothing else differs.
RF_PARAM_GRID = {
    "n_estimators": [400],
    "max_depth": [None, 12, 24],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features": ["sqrt", 0.5, 0.75],
}
# Bagging = a random forest with feature subsampling switched off, so
# max_features is pinned at 1.0 and the tree grid mirrors RF's.
BAG_PARAM_GRID = {
    "n_estimators": [400],
    "max_samples": [0.7, 1.0],
    "tree_max_depth": [None, 12, 24],
    "tree_min_samples_leaf": [1, 2, 4, 8],
}
GBM_PARAM_GRID = {
    "n_estimators": [200, 400],
    "learning_rate": [0.03, 0.1],
    "max_depth": [2, 3, 4],
    "min_samples_leaf": [1, 4],
    "subsample": [0.7, 1.0],
}
XGB_PARAM_GRID = {
    "n_estimators": [400],
    "learning_rate": [0.03, 0.1],
    "max_depth": [2, 3, 4],
    "subsample": [0.7, 1.0],
    "colsample_bytree": [0.7, 1.0],
    "min_child_weight": [1, 5],
}

PARAM_GRIDS = {"RF": RF_PARAM_GRID, "RF_XYZ": RF_PARAM_GRID,
               "BAG": BAG_PARAM_GRID, "GBM": GBM_PARAM_GRID,
               "XGB": XGB_PARAM_GRID}

# Which methods use coordinates as predictors, and which can produce an
# out-of-bag estimate (bootstrap ensembles only).
USES_COORDS = {"RF": False, "RF_XYZ": True, "BAG": False,
               "GBM": False, "XGB": False}
HAS_OOB = {"RF": True, "RF_XYZ": True, "BAG": True,
           "GBM": False, "XGB": False}

SELECTIONS = ["cv", "oob"]

# Single-threaded by default. The reported run was produced this way, and the
# sklearn/xgboost estimators here are seeded but not thread-order independent,
# so changing this can move the last decimals. Raise it if you accept that.
N_JOBS = int(os.environ.get("CLASSIC_N_JOBS", "1"))

SEED = 42
FOLDS_TO_RUN = None       # None = all folds; or e.g. [0, 1]

if SMOKE:
    # Plumbing check only: a handful of candidates per method, small ensembles.
    print("[WARN] --smoke: cut-down grids. NOT paper numbers.")
    RF_PARAM_GRID = {"n_estimators": [50], "max_depth": [12],
                     "min_samples_leaf": [1, 4], "max_features": ["sqrt"]}
    BAG_PARAM_GRID = {"n_estimators": [50], "max_samples": [0.7],
                      "tree_max_depth": [12], "tree_min_samples_leaf": [1, 4]}
    GBM_PARAM_GRID = {"n_estimators": [50], "learning_rate": [0.1],
                      "max_depth": [3], "min_samples_leaf": [1, 4],
                      "subsample": [0.7]}
    XGB_PARAM_GRID = {"n_estimators": [50], "learning_rate": [0.1],
                      "max_depth": [3], "subsample": [0.7],
                      "colsample_bytree": [0.7], "min_child_weight": [1, 5]}
    PARAM_GRIDS = {"RF": RF_PARAM_GRID, "RF_XYZ": RF_PARAM_GRID,
                   "BAG": BAG_PARAM_GRID, "GBM": GBM_PARAM_GRID,
                   "XGB": XGB_PARAM_GRID}

BAD_TOKENS = ["-", "--", "", " ", "NA", "N/A", "na", "null", "NULL", "None", "none"]


# ============================================================
# Helpers
# ============================================================

def regression_metrics(y_true, y_pred):
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(((yt - yp) ** 2).sum())
    ss_tot = float(((yt - yt.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / max(ss_tot, 1e-300)
    rmse = float(np.sqrt(ss_res / yt.size))
    mae = float(np.abs(yt - yp).mean())
    return r2, rmse, mae


def fingerprint(arr):
    """Stable SHA1 of an array's bytes — identifies the exact inputs of a run."""
    a = np.ascontiguousarray(np.asarray(arr))
    return hashlib.sha1(a.tobytes()).hexdigest()[:16]


class MedianImputer:
    """Per-fold, train-only median imputation for a feature matrix."""
    def __init__(self):
        self.med_ = None

    def fit(self, X):
        self.med_ = np.nanmedian(X, axis=0)
        return self

    def transform(self, X):
        X = X.copy()
        for j in range(X.shape[1]):
            bad = np.isnan(X[:, j])
            X[bad, j] = self.med_[j]
        return X


# ============================================================
# Estimators
# ============================================================

def build_estimator(method, params, seed, want_oob):
    """One estimator, configured. want_oob only matters for the baggers."""
    if method in ("RF", "RF_XYZ"):
        return RandomForestRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            bootstrap=True,
            oob_score=bool(want_oob),
            n_jobs=N_JOBS, random_state=seed)

    if method == "BAG":
        # max_features=1.0 is what makes this bagging rather than a forest:
        # every split sees every predictor, so the only randomness is the
        # bootstrap sample.
        base = DecisionTreeRegressor(
            max_depth=params["tree_max_depth"],
            min_samples_leaf=params["tree_min_samples_leaf"],
            random_state=seed)
        return BaggingRegressor(
            estimator=base,
            n_estimators=params["n_estimators"],
            max_samples=params["max_samples"],
            max_features=1.0,
            bootstrap=True,
            oob_score=bool(want_oob),
            n_jobs=N_JOBS, random_state=seed)

    if method == "GBM":
        return GradientBoostingRegressor(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            subsample=params["subsample"],
            random_state=seed)

    if method == "XGB":
        return XGBRegressor(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            min_child_weight=params["min_child_weight"],
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=N_JOBS, random_state=seed, verbosity=0)

    raise ValueError(f"Unknown method {method}")


def design_matrices(method, feats_tr, coords_tr, feats_te, coords_te):
    """Train-only imputation, then attach coordinates if the method wants them."""
    imp = MedianImputer().fit(feats_tr)
    Xtr, Xte = imp.transform(feats_tr), imp.transform(feats_te)
    if USES_COORDS[method]:
        Xtr = np.column_stack([coords_tr, Xtr])
        Xte = np.column_stack([coords_te, Xte])
    return Xtr, Xte


def member_spread(est, X):
    """Ensemble disagreement, where the members are independent predictors.

    Meaningful for bagged ensembles. Boosting members are increments of one
    additive model, so their spread is not an uncertainty and is not reported.
    """
    members = getattr(est, "estimators_", None)
    if members is None or not isinstance(est, (RandomForestRegressor,
                                               BaggingRegressor)):
        return np.full(len(X), np.nan)
    try:
        if isinstance(est, BaggingRegressor):
            preds = np.stack([m.predict(X[:, f]) for m, f in
                              zip(members, est.estimators_features_)])
        else:
            preds = np.stack([m.predict(X) for m in members])
        return preds.std(axis=0)
    except Exception:
        return np.full(len(X), np.nan)


def fit_predict(method, feats_tr, coords_tr, y_tr, feats_te, coords_te,
                seed, params, want_oob=False):
    """Fit on train, predict test. Optionally also return the OOB and train fits."""
    Xtr, Xte = design_matrices(method, feats_tr, coords_tr, feats_te, coords_te)
    est = build_estimator(method, params, seed, want_oob and HAS_OOB[method])
    est.fit(Xtr, y_tr)
    pred = np.asarray(est.predict(Xte), dtype=np.float64)
    unc = member_spread(est, Xte)

    oob_pred = None
    if want_oob and HAS_OOB[method]:
        op = getattr(est, "oob_prediction_", None)
        if op is not None:
            oob_pred = np.asarray(op, dtype=np.float64)
    train_pred = np.asarray(est.predict(Xtr), dtype=np.float64) if want_oob else None
    return pred, unc, oob_pred, train_pred


# ============================================================
# Nested spatial search, scored by BOTH criteria
# ============================================================

def expand_param_grid(grid):
    keys = list(grid)
    return [dict(zip(keys, values))
            for values in product(*(grid[key] for key in keys))]


def method_candidates(method):
    return expand_param_grid(PARAM_GRIDS[method])


def params_json(params):
    return json.dumps(params, sort_keys=True, separators=(",", ":"))


def make_inner_splits(roles, outer_fold, valid_mask):
    """Reuse the other frozen spatial folds inside the current outer-train set."""
    outer_train = (roles[:, outer_fold] == 0) & valid_mask
    splits = []
    for inner_fold in range(roles.shape[1]):
        if inner_fold == outer_fold:
            continue
        inner_train = outer_train & (roles[:, inner_fold] == 0)
        inner_valid = outer_train & (roles[:, inner_fold] == 1)
        if int(inner_train.sum()) >= 10 and int(inner_valid.sum()) >= 2:
            splits.append((inner_fold, inner_train, inner_valid))
    if len(splits) < 2:
        raise ValueError(f"Outer fold {outer_fold + 1} has fewer than two "
                         "usable spatial inner folds.")
    return splits


def tune_method(method, candidates, feats_all, coords_all, y_all, roles, outer_fold):
    """Score every candidate under both criteria; return one winner per criterion."""
    valid_mask = ~np.isnan(y_all) & np.isfinite(coords_all).all(axis=1)
    inner_splits = make_inner_splits(roles, outer_fold, valid_mask)
    outer_train = (roles[:, outer_fold] == 0) & valid_mask
    summary_rows, detail_rows = [], []

    for candidate_id, params in enumerate(candidates, start=1):
        pooled_true, pooled_pred = [], []
        error_message = ""

        # ---- criterion 1: pooled spatial inner-CV ----
        for inner_fold, inner_train, inner_valid in inner_splits:
            started = time.time()
            try:
                pred, _, _, _ = fit_predict(
                    method,
                    feats_all[inner_train], coords_all[inner_train], y_all[inner_train],
                    feats_all[inner_valid], coords_all[inner_valid],
                    SEED + outer_fold * 1000 + inner_fold, params)
                truth = y_all[inner_valid]
                r2, rmse, mae = regression_metrics(truth, pred)
                pooled_true.append(truth)
                pooled_pred.append(pred)
                status, error = "ok", ""
            except Exception as exc:
                r2 = rmse = mae = np.nan
                status = "failed"
                error = f"{type(exc).__name__}: {exc}"
                error_message = error

            detail_rows.append({
                "method": method, "outer_fold": outer_fold + 1,
                "candidate_id": candidate_id, "inner_fold": inner_fold + 1,
                "n_train": int(inner_train.sum()), "n_valid": int(inner_valid.sum()),
                "R2": r2, "RMSE": rmse, "MAE": mae,
                "status": status, "error": error,
                "params_json": params_json(params),
                "fit_time_sec": round(time.time() - started, 3)})
            if status == "failed":
                break

        if error_message or len(pooled_true) != len(inner_splits):
            cv_r2 = cv_rmse = cv_mae = np.nan
            status = "failed"
        else:
            cv_r2, cv_rmse, cv_mae = regression_metrics(
                np.concatenate(pooled_true), np.concatenate(pooled_pred))
            status = "ok"

        # ---- criterion 2: out-of-bag on the outer-train rows ----
        oob_r2 = oob_rmse = oob_mae = np.nan
        train_r2 = np.nan
        if status == "ok":
            try:
                _, _, oob_pred, train_pred = fit_predict(
                    method,
                    feats_all[outer_train], coords_all[outer_train], y_all[outer_train],
                    feats_all[outer_train], coords_all[outer_train],
                    SEED + outer_fold, params, want_oob=True)
                y_ot = y_all[outer_train]
                if oob_pred is not None:
                    seen = np.isfinite(oob_pred)
                    if seen.sum() >= 3:
                        oob_r2, oob_rmse, oob_mae = regression_metrics(
                            y_ot[seen], oob_pred[seen])
                if train_pred is not None:
                    train_r2, _, _ = regression_metrics(y_ot, train_pred)
            except Exception as exc:
                error_message = error_message or f"{type(exc).__name__}: {exc}"

        summary_rows.append({
            "method": method, "outer_fold": outer_fold + 1,
            "candidate_id": candidate_id, "n_inner_folds": len(pooled_true),
            "SUBCV_R2": cv_r2, "SUBCV_RMSE": cv_rmse, "SUBCV_MAE": cv_mae,
            "OOB_R2": oob_r2, "OOB_RMSE": oob_rmse, "OOB_MAE": oob_mae,
            "TRAIN_R2": train_r2,
            "status": status, "error": error_message,
            "params_json": params_json(params)})

    summary = pd.DataFrame(summary_rows)
    ok = summary[summary["status"].eq("ok")]
    if ok.empty:
        raise RuntimeError(f"All {len(candidates)} candidates failed for {method}.")

    best = {}
    for sel, col in (("cv", "SUBCV_RMSE"), ("oob", "OOB_RMSE")):
        usable = ok[np.isfinite(ok[col])]
        if usable.empty:
            best[sel] = None          # OOB is undefined for boosting
            continue
        order = usable.sort_values([col, "candidate_id"])["candidate_id"].tolist()
        summary[f"rank_{sel}"] = summary["candidate_id"].map(
            {cid: r for r, cid in enumerate(order, 1)})
        summary[f"selected_{sel}"] = summary["candidate_id"].eq(int(order[0]))
        best[sel] = (int(order[0]), dict(candidates[int(order[0]) - 1]))

    return best, summary, pd.DataFrame(detail_rows), \
        [f + 1 for f, _, _ in inner_splits]


# ============================================================
# Reporting
# ============================================================

def pooled_metrics(yt, yp):
    yt = np.asarray(yt, dtype=np.float64)
    yp = np.asarray(yp, dtype=np.float64)
    ss_res = float(((yt - yp) ** 2).sum())
    ss_tot = float(((yt - yt.mean()) ** 2).sum())
    cov = float(((yt - yt.mean()) * (yp - yp.mean())).mean())
    return {"n": int(yt.size),
            "POOLED_R2": 1.0 - ss_res / max(ss_tot, 1e-300),
            "POOLED_RMSE": float(np.sqrt(ss_res / yt.size)),
            "POOLED_MAE": float(np.abs(yt - yp).mean()),
            "POOLED_CCC": float(2.0 * cov / (yt.var() + yp.var()
                                             + (yt.mean() - yp.mean()) ** 2))}


def paired_tests(df_long, selection="cv"):
    """Wilcoxon signed-rank on per-sample squared errors, every method pair.

    Pooled R2 cannot say whether a gap is systematic or the work of a handful of
    samples; the paired test can, because every method predicts the same rows in
    the same folds.
    """
    sub0 = df_long[df_long.selection == selection]
    rows = []
    for target in TARGET_COLS:
        sub = sub0[sub0.target == target]
        err = {}
        for method in METHODS:
            m = sub[sub.method == method].sort_values("row")
            if len(m):
                err[method] = ((m.y_true.values - m.y_pred.values) ** 2,
                               m.row.values)
        for i, a in enumerate(METHODS):
            for b in METHODS[i + 1:]:
                if a not in err or b not in err:
                    continue
                ea, ra = err[a]
                eb, rb = err[b]
                if not np.array_equal(ra, rb):
                    raise RuntimeError(f"{a} vs {b}/{target}: rows do not align.")
                stat, p = wilcoxon(ea, eb)
                rows.append({"selection": selection, "target": target,
                             "method_a": a, "method_b": b,
                             "mean_sq_err_a": ea.mean(), "mean_sq_err_b": eb.mean(),
                             "wilcoxon_p": p,
                             "verdict": (f"{a} better" if p < 0.05 and ea.mean() < eb.mean()
                                         else f"{b} better" if p < 0.05 else "tie")})
    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def run_folds():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_BASE / f"TREES_{DATA_PATH.stem}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("TREE-ENSEMBLE BASELINES - FINAL RUNNER")
    print(f"run dir : {run_dir}")
    print(f"data    : {DATA_PATH}")
    print(f"folds   : {ROLES_PATH}")
    print(f"methods : {METHODS}   (OOB available for "
          f"{[m for m in METHODS if HAS_OOB[m]]})")
    print("=" * 78)

    # ---- data ----
    if DATA_PATH.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(DATA_PATH)
    else:
        df = pd.read_csv(DATA_PATH, low_memory=False)
    for c in COORD_COLS + FEATURE_COLS + TARGET_COLS:
        if c not in df.columns:
            raise KeyError(f"Missing column {c} in {DATA_PATH}")
        df[c] = pd.to_numeric(df[c].replace(BAD_TOKENS, np.nan), errors="coerce")

    roles = np.load(ROLES_PATH)
    if roles.shape[0] != len(df):
        raise ValueError(
            f"roles rows ({roles.shape[0]}) != data rows ({len(df)}) — "
            f"{ROLES_PATH.name} was not built from {DATA_PATH.name}. "
            "Either re-run make_folds.py against this table, or clear "
            "DATA_PATH_OVERRIDE so the table is taken from fold_config.json.")
    n_folds = roles.shape[1]

    fdf = pd.read_csv(FOLDS_CSV)
    d = np.abs(df[COORD_COLS].values - fdf[["x", "y", "z"]].values)
    if np.nanmax(d) > 0.01:
        raise ValueError("Coordinates do not match the folds CSV row-by-row "
                         f"(max diff {np.nanmax(d):.3f} m). Row order changed?")

    coords_all = df[COORD_COLS].values.astype(np.float64)
    feats_all = df[FEATURE_COLS].values.astype(np.float64)
    coord_fp, roles_fp = fingerprint(coords_all), fingerprint(roles)
    print(f"rows {len(df)} | folds {n_folds} | row-order check OK")
    print(f"coord fingerprint {coord_fp} | roles fingerprint {roles_fp}")
    print(f"test n per fold: {[int((roles[:, f] == 1).sum()) for f in range(n_folds)]}")
    print(f"candidates: " + ", ".join(f"{m}={len(method_candidates(m))}"
                                      for m in METHODS) + "\n")

    fold_list = list(range(n_folds)) if FOLDS_TO_RUN is None else list(FOLDS_TO_RUN)

    long_rows, metric_rows = [], []
    search_summary_frames, search_detail_frames, best_params_rows = [], [], []
    t_start = time.time()

    for target in TARGET_COLS:
        y_all = df[target].values.astype(np.float64)

        for f in fold_list:
            fold_no = f + 1
            tr = (roles[:, f] == 0) & ~np.isnan(y_all)
            te = roles[:, f] == 1
            te_rows = np.where(te)[0]
            y_te = y_all[te]
            ok_true = ~np.isnan(y_te)
            print(f"TARGET {target} | FOLD {fold_no}/{n_folds} | "
                  f"train={int(tr.sum())} test={int(te.sum())}")

            for method in METHODS:
                t0 = time.time()
                candidates = method_candidates(method)
                try:
                    best, search_summary, search_details, inner_folds = tune_method(
                        method, candidates, feats_all, coords_all, y_all, roles, f)
                except Exception as e:
                    print(f"  [{method}] TUNING FAILED: {e}")
                    continue

                search_summary.insert(0, "target", target)
                search_details.insert(0, "target", target)
                search_summary_frames.append(search_summary)
                search_detail_frames.append(search_details)

                for sel in SELECTIONS:
                    if best.get(sel) is None:
                        continue
                    cand_id, best_params = best[sel]
                    row = search_summary[
                        search_summary.candidate_id.eq(cand_id)].iloc[0]

                    try:
                        y_pred, y_unc, _, _ = fit_predict(
                            method,
                            feats_all[tr], coords_all[tr], y_all[tr],
                            feats_all[te], coords_all[te],
                            SEED + f, best_params)
                    except Exception as e:
                        print(f"  [{method}/{sel}] REFIT FAILED: {e}")
                        continue

                    best_params_rows.append({
                        "target": target, "outer_fold": fold_no, "method": method,
                        "selection": sel,
                        "selection_metric": ("pooled spatial inner-CV RMSE"
                                             if sel == "cv" else "out-of-bag RMSE"),
                        "inner_folds": ",".join(map(str, inner_folds)),
                        "SUBCV_R2": row["SUBCV_R2"], "OOB_R2": row["OOB_R2"],
                        "TRAIN_R2": row["TRAIN_R2"],
                        "best_params_json": params_json(best_params)})

                    for i, r_id in enumerate(te_rows):
                        long_rows.append({
                            "row": int(r_id),
                            "x": coords_all[r_id, 0], "y": coords_all[r_id, 1],
                            "z": coords_all[r_id, 2],
                            "fold": fold_no, "target": target, "method": method,
                            "selection": sel,
                            "y_true": y_te[i], "y_pred": y_pred[i],
                            "y_unc": float(y_unc[i])})

                    if ok_true.sum() >= 2:
                        r2, rmse, mae = regression_metrics(y_te[ok_true],
                                                           y_pred[ok_true])
                    else:
                        r2 = rmse = mae = np.nan
                    metric_rows.append({
                        "target": target, "fold": fold_no, "method": method,
                        "selection": sel,
                        "n_train": int(tr.sum()), "n_test": int(ok_true.sum()),
                        "SUBCV_R2": row["SUBCV_R2"], "OOB_R2": row["OOB_R2"],
                        "TRAIN_R2": row["TRAIN_R2"],
                        "TEST_R2": r2, "TEST_RMSE": rmse, "TEST_MAE": mae,
                        "best_params_json": params_json(best_params)})
                    print(f"  [{method:6s}/{sel:3s}] TEST R2={r2:6.3f} "
                          f"RMSE={rmse:7.3f} | SUBCV={row['SUBCV_R2']:6.3f} "
                          f"OOB={row['OOB_R2']:6.3f} TRAIN={row['TRAIN_R2']:6.3f}")
                print(f"  ...{method} done in {time.time() - t0:.1f} s")

    elapsed = time.time() - t_start

    # ---- save ----
    df_long = pd.DataFrame(long_rows)
    df_long.to_csv(run_dir / "predictions_long_TREES.csv", index=False)

    df_metrics = pd.DataFrame(metric_rows)
    df_metrics.to_csv(run_dir / "fold_metrics.csv", index=False)

    pd.concat(search_summary_frames, ignore_index=True).to_csv(
        run_dir / "inner_cv_search_summary.csv", index=False)
    pd.concat(search_detail_frames, ignore_index=True).to_csv(
        run_dir / "inner_cv_search_by_fold.csv", index=False)
    df_best = pd.DataFrame(best_params_rows)
    df_best.to_csv(run_dir / "best_params_by_fold.csv", index=False)

    pooled_rows = []
    for target in TARGET_COLS:
        for method in METHODS:
            for sel in SELECTIONS:
                sl = df_long[(df_long.target == target)
                             & (df_long.method == method)
                             & (df_long.selection == sel)].dropna(subset=["y_true"])
                if len(sl) < 3:
                    continue
                pooled_rows.append({"target": target, "method": method,
                                    "selection": sel,
                                    **pooled_metrics(sl.y_true.values,
                                                     sl.y_pred.values)})
    df_pooled = pd.DataFrame(pooled_rows)
    df_pooled.to_csv(run_dir / "pooled_summary.csv", index=False)

    # what did each criterion cost?
    oob_vs_cv = (df_pooled.pivot_table(index=["target", "method"],
                                       columns="selection",
                                       values="POOLED_R2")
                 .reset_index())
    if {"cv", "oob"} <= set(oob_vs_cv.columns):
        oob_vs_cv["cv_minus_oob"] = oob_vs_cv["cv"] - oob_vs_cv["oob"]
    oob_vs_cv.to_csv(run_dir / "oob_vs_cv.csv", index=False)

    df_paired = pd.concat([paired_tests(df_long, s) for s in SELECTIONS
                           if (df_long.selection == s).any()], ignore_index=True)
    df_paired.to_csv(run_dir / "paired_tests.csv", index=False)

    with open(run_dir / "run_config.json", "w", encoding="utf-8") as fjs:
        json.dump({
            "created_at": datetime.now().isoformat(),
            "script": "classic_final.py",
            "data_path": str(DATA_PATH), "n_rows": int(len(df)),
            "roles_path": str(ROLES_PATH),
            "coord_fingerprint_sha1": coord_fp,
            "roles_fingerprint_sha1": roles_fp,
            "coord_cols": COORD_COLS, "feature_cols": FEATURE_COLS,
            "target_cols": TARGET_COLS, "methods": METHODS,
            "uses_coords": USES_COORDS, "has_oob": HAS_OOB,
            "selections": SELECTIONS,
            "parameter_grids": {m: PARAM_GRIDS[m] for m in METHODS},
            "candidate_counts": {m: len(method_candidates(m)) for m in METHODS},
            "nested_cv": {
                "inner_split": "other frozen spatial folds intersected with outer-train",
                "selection_metrics": {"cv": "pooled inner-fold RMSE",
                                      "oob": "out-of-bag RMSE on outer-train"},
                "oob_undefined_for": [m for m in METHODS if not HAS_OOB[m]],
                "outer_test_used_for_tuning": False,
                "preprocessing_fit_scope": "current inner-train or outer-train only"},
            "n_jobs": N_JOBS, "seed": SEED, "smoke": bool(SMOKE),
            "elapsed_sec": elapsed}, fjs, indent=2)

    # ---- report ----
    fmt = lambda v: f"{v:.3f}"
    print("\n" + "=" * 78)
    print("POOLED SUMMARY - CV-selected (headline)")
    print("=" * 78)
    for target in TARGET_COLS:
        print(f"\n  --- {target} ---")
        print(df_pooled[(df_pooled.target == target) & (df_pooled.selection == "cv")]
              .sort_values("POOLED_R2", ascending=False)
              [["method", "n", "POOLED_R2", "POOLED_RMSE", "POOLED_MAE", "POOLED_CCC"]]
              .to_string(index=False, float_format=fmt))

    print("\n" + "=" * 78)
    print("OOB-SELECTED vs CV-SELECTED (pooled test R2; boosting has no OOB)")
    print("=" * 78)
    print(oob_vs_cv.to_string(index=False, float_format=fmt))

    print("\n" + "=" * 78)
    print("INTERNAL ESTIMATE vs HONEST FOLD SCORE (CV-selected models)")
    print("=" * 78)
    gap = (df_metrics[df_metrics.selection == "cv"]
           .groupby(["target", "method"])[["TRAIN_R2", "OOB_R2", "SUBCV_R2", "TEST_R2"]]
           .mean().reset_index())
    gap["TRAIN_minus_TEST"] = gap["TRAIN_R2"] - gap["TEST_R2"]
    gap["OOB_minus_TEST"] = gap["OOB_R2"] - gap["TEST_R2"]
    print(gap.to_string(index=False, float_format=fmt))

    print("\n" + "=" * 78)
    print("PER-FOLD TEST R2 (CV-selected)")
    print("=" * 78)
    for target in TARGET_COLS:
        piv = (df_metrics[(df_metrics.target == target)
                          & (df_metrics.selection == "cv")]
               .pivot(index="method", columns="fold", values="TEST_R2")
               .reindex(METHODS))
        print(f"\n  --- {target} ---")
        print(piv.to_string(float_format=fmt))

    print("\n" + "=" * 78)
    print("PAIRED WILCOXON on per-sample squared errors (CV-selected)")
    print("=" * 78)
    print(df_paired[df_paired.selection == "cv"]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print(f"\nSaved to {run_dir}")
    print(f"Elapsed  {elapsed / 60:.1f} min")
    print("=" * 78)
    return run_dir


if __name__ == "__main__":
    run_folds()
