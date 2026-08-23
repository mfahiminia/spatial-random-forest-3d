# -*- coding: utf-8 -*-
r"""
SRF trainer (regression + classification) — organized run outputs.

SPLIT_MODE:
  "folds"  : spatial block CV from make_folds.py (CV_folds roles file) — the
             reported evaluation. Per fold f: train = roles[:,f]==0,
             test = roles[:,f]==1, buffer (==2) used by nobody. Hyperparameters
             are tuned INSIDE each fold's training rows only; the selected
             config is refit per fold and scored on that fold's test rows.
             Emits the long prediction table (predictions_long_*.csv) that the
             method-comparison harness consumes.
  "random" : 80/20 random split. Optimistic on overlapping voxel windows; kept
             only as the counter-example the validation section argues against.
  "none"   : all rows train (no evaluation split) — used to export the
             zone-of-influence importance maps on all data.

Run folder layout (<root>\SRF_runs\SRF_run_YYYYMMDD_HHMMSS\):
  00_metadata\run_config.json, prepared.pkl
  01_grid_search\grid_results_{TARGET}[ _foldN].csv
  02_models\best_{TARGET}[ _foldN].joblib, meta_{TARGET}.json
  03_test_evaluation\predictions_long_SRF.csv   (row,x,y,z,fold,target,method,
                                                 y_true,y_pred,y_unc)
                    fold_metrics_{TARGET}.csv, test_metrics.csv
  04_importance\...                              (single-split modes only)
  05_full_prediction\                            (filled by srf_predict.py)
  summary.csv

Run:  python srf_train.py     (or %run in Jupyter)
"""

import json
import hashlib
import os
import sys
import time
from datetime import datetime
from itertools import product
from pathlib import Path

# Make srf_core_final importable no matter where this is run from.
try:
    _HERE = Path(__file__).resolve().parent
except NameError:  # notebook cell
    _HERE = Path.cwd()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import numpy as np
import pandas as pd
import joblib
from joblib import Parallel, delayed

# Reload guard: a Jupyter kernel may hold an older srf_core_final in memory.
# Version-based check — attribute checks go stale, the version number cannot.
import importlib
import srf_core_final as _core
_NEED_CORE_API = 8
if getattr(_core, "CORE_API_VERSION", 0) < _NEED_CORE_API:
    _core = importlib.reload(_core)
if getattr(_core, "CORE_API_VERSION", 0) < _NEED_CORE_API:
    raise RuntimeError(
        f"srf_core_final is outdated (need API v{_NEED_CORE_API}, got "
        f"v{getattr(_core, 'CORE_API_VERSION', 0)}). The file on disk is old or "
        f"a stale copy shadows it ({getattr(_core, '__file__', '?')}) — "
        "restart the kernel or check sys.path.")

from srf_core_final import (
    build_forest_from_config,
    generate_z_rotations_4,
    build_rotation_index_maps,
    voxel_maps_from_feature_maps,
    build_aniso_weights_and_masks,
    build_aniso_features,
    rotate_voxel_flats,
    build_feature_names,
    reshape_base_importance_to_zones,
    normalized_entropy,
    ANISO_FEAT_NAMES_8,
)


# ============================================================
# 0) USER CONFIG
# ============================================================

# ---- Pattern + kernel variant: these two switches drive every dependent path,
#      the method label, and the augmentation/anisotropy switches. -------------
# The four "local" variants are the 2x2 of augmentation x anisotropy on the
# SAME local X, so the two can be separated instead of moving together:
#   "local"       : augmentation + anisotropy summaries — the paper's SRF. DEFAULT.
#   "local_aug"   : augmentation only, no anisotropy block.
#   "local_aniso" : anisotropy block only, no augmentation — patterns stay in
#                   the true geological frame.
#   "local_plain" : neither.
# "multiscale" : appends the fixed directional features built by
#                build_multiscale_vectors.py (not augmented/rotated).
#
# Default is "local_aniso": the 6-arm ablation showed z-rotation augmentation
# degrades the honest sub-CV metric in all four paired contrasts (Wilcoxon
# p=0.0083) because a 90 deg rotation maps the 120 m major continuity direction
# onto the 50 m semi direction — geology that does not exist in this deposit.
# The anisotropy block on its own is positive. Selected on sub-CV only.
PATTERN_VARIANT = os.environ.get(
    "SRF_PATTERN_VARIANT", "local_aniso").strip().lower()
# "k5" = 5x5x5 (125 voxels, X_train_125) | "k3" = 3x3x3 (27 voxels, X_train_27)
KERNEL_VARIANT = os.environ.get(
    "SRF_KERNEL_VARIANT", "k3").strip().lower()
QUICK_RUN = os.environ.get(
    "SRF_QUICK", "0").strip().lower() in ("1", "true", "yes")

_KV = {"k5": {"k": 5, "tag": "125"}, "k3": {"k": 3, "tag": "27"}}

# pattern variant -> (USE_AUGMENTATION, USE_ANISO_FEATS)
PATTERN_VARIANTS = {
    "local":       (True, True),
    "local_aug":   (True, False),
    "local_aniso": (False, True),
    "local_plain": (False, False),
    "multiscale":  (False, False),
}

# Every input and output hangs off one root. Defaults to the repository folder;
# set the SRF_PROJECT_ROOT environment variable to run against data elsewhere.
PROJECT_ROOT = Path(os.environ.get("SRF_PROJECT_ROOT", _HERE))

VECTOR_DIR = PROJECT_ROOT / "Vector"       # from build_voxels.py
FOLDS_DIR = PROJECT_ROOT / "CV_folds"      # from make_folds.py
RUNS_BASE = PROJECT_ROOT / "SRF_runs"      # written by this script


def set_variant(pattern=None, kernel=None, folds_dir=None):
    """Set PATTERN_VARIANT / KERNEL_VARIANT and re-derive EVERY dependent global
    in one place: feature paths, method label, augmentation and anisotropy
    switches, roles/blocks files.

    Driver scripts (srf_ablation_run.py, srf_sensitivity.py, ...) must go
    through this instead of assigning individual globals. Setting X_path by hand
    while PATTERN_VARIANT still says something else either trips the
    feature-width assertion or — worse — writes a run_config.json that
    misdescribes the run that produced it.
    """
    g = globals()
    if pattern is not None:
        g["PATTERN_VARIANT"] = str(pattern).strip().lower()
    if kernel is not None:
        g["KERNEL_VARIANT"] = str(kernel).strip().lower()
    pv, kv = g["PATTERN_VARIANT"], g["KERNEL_VARIANT"]
    assert pv in PATTERN_VARIANTS, f"Bad PATTERN_VARIANT: {pv}"
    assert kv in _KV, f"Bad KERNEL_VARIANT: {kv}"
    if folds_dir is not None:
        g["FOLDS_DIR"] = Path(folds_dir)
    fd = g["FOLDS_DIR"]
    tag = _KV[kv]["tag"]
    ms = "_ms" if pv == "multiscale" else ""

    g["KERNEL_SIZE"] = _KV[kv]["k"]
    g["X_path"] = VECTOR_DIR / f"X_train_{tag}{ms}.npy"
    g["X_INFER_PATH"] = VECTOR_DIR / f"X_infer_{tag}{ms}.npy"
    g["MULTISCALE_SPEC_PATH"] = VECTOR_DIR / f"multiscale_spec_{tag}.json"
    g["y_path"] = VECTOR_DIR / f"y_train_{tag}.npy"
    g["CENTERS_PATH"] = VECTOR_DIR / f"centers_train_{tag}.npy"
    # folds mode: row order in these MUST match the X rows above
    g["CV_ROLES_PATH"] = fd / kv / f"roles_{kv}.npy"
    g["CV_FOLDS_CSV"] = fd / kv / f"folds_{kv}.csv"   # block_id per row
    g["USE_AUGMENTATION"], g["USE_ANISO_FEATS"] = PATTERN_VARIANTS[pv]
    # method label in predictions_long: keep the variants distinguishable
    g["METHOD_NAME"] = ("SRF" if (pv == "local" and kv == "k5")
                        else f"SRF_{kv}" if pv == "local"
                        else f"SRF_{pv.upper()}_{kv}")


set_variant()      # establish the derived globals at import time

# ---- Task: "regression" or "classification" ----
TASK = "regression"

# ---- Split mode ----
# "folds"  : spatial block CV (the reported evaluation)
# "random" : 80/20 random split — kept only as the optimistic counter-example
#            the validation section argues against
# "none"   : all rows train, no evaluation split
SPLIT_MODE = "folds"
TEST_FRACTION = 0.20      # random mode only
SPLIT_SEED = 42           # random mode only

PROPS_COV = ["FE", "FEO", "Magsus", "S"]
MASK_NAME = "MASK"
PROPS_TOTAL = PROPS_COV + [MASK_NAME]

# y columns if y is 2D. For classification set e.g. {"LITHO": 0}.
TARGET_COLS = {"DTR": 0, "MAGNETIC": 1}

# ---- Augmentation (paper Fig 1(f)) + anisotropy feature engineering ----
# USE_AUGMENTATION / USE_ANISO_FEATS are set by set_variant() above.
# Augmentation is the 4 z-rotations only: CELL_SIZE is anisotropic (5,5,2), so
# the 24 cube rotations of an isotropic grid are not physically valid here.
CELL_SIZE = (5.0, 5.0, 2.0)
A_MAJOR = 120.0
A_SEMI = 50.0
A_MINOR = 25.0
AZIMUTH_DEG = 110.0
DIP_DEG = 25.0
ANISO_SHARPNESS = 0.5
ANISO_VAR_LOG1P = True
ANISO_VAR_ZSCORE = True
ANISO_STD_EPS = 1e-6

# ---- Hyperparameter selection inside each fold ----
# "subfolds": spatial sub-fold CV — the training BLOCKS of each fold are split
#             into N_SUBFOLDS spatial groups; every config is scored by pooled
#             held-out performance across sub-folds. Selects configs that
#             survive spatial shift (OOB systematically picks interpolation-
#             overfit configs: depth 10 / min_leaf 4 / max_samples 1.0).
# "oob"     : rank by out-of-bag error instead. Retained for comparison; it is
#             the selection rule the spatial sub-fold CV replaced.
TUNE_MODE = "subfolds"
N_SUBFOLDS = 3
SUBFOLD_SEED = 777
# Seed of the final refit (Rank 1 config on ALL of the fold's training rows).
# Fixed and reported so the per-fold models are reproducible.
REFIT_SEED = 999

# ---- Grid search (tuned per split) ----
# Re-centred from the sensitivity analysis (srf_sensitivity.py: a 735-config
# factorial plus a 216-config edge sweep on local_aniso/k3). Two things came
# out of it.
#
# First, a NOISE FLOOR: the same config refit under 12 different seeds moves
# pooled spatial-CV R^2 by 0.0137 (DTR) / 0.0065 (MAGNETIC). Nothing smaller
# than that is a real difference, and several axes of the old grid moved less
# than that across their whole range — they were being searched for nothing.
#
# Second, the ranges were mis-centred rather than too narrow:
#   n_estimators  flat from 30 to 750 trees with no trend at all -> fixed at 150
#   max_depth     anything >=3 is equivalent; 10 costs 3x for nothing, 2 is bad
#   min_leaf      50 collapses (-0.057); the useful band is 4-16
#   max_features  the ONLY factor with a genuine optimum: 6 candidate features
#                 of 167 for DTR, 12 for MAGNETIC. The old floor was "sqrt"=12,
#                 so DTR's optimum sat outside the grid entirely. Given as
#                 explicit counts now — "sqrt" depends on the design width and
#                 silently moves if the feature block changes.
#   max_samples   whole range moves less than the noise floor -> fixed at 0.7
#
# Net: 27 configs instead of 216, same accuracy, 8x cheaper. split_quantiles and
# bootstrap stay fixed — discretisation/resampling details, not model choices.
PARAM_GRID = {
    "n_estimators":    [150],
    "max_depth":       [3, 4, 6],
    "min_leaf":        [4, 8, 16],
    "max_features":    [6, 9, 12],
    "max_samples":     [0.7],
    "split_quantiles": [64],
    "bootstrap":       [True],
}   # 1*3*3*3*1 = 27 configs

# The grid above was trimmed on a REGRESSION ablation and is wrong for a
# 3-class problem with a ~9% class: depth <= 6 gives at most 64 leaves,
# min_leaf up to 16 exceeds a quarter of the high-grade training rows, and
# max_features 6-12 of 135 means a split rarely sees the centre voxel. A
# standard forest given these settings reproduces SRF's failure exactly
# (bal.acc 0.54/0.51, high recall 0.06/0.00), so the settings are the cause.
if TASK == "classification":
    PARAM_GRID = {
        "n_estimators":    [300],
        "max_depth":       [8, 14, 24],   # SpatialDecisionTree needs an int
        "min_leaf":        [1, 2, 4],
        "max_features":    [12, 24, 40],
        "max_samples":     [0.7],
        "split_quantiles": [64],
        "bootstrap":       [True],
        "class_balance":   [None, "balanced"],
    }   # 3*3*3*2 = 54 configs

if QUICK_RUN:
    # Fast diagnostic ablation only. These are NOT paper numbers — unset
    # SRF_QUICK to get the full search above.
    print("[WARN] SRF_QUICK is set: 4-config diagnostic grid, not the full "
          "search. Unset SRF_QUICK before the final run.")
    PARAM_GRID = {
        "n_estimators":    [60],
        "max_depth":       [4, 8],
        "min_leaf":        [10],
        "max_features":    ["sqrt", "third"],
        "max_samples":     [0.7],
        "split_quantiles": [64],
        "bootstrap":       [True],
    }   # 4 configs

# ---- Exceedance probability (regression) ----
# For each threshold thr: P(y > thr) = fraction of trees predicting above thr
# (identity-rotation view; nonparametric, no Gaussian assumption). Emitted as
# p_gt{thr:g} columns in predictions_long + pooled Brier / AUC in summary.csv.
# In classification mode the per-class probabilities (p_<label>) are emitted
# instead. Set [] to disable.
EXCEEDANCE_THRESHOLDS = [55.0, 80.0]

# ---- Permutation importance (single-split modes only; skipped in folds mode) ----
IMPORTANCE_ENABLE = True
IMPORTANCE_MAX_OOB_PER_TREE = 256
IMPORTANCE_MAX_TREES = None
IMPORTANCE_N_REPEATS = 1
IMPORTANCE_SEED = 0
IMPORTANCE_SAVE_PNG = True

# ---- Parallel ----
# Grid configs are fit independently, each with its own derived seed, so results
# do not depend on this value (verified: N_JOBS 16 and -1 give bit-identical
# summary.csv and prediction tables). -1 = all cores.
N_JOBS = int(os.environ.get("SRF_N_JOBS", "-1"))

ANISO_FEATS_PER_PROP = 8


# ============================================================
# Helpers
# ============================================================

def _to_py(obj):
    if obj is None:
        return None
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [_to_py(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_py(v) for k, v in obj.items()}
    return str(obj)


def save_json(d, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_py(d), f, ensure_ascii=False, indent=2)


def load_multiscale_spec(path):
    with open(path, encoding="utf-8") as f:
        spec = json.load(f)
    claimed = spec.get("contract_sha256")
    unsigned = dict(spec)
    unsigned.pop("contract_sha256", None)
    payload = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    actual = hashlib.sha256(payload).hexdigest()
    if claimed != actual:
        raise ValueError(
            f"Multi-scale contract hash mismatch in {path}: "
            f"claimed={claimed}, actual={actual}")
    if int(spec["kernel_size"]) != int(KERNEL_SIZE):
        raise ValueError(
            f"Multi-scale spec kernel k={spec['kernel_size']} but trainer uses "
            f"k={KERNEL_SIZE}.")
    return spec


def regression_metrics(y_true, y_pred):
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(((yt - yp) ** 2).sum())
    ss_tot = float(((yt - yt.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / max(ss_tot, 1e-300)
    rmse = float(np.sqrt(ss_res / yt.size))
    mae = float(np.abs(yt - yp).mean())
    return r2, rmse, mae


def classification_metrics(y_true, y_pred):
    """Returns (balanced_error, balanced_accuracy).

    Deliberately BALANCED rather than the plain error rate: the high-grade
    class is ~9% of samples, so a model that never predicts it still scores
    ~0.91 plain accuracy. Ranking ascends on the returned error, so balanced
    error makes selection optimise balanced accuracy — the same objective the
    point-support classifiers use, which is what makes them comparable."""
    yt = np.asarray(y_true); yp = np.asarray(y_pred)
    recalls = [float((yp[yt == c] == c).mean())
               for c in np.unique(yt) if (yt == c).any()]
    bacc = float(np.mean(recalls)) if recalls else 0.0
    return 1.0 - bacc, bacc


def make_run_dirs(base: Path) -> dict:
    run_name = "SRF_run_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    run = base / run_name
    dirs = {
        "run": run,
        "metadata": run / "00_metadata",
        "grid": run / "01_grid_search",
        "models": run / "02_models",
        "test": run / "03_test_evaluation",
        "importance": run / "04_importance",
        "fullpred": run / "05_full_prediction",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def compute_aniso_pack(X_tr, rot_maps, rot_maps_vox, n_props_total, n_props_aniso):
    """Aniso weights/masks (geometry, split-independent) + variance-feature
    normalisation computed on the given TRAINING rows only (leakage-safe)."""
    w0, masks0 = build_aniso_weights_and_masks(
        k=KERNEL_SIZE, cell_size=CELL_SIZE,
        a_major=A_MAJOR, a_semi=A_SEMI, a_minor=A_MINOR,
        azimuth_deg=AZIMUTH_DEG, dip_deg=DIP_DEG, sharpness=ANISO_SHARPNESS,
    )
    w_flats = rotate_voxel_flats(w0, rot_maps_vox)
    masks_flats = {kk: rotate_voxel_flats(masks0[kk], rot_maps_vox)
                   for kk in ("major", "semi", "minor")}

    var_cols = []
    for p in range(n_props_aniso):
        b = p * ANISO_FEATS_PER_PROP
        var_cols.extend([b + 1, b + 3, b + 5, b + 7])
    var_idx = np.array(var_cols, dtype=np.int64)

    v_blocks = []
    for r in range(len(rot_maps)):
        feats_r = build_aniso_features(
            X_tr[:, rot_maps[r]], w_flats[r],
            masks_flats["major"][r], masks_flats["semi"][r], masks_flats["minor"][r],
            KERNEL_SIZE, n_props_total, n_props_aniso,
        )
        v_blocks.append(feats_r[:, var_idx].astype(np.float64))
    v = np.vstack(v_blocks)
    if ANISO_VAR_LOG1P:
        v = np.log1p(np.maximum(v, 0.0))
    mu = v.mean(axis=0)
    sigma = np.maximum(v.std(axis=0), ANISO_STD_EPS)

    return {"w_flats": w_flats, "masks_flats": masks_flats, "var_idx": var_idx,
            "var_mu": mu.astype(np.float32), "var_sigma": sigma.astype(np.float32)}


# ============================================================
# Grid-search runners — everything passed explicitly (worker-safe).
# ext_tr = precomputed design matrices (one per rotation view) for the fold's
# training rows: config-independent, so computed ONCE per fold (big speedup).
# ============================================================

def make_subfolds(block_ids, n_sub, seed):
    """Assign rows to spatial sub-folds by their fold-builder BLOCK, greedy-
    balanced. Falls back to random rows if there are too few blocks."""
    rng = np.random.default_rng(seed)
    occ, cnt = np.unique(block_ids, return_counts=True)
    if occ.size < n_sub:
        return rng.integers(0, n_sub, size=block_ids.size)
    order = rng.permutation(occ.size)
    tot = np.zeros(n_sub, dtype=np.int64)
    mp = {}
    for i in order:
        s = int(np.argmin(tot))
        mp[int(occ[i])] = s
        tot[s] += cnt[i]
    return np.array([mp[int(b)] for b in block_ids], dtype=np.int8)


def run_single_config(cfg_id, cfg, y_tr, ext_tr, rot_maps, aniso_pack, run_cfg, base_seed=123):
    """OOB scoring (TUNE_MODE='oob')."""
    start = time.perf_counter()
    try:
        model = build_forest_from_config(cfg, run_cfg, rot_maps, aniso_pack, base_seed + cfg_id)
        model.fit(None, y_tr, ext_rots=ext_tr)
        scores = model.oob_scores(y_tr)
        row = {"config_id": cfg_id, **cfg, **scores,
               "fit_time_sec": float(time.perf_counter() - start)}
    except Exception as e:
        row = {"config_id": cfg_id, **cfg,
               "fit_time_sec": float(time.perf_counter() - start),
               "error": str(e)[:300]}
    return row


def run_single_config_subcv(cfg_id, cfg, y_tr, ext_tr, sub_ids,
                            rot_maps, aniso_pack, run_cfg, base_seed=123):
    """Spatial sub-fold CV scoring: fit on all-but-one sub-fold, predict the
    held-out sub-fold, pool over sub-folds -> SUBCV_* metrics."""
    start = time.perf_counter()
    is_clf = (run_cfg["task"] == "classification")
    n = len(y_tr)
    try:
        pred = np.empty(n, dtype=object) if is_clf else np.full(n, np.nan)
        seen = np.zeros(n, dtype=bool)
        for s in range(int(sub_ids.max()) + 1):
            te = sub_ids == s
            tr = ~te
            if te.sum() == 0 or tr.sum() < 4 * int(cfg["min_leaf"]):
                continue
            model = build_forest_from_config(
                cfg, run_cfg, rot_maps, aniso_pack, base_seed + cfg_id * 100 + s)
            model.fit(None, y_tr[tr], ext_rots=[E[tr] for E in ext_tr])
            pred[te] = model.predict(None, ext0=ext_tr[0][te])
            seen[te] = True

        if seen.sum() < 4:
            raise RuntimeError("too few sub-fold predictions")

        if is_clf:
            err = float(np.mean(np.asarray(pred[seen]) != np.asarray(y_tr)[seen]))
            scores = {"SUBCV_ERR": err, "SUBCV_ACC": 1.0 - err,
                      "SUBCV_n": int(seen.sum())}
        else:
            yt = np.asarray(y_tr, dtype=np.float64)[seen]
            yp = pred[seen].astype(np.float64)
            ss_res = float(((yt - yp) ** 2).sum())
            ss_tot = float(((yt - yt.mean()) ** 2).sum())
            scores = {"SUBCV_R2": 1.0 - ss_res / max(ss_tot, 1e-300),
                      "SUBCV_RMSE": float(np.sqrt(ss_res / yt.size)),
                      "SUBCV_MAE": float(np.abs(yt - yp).mean()),
                      "SUBCV_n": int(seen.sum())}
        row = {"config_id": cfg_id, **cfg, **scores,
               "fit_time_sec": float(time.perf_counter() - start)}
    except Exception as e:
        row = {"config_id": cfg_id, **cfg,
               "fit_time_sec": float(time.perf_counter() - start),
               "error": str(e)[:300]}
    return row


def grid_search_and_refit(y_tr, ext_tr, sub_ids, rot_maps, aniso_pack, run_cfg,
                          configs, tag, grid_dir):
    """Grid search on the fold's training rows (spatial sub-fold CV when
    sub_ids given, else OOB), then refit Rank 1 on ALL training rows.
    Returns (df_results, best_cfg, best_model, rank_key)."""
    use_subcv = (TUNE_MODE == "subfolds") and (sub_ids is not None)
    if use_subcv:
        rank_key = "SUBCV_R2" if TASK == "regression" else "SUBCV_ERR"
        rank_ascending = (TASK == "classification")
        runner = delayed(run_single_config_subcv)
        args = lambda i, c: (i, c, y_tr, ext_tr, sub_ids, rot_maps, aniso_pack, run_cfg)
    else:
        rank_key = "OOB_R2" if TASK == "regression" else "OOB_ERR"
        rank_ascending = (TASK == "classification")
        runner = delayed(run_single_config)
        args = lambda i, c: (i, c, y_tr, ext_tr, rot_maps, aniso_pack, run_cfg)

    t0 = time.perf_counter()
    results_list = Parallel(n_jobs=N_JOBS, verbose=5)(
        runner(*args(cfg_id, cfg)) for cfg_id, cfg in enumerate(configs))
    grid_time = time.perf_counter() - t0

    df_results = pd.DataFrame(results_list)
    if rank_key not in df_results.columns:
        raise RuntimeError(f"All configs failed for {tag}. First error: "
                           f"{df_results.get('error', pd.Series(['?'])).iloc[0]}")
    df_results = (df_results
                  .sort_values(rank_key, ascending=rank_ascending, na_position="last")
                  .reset_index(drop=True))
    df_results.insert(0, "RankID", np.arange(1, len(df_results) + 1))
    df_results.to_csv(grid_dir / f"grid_results_{tag}.csv", index=False)

    best_row = df_results.iloc[0]
    best_cfg = {k: best_row[k] for k in PARAM_GRID.keys()}

    best_model = build_forest_from_config(best_cfg, run_cfg, rot_maps,
                                          aniso_pack, REFIT_SEED)
    best_model.fit(None, y_tr, ext_rots=ext_tr)

    print(f"[{tag}] grid {grid_time:.1f}s ({'subCV' if use_subcv else 'OOB'}) "
          f"| best {rank_key}={best_row[rank_key]:.4f} "
          f"| cfg={ {k: _to_py(v) for k, v in best_cfg.items()} }")
    return df_results, best_cfg, best_model, rank_key


def predict_with_uncertainty(model, X_te, ext0=None):
    """Predictions + per-sample uncertainty on the identity-rotation view.
    Regression: mean/std across trees + exceedance probs P(y > thr).
    Classification: label + entropy + per-class probabilities.
    Returns (y_pred, y_unc, extras) where extras maps column name -> array."""
    Xe = ext0 if ext0 is not None else model.build_design_matrix(X_te)
    if model.task == "classification":
        proba = model._proba_from_ext(Xe)
        labels = model.classes_[np.argmax(proba, axis=1)]
        extras = {f"p_{c}": proba[:, k].astype(np.float32)
                  for k, c in enumerate(model.classes_)}
        return labels, normalized_entropy(proba), extras
    tree_preds = np.stack([t.predict(Xe) for t in model.trees_])
    extras = {f"p_gt{thr:g}": (tree_preds > thr).mean(axis=0).astype(np.float32)
              for thr in EXCEEDANCE_THRESHOLDS}
    return (tree_preds.mean(axis=0).astype(np.float32),
            tree_preds.std(axis=0).astype(np.float32), extras)


# ============================================================
# Importance / zone-of-influence export (single-split modes)
# ============================================================

def export_importance(model, X_tr, y_tr, target_name, out_dir: Path, feature_names, run_cfg):
    print("\nComputing OOB permutation importance (paper Sect 2.3)...")
    t0 = time.perf_counter()
    imp = model.permutation_importance(
        X_tr, y_tr,
        max_oob_per_tree=IMPORTANCE_MAX_OOB_PER_TREE,
        max_trees=IMPORTANCE_MAX_TREES,
        n_repeats=IMPORTANCE_N_REPEATS,
        random_state=IMPORTANCE_SEED,
    )
    print(f"Importance done in {time.perf_counter() - t0:.1f} s "
          f"({imp['n_trees_used']} trees, baseline error {imp['baseline_error']:.6g})")

    k = run_cfg["kernel_size"]
    npt = run_cfg["n_props_total"]
    mean = imp["importance_mean"]

    df_imp = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": mean,
        "importance_std": imp["importance_std"],
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    imp_path = out_dir / f"importance_full_{target_name}.csv"
    df_imp.to_csv(imp_path, index=False)

    zones = reshape_base_importance_to_zones(mean, k, npt)
    half = k // 2
    long_rows = []
    zone_paths = []
    for p, prop in enumerate(PROPS_TOTAL):
        zpath = out_dir / f"zone_{prop}_{target_name}.npy"
        np.save(zpath, zones[p])
        zone_paths.append(str(zpath))
        for i in range(k):
            for j in range(k):
                for kk in range(k):
                    long_rows.append({"prop": prop,
                                      "off_x": i - half, "off_y": j - half, "off_z": kk - half,
                                      "importance": float(zones[p][i, j, kk])})
    zone_long_path = out_dir / f"zone_long_{target_name}.csv"
    pd.DataFrame(long_rows).to_csv(zone_long_path, index=False)

    k3 = k ** 3
    per_var_rows = [{"prop": prop, "block": "voxels",
                     "importance_total": float(zones[p].sum())}
                    for p, prop in enumerate(PROPS_TOTAL)]
    n_static = int(run_cfg.get("n_static_features", 0))
    if n_static:
        static_start = k3 * npt
        static_names = feature_names[static_start:static_start + n_static]
        static_imp = mean[static_start:static_start + n_static]
        for axis in ("major", "semi", "minor"):
            keep = np.array(
                [name.startswith(f"ms_{axis}_") for name in static_names],
                dtype=bool)
            per_var_rows.append({
                "prop": axis,
                "block": "multiscale",
                "importance_total": float(static_imp[keep].sum()),
            })
    if run_cfg["use_aniso_feats"]:
        aniso_start = k3 * npt + n_static
        for p, prop in enumerate(PROPS_COV):
            seg = mean[aniso_start + p * ANISO_FEATS_PER_PROP:
                       aniso_start + (p + 1) * ANISO_FEATS_PER_PROP]
            per_var_rows.append({"prop": prop, "block": "aniso_feats",
                                 "importance_total": float(seg.sum())})
    per_var_path = out_dir / f"importance_by_variable_{target_name}.csv"
    pd.DataFrame(per_var_rows).to_csv(per_var_path, index=False)

    png_path = None
    if IMPORTANCE_SAVE_PNG:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(npt, k, figsize=(2.1 * k, 2.1 * npt), squeeze=False)
            vmax = max(z.max() for z in zones) or 1.0
            for p in range(npt):
                for zi in range(k):
                    ax = axes[p][zi]
                    ax.imshow(zones[p][:, :, zi].T, origin="lower",
                              vmin=0.0, vmax=vmax, cmap="viridis")
                    ax.set_xticks([]); ax.set_yticks([])
                    if p == 0:
                        ax.set_title(f"z={zi - half:+d}", fontsize=9)
                    if zi == 0:
                        ax.set_ylabel(PROPS_TOTAL[p], fontsize=9)
            fig.suptitle(f"Zone of influence — {target_name}", fontsize=11)
            fig.tight_layout()
            png_path = out_dir / f"zone_of_influence_{target_name}.png"
            fig.savefig(png_path, dpi=150)
            plt.close(fig)
        except Exception as e:
            print(f"[WARN] PNG export skipped: {e}")

    print("\nTop 15 predictors by permutation importance:")
    print(df_imp.head(15).to_string(index=False))
    return {"importance_csv": str(imp_path), "zone_long_csv": str(zone_long_path),
            "per_variable_csv": str(per_var_path), "zone_npy": zone_paths,
            "zone_png": str(png_path) if png_path else None,
            "baseline_error": float(imp["baseline_error"]),
            "n_trees_used": int(imp["n_trees_used"])}


# ============================================================
# MAIN TRAINING
# ============================================================

def run_training():
    assert TASK in ("regression", "classification"), f"Bad TASK: {TASK}"
    assert SPLIT_MODE in ("folds", "random", "none"), f"Bad SPLIT_MODE: {SPLIT_MODE}"
    assert PATTERN_VARIANT in PATTERN_VARIANTS, (
        f"Bad PATTERN_VARIANT: {PATTERN_VARIANT}")
    assert X_path.exists() and y_path.exists(), "X or y file not found."

    n_props_total = len(PROPS_TOTAL)
    n_props_aniso = len(PROPS_COV)
    base_d = (KERNEL_SIZE ** 3) * n_props_total
    multiscale_spec = None
    n_static_features = 0
    if PATTERN_VARIANT == "multiscale":
        if not MULTISCALE_SPEC_PATH.exists():
            raise FileNotFoundError(
                f"Missing {MULTISCALE_SPEC_PATH}. Run "
                "build_multiscale_vectors.py first.")
        multiscale_spec = load_multiscale_spec(MULTISCALE_SPEC_PATH)
        n_static_features = int(multiscale_spec["n_static_features"])
    expected_d = base_d + n_static_features

    dirs = make_run_dirs(RUNS_BASE)
    print("=" * 70)
    print(f"SRF TRAINING RUN: {dirs['run']}")
    print(f"SPLIT_MODE: {SPLIT_MODE} | TASK: {TASK}")
    print("=" * 70)

    X = np.load(X_path).astype(np.float32)
    y_all = np.load(y_path)
    N, D = X.shape
    assert D == expected_d, (
        f"Expected local {base_d} + static {n_static_features} = "
        f"{expected_d} features; got {D}")
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN/Inf; rebuild the feature arrays.")

    if CENTERS_PATH.exists():
        centers = np.load(CENTERS_PATH).astype(np.float64)
        assert centers.shape[0] == N, (
            f"centers rows ({centers.shape[0]}) != X rows ({N}) — "
            "regenerate with build_voxels.py")
    else:
        print(f"[WARN] centers file not found ({CENTERS_PATH}); x,y,z = NaN in outputs.")
        centers = np.full((N, 3), np.nan)

    # ---- splits ----
    roles = None
    if SPLIT_MODE == "folds":
        assert CV_ROLES_PATH.exists(), (
            f"Roles file not found: {CV_ROLES_PATH}. Run make_folds.py first.")
        roles = np.load(CV_ROLES_PATH)
        assert roles.shape[0] == N, (
            f"Roles rows ({roles.shape[0]}) != X rows ({N}). The folds were built "
            "for a different extraction — re-run make_folds.py.")
        n_folds = roles.shape[1]
        splits = [(f, roles[:, f] == 0, roles[:, f] == 1) for f in range(n_folds)]
        for f, tr, te in splits:
            n_buf = int((roles[:, f] == 2).sum())
            print(f"fold {f + 1}: train={int(tr.sum())} test={int(te.sum())} buffer={n_buf}")
    elif SPLIT_MODE == "random":
        rng_split = np.random.default_rng(SPLIT_SEED)
        perm = rng_split.permutation(N)
        n_test = int(np.floor(TEST_FRACTION * N))
        te_mask = np.zeros(N, dtype=bool); te_mask[perm[:n_test]] = True
        splits = [(0, ~te_mask, te_mask)]
        print(f"Random split | seed={SPLIT_SEED} | train={int((~te_mask).sum())} "
              f"| test={n_test}")
        print("[NOTE] Random split of overlapping kernels is optimistic - "
              "use SPLIT_MODE='folds' for honest comparison numbers.")
    else:
        splits = [(0, np.ones(N, dtype=bool), np.zeros(N, dtype=bool))]
        print("No split: all data as TRAIN.")
    multi_fold = len(splits) > 1

    # ---- targets ----
    if y_all.ndim == 1:
        target_map = {"Y": None}
    elif y_all.ndim == 2:
        target_map = dict(TARGET_COLS)
    else:
        raise ValueError(f"Unsupported y shape: {y_all.shape}")

    def y_col(col):
        yt = y_all if y_all.ndim == 1 else y_all[:, col]
        return yt.astype(np.float32) if TASK == "regression" else yt

    # ---- rotations ----
    rotations = (generate_z_rotations_4() if USE_AUGMENTATION
                 else [((0, 1, 2), (1, 1, 1))])

    rot_maps = build_rotation_index_maps(KERNEL_SIZE, n_props_total, rotations)
    rot_maps_vox = voxel_maps_from_feature_maps(rot_maps, n_props_total)

    print(f"pattern: {PATTERN_VARIANT} | static features: {n_static_features}")
    print(f"augmentation: {USE_AUGMENTATION} ({len(rotations)} rotations) "
          f"| anisotropy features: {USE_ANISO_FEATS}\n")

    # ---- run_cfg (what a model needs to be rebuilt) ----
    run_cfg = {
        "task": TASK,
        "quick_run": bool(QUICK_RUN),
        "pattern_variant": PATTERN_VARIANT,
        "kernel_size": int(KERNEL_SIZE),
        "n_props_total": int(n_props_total),
        "n_props_aniso": int(n_props_aniso),
        "n_static_features": int(n_static_features),
        "props_cov": list(PROPS_COV),
        "props_total": list(PROPS_TOTAL),
        "mask_name": MASK_NAME,
        "use_augmentation": bool(USE_AUGMENTATION),
        "rotations_count": int(len(rotations)),
        "use_aniso_feats": bool(USE_ANISO_FEATS),
        "aniso_var_log1p": bool(ANISO_VAR_LOG1P),
        "aniso_var_zscore": bool(ANISO_VAR_ZSCORE),
        "cell_size": tuple(float(x) for x in CELL_SIZE),
        "a_major": float(A_MAJOR), "a_semi": float(A_SEMI), "a_minor": float(A_MINOR),
        "azimuth_deg": float(AZIMUTH_DEG), "dip_deg": float(DIP_DEG),
        "aniso_sharpness": float(ANISO_SHARPNESS),
        "X_path": str(X_path), "y_path": str(y_path),
        "x_infer_path": str(X_INFER_PATH),
        "multiscale_spec_path": (
            str(MULTISCALE_SPEC_PATH) if multiscale_spec is not None else None),
        "multiscale_contract_sha256": (
            multiscale_spec["contract_sha256"]
            if multiscale_spec is not None else None),
        "centers_path": str(CENTERS_PATH),
        "target_cols": {k: (v if v is not None else 0) for k, v in target_map.items()},
        "expected_d": int(expected_d),
        "method_name": METHOD_NAME,
        "split": {
            "mode": SPLIT_MODE,
            "n_folds": len(splits),
            "cv_roles_path": str(CV_ROLES_PATH) if SPLIT_MODE == "folds" else None,
            "test_fraction": float(TEST_FRACTION) if SPLIT_MODE == "random" else None,
            "split_seed": int(SPLIT_SEED) if SPLIT_MODE == "random" else None,
            "n_total": int(N),
            "fold_sizes": [{"fold": f + 1, "train": int(tr.sum()), "test": int(te.sum())}
                           for f, tr, te in splits],
        },
        "param_grid": {k: list(v) for k, v in PARAM_GRID.items()},
        "tune_mode": TUNE_MODE,
        "n_subfolds": int(N_SUBFOLDS) if TUNE_MODE == "subfolds" else None,
        "subfold_seed": int(SUBFOLD_SEED) if TUNE_MODE == "subfolds" else None,
        "refit_seed": int(REFIT_SEED),
        "created_at": datetime.now().isoformat(),
    }
    save_json(run_cfg, dirs["metadata"] / "run_config.json")

    # prepared.pkl: FULL-data aniso pack so srf_predict.py deploys exactly
    full_pack = (compute_aniso_pack(X, rot_maps, rot_maps_vox,
                                    n_props_total, n_props_aniso)
                 if USE_ANISO_FEATS else None)
    joblib.dump({"rot_maps": rot_maps, "aniso_pack": full_pack,
                 "train_idx": np.arange(N), "roles": roles},
                dirs["metadata"] / "prepared.pkl")
    print(f"Saved config  : {dirs['metadata'] / 'run_config.json'}")
    print(f"Saved prepared: {dirs['metadata'] / 'prepared.pkl'}\n")

    feature_names = build_feature_names(
        KERNEL_SIZE, PROPS_TOTAL, PROPS_COV, False)
    if multiscale_spec is not None:
        feature_names.extend(multiscale_spec["feature_names"])
    if USE_ANISO_FEATS:
        all_names = build_feature_names(
            KERNEL_SIZE, PROPS_TOTAL, PROPS_COV, True)
        feature_names.extend(all_names[base_d:])
    expected_design_d = expected_d + (
        n_props_aniso * ANISO_FEATS_PER_PROP if USE_ANISO_FEATS else 0)
    if len(feature_names) != expected_design_d:
        raise RuntimeError(
            f"Feature-name contract has {len(feature_names)} names; "
            f"expected {expected_design_d}.")
    keys = list(PARAM_GRID.keys())
    configs = [dict(zip(keys, vals)) for vals in product(*PARAM_GRID.values())]
    print(f"Grid configurations per split: {len(configs)} | tuning: {TUNE_MODE}"
          + (f" ({N_SUBFOLDS} spatial sub-folds)" if TUNE_MODE == "subfolds" else ""))

    # block ids for spatial sub-fold tuning (from the fold builder's CSV)
    block_ids_all = None
    if TUNE_MODE == "subfolds":
        if SPLIT_MODE == "folds" and CV_FOLDS_CSV.exists():
            block_ids_all = pd.read_csv(CV_FOLDS_CSV)["block_id"].values
            assert block_ids_all.shape[0] == N, (
                f"folds CSV rows ({block_ids_all.shape[0]}) != X rows ({N})")
        else:
            print("[WARN] TUNE_MODE='subfolds' needs SPLIT_MODE='folds' + folds CSV "
                  "-> falling back to OOB tuning.")

    long_rows = []                       # rows of predictions_long_*.csv
    fold_metric_rows = []                # per fold per target
    best_meta = {t: [] for t in target_map}

    # ============ split loop ============
    for fold_id, tr_mask, te_mask in splits:
        fold_no = fold_id + 1
        print("=" * 70)
        print(f"SPLIT {fold_no}/{len(splits)} | train={int(tr_mask.sum())} "
              f"test={int(te_mask.sum())}")
        print("=" * 70)

        X_tr = X[tr_mask]
        X_te = X[te_mask]
        te_rows = np.where(te_mask)[0]

        # leakage-safe: aniso normalisation from THIS split's training rows only
        aniso_pack = (compute_aniso_pack(X_tr, rot_maps, rot_maps_vox,
                                         n_props_total, n_props_aniso)
                      if USE_ANISO_FEATS else None)

        # Precompute ALL design matrices ONCE per fold (config-independent):
        # every grid config / sub-fold reuses these slices instead of
        # re-transforming X — the single biggest speed win.
        t_ext = time.perf_counter()
        proto = build_forest_from_config(configs[0], run_cfg, rot_maps, aniso_pack, 0)
        ext_full = proto.build_design_matrices(X)
        ext_tr = [E[tr_mask] for E in ext_full]
        ext_te0 = ext_full[0][te_mask]
        print(f"design matrices: {len(ext_full)} rotation view(s), "
              f"{ext_full[0].shape[1]} features, {time.perf_counter() - t_ext:.1f} s")

        sub_ids = None
        if block_ids_all is not None:
            sub_ids = make_subfolds(block_ids_all[tr_mask], N_SUBFOLDS,
                                    SUBFOLD_SEED + 1000 * fold_no)
            sizes = [int((sub_ids == s).sum()) for s in range(N_SUBFOLDS)]
            print(f"spatial sub-folds (tuning): sizes={sizes}")

        for target_name, col in target_map.items():
            y_t = y_col(col)
            y_tr, y_te = y_t[tr_mask], y_t[te_mask]

            tag = f"{target_name}_fold{fold_no}" if multi_fold else target_name
            df_results, best_cfg, best_model, rank_key = grid_search_and_refit(
                y_tr, ext_tr, sub_ids, rot_maps, aniso_pack, run_cfg, configs,
                tag, dirs["grid"])
            oob_best = best_model.oob_scores(y_tr)
            best_row0 = df_results.iloc[0]
            sel_score = {f"BEST_{rank_key}": _to_py(best_row0[rank_key])}

            row = {"TARGET": target_name, "fold": fold_no,
                   "n_train": int(tr_mask.sum()), "n_test": int(te_mask.sum()),
                   **{k: _to_py(v) for k, v in best_cfg.items()},
                   **sel_score,
                   **{k: _to_py(v) for k, v in oob_best.items()}}

            if X_te.shape[0] > 0:
                y_pred, y_unc, extras = predict_with_uncertainty(best_model, X_te, ext0=ext_te0)
                if TASK == "regression":
                    r2, rmse, mae = regression_metrics(y_te, y_pred)
                    row.update({"TEST_R2": r2, "TEST_RMSE": rmse, "TEST_MAE": mae})
                    print(f"[{tag}] TEST | R2={r2:.4f} RMSE={rmse:.4f} MAE={mae:.4f}")
                else:
                    err, acc = classification_metrics(y_te, y_pred)
                    row.update({"TEST_ERR": err, "TEST_ACC": acc,
                                "TEST_entropy_mean": float(np.mean(y_unc))})
                    print(f"[{tag}] TEST | err={err:.4f} acc={acc:.4f}")

                for i, r_id in enumerate(te_rows):
                    long_rows.append({
                        "row": int(r_id),
                        "x": centers[r_id, 0], "y": centers[r_id, 1], "z": centers[r_id, 2],
                        "fold": fold_no, "target": target_name, "method": METHOD_NAME,
                        "y_true": y_te[i], "y_pred": y_pred[i], "y_unc": float(y_unc[i]),
                        **{k: float(v[i]) for k, v in extras.items()},
                    })

            fold_metric_rows.append(row)
            best_meta[target_name].append({"fold": fold_no,
                                           "best_hparams": {k: _to_py(v) for k, v in best_cfg.items()},
                                           "metrics": {k: _to_py(v) for k, v in row.items()
                                                       if k.startswith(("OOB", "TEST"))}})

            model_path = dirs["models"] / f"best_{tag}.joblib"
            best_model.meta_ = {"version": "SRF_V4_folds", "run_dir": str(dirs["run"]),
                                "target": target_name, "fold": fold_no,
                                "trained_on": f"split_{SPLIT_MODE}_fold{fold_no}_train",
                                "task": TASK, "run_cfg": run_cfg,
                                "best_hparams": {k: _to_py(v) for k, v in best_cfg.items()}}
            joblib.dump(best_model, model_path)

            # importance only in single-split modes (per-fold would multiply cost)
            if IMPORTANCE_ENABLE and not multi_fold:
                export_importance(best_model, X_tr, y_tr, target_name,
                                  dirs["importance"], feature_names, run_cfg)
            elif IMPORTANCE_ENABLE and multi_fold and fold_id == 0 and target_name == list(target_map)[0]:
                print("[NOTE] importance skipped in folds mode "
                      "(run SPLIT_MODE='none' for zone-of-influence on all data).")

    # ============ outputs ============
    df_long = pd.DataFrame(long_rows)
    if len(df_long) > 0:
        long_path = dirs["test"] / f"predictions_long_{METHOD_NAME}.csv"
        df_long.to_csv(long_path, index=False)
        print(f"\nSaved long predictions table: {long_path}")

    df_fold = pd.DataFrame(fold_metric_rows)
    summary_rows = []
    for target_name in target_map:
        sub = df_fold[df_fold["TARGET"] == target_name]
        sub.to_csv(dirs["test"] / f"fold_metrics_{target_name}.csv", index=False)
        agg = {"TARGET": target_name, "TASK": TASK, "SPLIT_MODE": SPLIT_MODE,
               "n_folds": len(splits)}
        # OOB_* = training-side (out-of-bag) error of the refit best model;
        # TEST_* = honest spatial-CV error. Both aggregated so the gap between
        # them is readable straight off summary.csv.
        metric_cols = [c for c in sub.columns
                       if c.startswith(("TEST_", "OOB_")) and c != "OOB_coverage"]
        for c in metric_cols:
            agg[f"{c}_mean"] = float(sub[c].mean())
            agg[f"{c}_std"] = float(sub[c].std(ddof=0))

        # POOLED metrics (headline numbers): every point predicted exactly
        # once across folds; sidesteps the per-fold variance artifact in R2.
        if TASK == "regression" and len(df_long) > 0:
            sl = df_long[df_long["target"] == target_name].dropna(subset=["y_true"])
            if len(sl) > 2:
                yt = sl["y_true"].values.astype(np.float64)
                yp = sl["y_pred"].values.astype(np.float64)
                ss_res = float(((yt - yp) ** 2).sum())
                ss_tot = float(((yt - yt.mean()) ** 2).sum())
                cov = float(((yt - yt.mean()) * (yp - yp.mean())).mean())
                agg["POOLED_R2"] = 1.0 - ss_res / max(ss_tot, 1e-300)
                agg["POOLED_RMSE"] = float(np.sqrt(ss_res / yt.size))
                agg["POOLED_MAE"] = float(np.abs(yt - yp).mean())
                agg["POOLED_CCC"] = float(
                    2.0 * cov / (yt.var() + yp.var() + (yt.mean() - yp.mean()) ** 2))
                # exceedance skill: pooled Brier (lower = better) and rank AUC
                for thr in EXCEEDANCE_THRESHOLDS:
                    pc = f"p_gt{thr:g}"
                    if pc not in sl.columns:
                        continue
                    e = (yt > thr).astype(np.float64)
                    p = sl[pc].values.astype(np.float64)
                    agg[f"BRIER_gt{thr:g}"] = float(np.mean((p - e) ** 2))
                    n1, n0 = e.sum(), e.size - e.sum()
                    if n1 > 0 and n0 > 0:   # Mann-Whitney AUC
                        r = pd.Series(p).rank(method="average").values
                        agg[f"AUC_gt{thr:g}"] = float(
                            (r[e == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
        summary_rows.append(agg)
        save_json({"target": target_name, "folds": best_meta[target_name]},
                  dirs["models"] / f"meta_{target_name}.json")

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(dirs["run"] / "summary.csv", index=False)
    df_fold.to_csv(dirs["test"] / "test_metrics.csv", index=False)

    print("\n" + "=" * 70)
    print("CV SUMMARY (mean +/- std across folds):" if multi_fold else "SUMMARY:")
    print(df_summary.to_string(index=False))
    print("=" * 70)
    print(f"RUN COMPLETE: {dirs['run']}")
    if multi_fold:
        print("predictions_long_*.csv is the input to the method-comparison "
              "harness.")
    print("Next: srf_predict.py to select a model and predict the block model.")
    print("=" * 70)
    return dirs["run"]


if __name__ == "__main__":
    run_training()
