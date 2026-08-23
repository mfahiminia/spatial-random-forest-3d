# -*- coding: utf-8 -*-
"""
SRF model selection + full block-model prediction — step 4 of the pipeline.

Cross-validation (srf_train.py) answers how well the method generalises.
This script produces the deliverable map: it takes a training run, ranks the
grid configurations by their mean sub-fold CV score across folds, refits the
selected configuration on ALL labelled rows (deployment style — no data held
back, since generalisation has already been measured) and applies it to every
node of the block model in batches.

Workflow:
  1) Pick a training run (blank = most recent run in SRF_runs).
  2) Pick target(s).
  3) See the ranked config table, pick a RankID (blank = 1 = best).
  4) Refit + predict the block model (Vector\\X_infer_{k^3}.npy).

Outputs (inside the run folder):
  02_models\\deploy_{TARGET}_Rank{K}.joblib
  05_full_prediction\\PRED_{TARGET}_Rank{K}.npy          prediction vector
  05_full_prediction\\PRED_{TARGET}_Rank{K}.csv          row + prediction (+entropy)
  05_full_prediction\\prediction_meta_{TARGET}_Rank{K}.json

Requires the inference array Vector\\X_infer_{k^3}.npy from build_voxels.py
(not shipped with the repository — it is large and fully regenerable).

Run:  python srf_predict.py     (or %run in Jupyter)
Non-interactive use:
  from srf_predict import run_prediction
  run_prediction(run_dir=r"...\\SRF_run_20260725_161248",
                 targets=["DTR"], rank_ids={"DTR": 3}, interactive=False)
"""

import json
import hashlib
import os
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    _HERE = Path(__file__).resolve().parent
except NameError:  # notebook cell
    _HERE = Path.cwd()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import numpy as np
import pandas as pd
import joblib

# Reload guard: a Jupyter kernel may hold an older srf_core_final in memory.
# Version-based check — attribute checks go stale, the version number cannot.
import importlib
import srf_core_final as _core
_NEED_CORE_API = 8
if getattr(_core, "CORE_API_VERSION", 0) < _NEED_CORE_API:
    _core = importlib.reload(_core)
if getattr(_core, "CORE_API_VERSION", 0) < _NEED_CORE_API:
    raise RuntimeError(
        f"srf_core_final is outdated (need API v{_NEED_CORE_API}) — restart the "
        f"kernel or check sys.path ({getattr(_core, '__file__', '?')}).")

from srf_core_final import build_forest_from_config


# ============================================================
# USER CONFIG
# ============================================================

# Defaults to the repository folder; set SRF_PROJECT_ROOT to run elsewhere.
PROJECT_ROOT = Path(os.environ.get("SRF_PROJECT_ROOT", _HERE))

RUNS_BASE = PROJECT_ROOT / "SRF_runs"

# Fallback block model to predict (same feature contract as X_train). Normally
# unused: the path recorded in the run's run_config.json takes precedence, so
# deployment cannot silently use a different feature build than training did.
X_INFER_PATH = PROJECT_ROOT / "Vector" / "X_infer_27.npy"

# "all"  -> refit selected config on ALL rows (train+test): final deployment
# "train"-> refit on the training rows only (honest, but discards data)
REFIT_ON = "all"

# Optionally deploy with more trees than the grid config (None = keep config's).
N_ESTIMATORS_OVERRIDE = None

REFIT_SEED = 2026
BATCH_SIZE = 50_000          # inference batch (rows) to keep memory bounded
USE_TTA = False              # average predictions over the 4 rotations
TOP_K_SHOW = 20

# Exceedance probability maps (regression targets only):
# P(y > thr) = fraction of trees predicting above thr (identity-rotation view).
# Adds a P_GT{thr:g} column per threshold to the prediction CSV. [] disables.
EXCEEDANCE_THRESHOLDS = [55.0, 80.0]


# ============================================================
# Helpers
# ============================================================

def _to_py(obj):
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
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


def _latest_run(base: Path):
    runs = sorted([p for p in base.glob("SRF_run_*") if p.is_dir()])
    return runs[-1] if runs else None


def _parse_max_features(v):
    """Restore max_features from the grid CSV with its ORIGINAL type.

    The type is the meaning here, so a round-trip through the CSV must not
    change it: "sqrt"/"third"/"all" are rules, a float in (0, 1] is a FRACTION
    of the design width, and an int is an absolute COUNT of candidate features.
    Returning 6 as 6.0 makes the forest read it as a fraction and reject it, so
    numbers above 1 are returned as int. Numbers in (0, 1] stay float, which
    also resolves the one ambiguous value: 1 is read as the fraction 1.0
    (= every feature), never as a count of one candidate feature.
    """
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v).strip()
    if s.lower() in ("none", "nan", ""):
        return None
    try:
        f = float(s)
    except ValueError:
        return s                       # "sqrt" / "third" / "all"
    return int(f) if f > 1.0 else f


def _verify_multiscale_contract(run_cfg):
    spec_value = run_cfg.get("multiscale_spec_path")
    expected = run_cfg.get("multiscale_contract_sha256")
    if not spec_value:
        return
    path = Path(spec_value)
    if not path.exists():
        raise FileNotFoundError(f"Multi-scale contract not found: {path}")
    with open(path, encoding="utf-8") as f:
        spec = json.load(f)
    claimed = spec.get("contract_sha256")
    unsigned = dict(spec)
    unsigned.pop("contract_sha256", None)
    payload = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    actual = hashlib.sha256(payload).hexdigest()
    if claimed != actual or expected != actual:
        raise ValueError(
            "Multi-scale feature contract changed after training: "
            f"run={expected}, spec={claimed}, actual={actual}")


def _cfg_from_row(row):
    return {
        "n_estimators": int(float(row["n_estimators"])),
        "max_depth": int(float(row["max_depth"])),
        "min_leaf": int(float(row["min_leaf"])),
        "max_features": _parse_max_features(row["max_features"]),
        "max_samples": float(row["max_samples"]),
        "split_quantiles": int(float(row["split_quantiles"])),
        "bootstrap": str(row["bootstrap"]).strip().lower() in ("true", "1"),
    }


_HP_COLS = ["n_estimators", "max_depth", "min_leaf", "max_features",
            "max_samples", "split_quantiles", "bootstrap"]


def _load_ranked_grid(run_dir: Path, target: str, task: str):
    """Ranked config table for deployment selection.

    Single-split runs: grid_results_{target}.csv as-is.
    Folds-mode runs: per-fold tables grid_results_{target}_fold*.csv are
    aggregated — each config is scored by its MEAN selection metric across
    folds (SUBCV_R2 / SUBCV_ERR, falling back to OOB_*). Ranking by a mean over
    folds rather than by any single fold keeps deployment selection independent
    of which fold happened to be easiest. Configs that fail in a fold simply
    have fewer entries (the n_folds column shows coverage)."""
    gdir = run_dir / "01_grid_search"
    single = gdir / f"grid_results_{target}.csv"
    if single.exists():
        return pd.read_csv(single), "single-split"

    fold_files = sorted(gdir.glob(f"grid_results_{target}_fold*.csv"))
    if not fold_files:
        raise FileNotFoundError(
            f"No grid tables for {target} in {gdir} (neither "
            f"grid_results_{target}.csv nor grid_results_{target}_fold*.csv)")
    df = pd.concat([pd.read_csv(p) for p in fold_files], ignore_index=True)

    if task == "regression":
        metric, asc = ("SUBCV_R2", False) if "SUBCV_R2" in df.columns else ("OOB_R2", False)
    else:
        metric, asc = ("SUBCV_ERR", True) if "SUBCV_ERR" in df.columns else ("OOB_ERR", True)
    df = df.dropna(subset=[metric])

    hp = [c for c in _HP_COLS if c in df.columns]
    ranked = (df.groupby(hp, dropna=False)[metric]
              .agg(["mean", "std", "count"]).reset_index()
              .rename(columns={"mean": f"{metric}_mean",
                               "std": f"{metric}_std", "count": "n_folds"})
              .sort_values(f"{metric}_mean", ascending=asc)
              .reset_index(drop=True))
    ranked.insert(0, "RankID", np.arange(1, len(ranked) + 1))
    return ranked, f"folds-aggregated: mean {metric} over {len(fold_files)} folds"


# ============================================================
# MAIN
# ============================================================

def run_prediction(run_dir=None, targets=None, rank_ids=None,
                   x_infer_path=None, interactive=True):
    """rank_ids: dict {target_name: RankID} (missing entries default to 1)."""
    rank_ids = dict(rank_ids or {})

    # ---- resolve run dir ----
    if run_dir is None:
        latest = _latest_run(RUNS_BASE)
        if interactive:
            s = input(f"Run folder? Blank = latest ({latest}):\n> ").strip()
            run_dir = Path(s) if s else latest
        else:
            run_dir = latest
    run_dir = Path(run_dir) if run_dir else None
    if run_dir is None or not run_dir.exists():
        raise FileNotFoundError(f"Run folder not found: {run_dir}. Train first (srf_train.py).")

    meta_dir = run_dir / "00_metadata"
    with open(meta_dir / "run_config.json", encoding="utf-8") as f:
        run_cfg = json.load(f)
    _verify_multiscale_contract(run_cfg)
    prepared = joblib.load(meta_dir / "prepared.pkl")
    rot_maps = prepared["rot_maps"]
    aniso_pack = prepared["aniso_pack"]
    train_idx = prepared["train_idx"]

    task = run_cfg["task"]
    expected_d = int(run_cfg["expected_d"])
    print("=" * 70)
    print(f"RUN     : {run_dir}")
    print(f"TASK    : {task} | kernel k={run_cfg['kernel_size']} | "
          f"aug={run_cfg['use_augmentation']} | aniso={run_cfg['use_aniso_feats']}")
    print("=" * 70)

    # ---- resolve targets ----
    all_targets = list(run_cfg["target_cols"].keys())
    if targets is None:
        if interactive:
            s = input(f"Target? {all_targets} or blank = all:\n> ").strip()
            targets = [s] if s else all_targets
        else:
            targets = all_targets
    for t in targets:
        if t not in all_targets:
            raise ValueError(f"Unknown target {t}; available: {all_targets}")

    # ---- resolve X_infer: prefer the exact contract recorded during training ----
    k = int(run_cfg["kernel_size"])
    recorded_infer = run_cfg.get("x_infer_path")
    if recorded_infer:
        default_infer = Path(recorded_infer)
    else:
        default_infer = X_INFER_PATH.with_name(f"X_infer_{k ** 3}.npy")
        if not default_infer.exists():
            default_infer = X_INFER_PATH
    if x_infer_path is None:
        if interactive:
            s = input(f"Block-model X file? Blank = {default_infer}:\n> ").strip()
            x_infer_path = Path(s) if s else default_infer
        else:
            x_infer_path = default_infer
    x_infer_path = Path(x_infer_path)
    if not x_infer_path.exists():
        raise FileNotFoundError(f"X_infer not found: {x_infer_path}")

    # ---- load data ----
    X = np.load(run_cfg["X_path"]).astype(np.float32)
    y_all = np.load(run_cfg["y_path"])
    assert X.shape[1] == expected_d, f"X_train D mismatch: {X.shape[1]} vs {expected_d}"

    X_inf = np.load(x_infer_path).astype(np.float32)
    assert X_inf.shape[1] == expected_d, (
        f"X_infer has D={X_inf.shape[1]}, expected {expected_d} — was it built "
        f"with the same kernel/props as training?")
    n_inf = X_inf.shape[0]
    print(f"X_infer: {X_inf.shape} from {x_infer_path}\n")

    out_pred = run_dir / "05_full_prediction"
    out_models = run_dir / "02_models"

    for target in targets:
        ranked, grid_mode = _load_ranked_grid(run_dir, target, task)

        show_cols = [c for c in
                     ["RankID", "SUBCV_R2_mean", "SUBCV_R2_std", "SUBCV_ERR_mean",
                      "SUBCV_ERR_std", "n_folds",
                      "OOB_R2", "OOB_RMSE", "OOB_MAE", "OOB_ERR", "OOB_ACC",
                      "OOB_coverage", "n_estimators", "max_depth", "min_leaf",
                      "max_features", "max_samples", "split_quantiles", "fit_time_sec"]
                     if c in ranked.columns]
        print("-" * 70)
        print(f"SELECT MODEL | target: {target} | ranking: {grid_mode}")
        print(ranked.head(TOP_K_SHOW)[show_cols].to_string(index=False))
        print("-" * 70)

        rank = rank_ids.get(target)
        if rank is None:
            if interactive:
                s = input(f"RankID for {target}? Blank = 1:\n> ").strip()
                rank = int(s) if s else 1
            else:
                rank = 1
        if rank not in ranked["RankID"].values:
            raise ValueError(f"RankID {rank} not in grid table for {target}")
        row = ranked.loc[ranked["RankID"] == rank].iloc[0]
        cfg = _cfg_from_row(row)
        if N_ESTIMATORS_OVERRIDE is not None:
            cfg["n_estimators"] = int(N_ESTIMATORS_OVERRIDE)

        # ---- y for this target ----
        col = run_cfg["target_cols"][target]
        y_t = y_all[:, col] if y_all.ndim == 2 else y_all
        if task == "regression":
            y_t = y_t.astype(np.float32)

        if REFIT_ON == "train":
            X_fit, y_fit = X[train_idx], y_t[train_idx]
        else:
            X_fit, y_fit = X, y_t

        print(f"\nRefit {target} | Rank {rank} | on {REFIT_ON.upper()} data "
              f"({X_fit.shape[0]} rows) | {cfg}")
        t0 = time.perf_counter()
        model = build_forest_from_config(cfg, run_cfg, rot_maps, aniso_pack, REFIT_SEED)
        model.fit(X_fit, y_fit)
        fit_time = time.perf_counter() - t0
        oob = model.oob_scores(y_fit)
        print(f"Refit done in {fit_time:.1f} s | OOB sanity: "
              + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                         for k, v in oob.items()))

        # ---- batched inference ----
        print(f"Predicting {n_inf} block rows (batch {BATCH_SIZE}, tta={USE_TTA})...")
        t1 = time.perf_counter()
        if task == "regression":
            pred = np.empty(n_inf, dtype=np.float32)
            p_exc = {thr: np.empty(n_inf, dtype=np.float32)
                     for thr in EXCEEDANCE_THRESHOLDS}
            for s0 in range(0, n_inf, BATCH_SIZE):
                s1 = min(s0 + BATCH_SIZE, n_inf)
                if p_exc:
                    # per-tree predictions on the identity view: mean = the
                    # prediction, fraction above thr = exceedance probability
                    Xe = model.build_design_matrix(X_inf[s0:s1])
                    tp = np.stack([t.predict(Xe) for t in model.trees_])
                    for thr, arr in p_exc.items():
                        arr[s0:s1] = (tp > thr).mean(axis=0)
                    pred[s0:s1] = (model.predict(X_inf[s0:s1], tta=True)
                                   if USE_TTA else tp.mean(axis=0))
                else:
                    pred[s0:s1] = model.predict(X_inf[s0:s1], tta=USE_TTA)
            df_out = pd.DataFrame({"row": np.arange(n_inf), f"PRED_{target}": pred})
            for thr, arr in p_exc.items():
                df_out[f"P_GT{thr:g}"] = arr
            extra = {"pred_min": float(pred.min()), "pred_max": float(pred.max()),
                     "pred_mean": float(pred.mean()),
                     **{f"frac_P_GT{thr:g}>=0.5": float((arr >= 0.5).mean())
                        for thr, arr in p_exc.items()}}
        else:
            labels = np.empty(n_inf, dtype=object)
            entropy = np.empty(n_inf, dtype=np.float32)
            pmax = np.empty(n_inf, dtype=np.float32)
            proba_all = np.empty((n_inf, len(model.classes_)), dtype=np.float32)
            for s0 in range(0, n_inf, BATCH_SIZE):
                s1 = min(s0 + BATCH_SIZE, n_inf)
                proba = model.predict_proba(X_inf[s0:s1], tta=USE_TTA)
                proba_all[s0:s1] = proba
                labels[s0:s1] = model.classes_[np.argmax(proba, axis=1)]
                pmax[s0:s1] = proba.max(axis=1)
                p = np.clip(proba, 1e-12, 1.0)
                entropy[s0:s1] = (-(p * np.log(p)).sum(axis=1)
                                  / np.log(proba.shape[1])).astype(np.float32)
            pred = labels
            df_out = pd.DataFrame({"row": np.arange(n_inf),
                                   f"PRED_{target}": labels,
                                   "PROBA_MAX": pmax, "ENTROPY": entropy,
                                   **{f"P_{c}": proba_all[:, k]
                                      for k, c in enumerate(model.classes_)}})
            extra = {"entropy_mean": float(entropy.mean()),
                     **{f"frac_pred_{c}": float(np.mean(labels == c))
                        for c in model.classes_}}
        pred_time = time.perf_counter() - t1
        print(f"Prediction done in {pred_time:.1f} s")

        # ---- save ----
        tag = f"{target}_Rank{rank}"
        npy_path = out_pred / f"PRED_{tag}.npy"
        csv_path = out_pred / f"PRED_{tag}.csv"
        model_path = out_models / f"deploy_{tag}.joblib"
        meta_path = out_pred / f"prediction_meta_{tag}.json"

        np.save(npy_path, pred if task == "regression" else pred.astype(str))
        df_out.to_csv(csv_path, index=False)

        meta = {
            "created_at": datetime.now().isoformat(),
            "run_dir": str(run_dir),
            "target": target,
            "rank_id": int(rank),
            "task": task,
            "refit_on": REFIT_ON,
            "n_estimators_override": N_ESTIMATORS_OVERRIDE,
            "refit_seed": REFIT_SEED,
            "use_tta": bool(USE_TTA),
            "selected_config": _to_py(cfg),
            "grid_row": _to_py(row.to_dict()),
            "oob_after_refit": _to_py(oob),
            "x_infer_path": str(x_infer_path),
            "n_infer_rows": int(n_inf),
            "fit_time_sec": float(fit_time),
            "pred_time_sec": float(pred_time),
            "model_path": str(model_path),
            **extra,
        }
        model.meta_ = meta
        joblib.dump(model, model_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(_to_py(meta), f, ensure_ascii=False, indent=2)

        print(f"Saved prediction: {csv_path}")
        print(f"Saved npy       : {npy_path}")
        print(f"Saved model     : {model_path}")
        print(f"Saved meta      : {meta_path}\n")

    print("=" * 70)
    print("PREDICTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_prediction()
