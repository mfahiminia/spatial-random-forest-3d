# -*- coding: utf-8 -*-
r"""
Export the REPORTED GLS-RF model from a finished GLSRF.py fold run.

Why this is a separate step
---------------------------
A GLSRF.py run searches the covariance design as well as the tree
hyperparameters, and its `predictions_long_GLSRF.csv` therefore holds, for each
fold, whatever design won in that fold — sometimes isotropic, sometimes
anisotropic — scored under whichever prediction rule was active.

That is the right output for the design comparison, and the wrong output to
report as "the model". The reported estimator is one fixed specification:

    * covariance design  ANISO / exp, held FIXED across all folds
    * prediction rule    the mean function m(X) only (Eq. 11 residual kriging
                         is computed and stored, but is not the headline)

Selection stays honest. The design is chosen by the mean sub-fold CV score
across folds, computed inside each fold's training rows; the held-out blocks
are never consulted. Within that fixed design, the tree hyperparameters are
still the per-fold sub-CV winners, exactly as during the run.

Why not just re-run with the design pinned: the random seed of each fit is
derived from the configuration's index in the full enumeration
(`seed = SEED + fold*10000 + config_id*10`). Shrinking the grid renumbers the
configurations and therefore reseeds every forest, which perturbs the results
for no reason. Reading the run's own grid tables keeps the indices, and the
seeds, identical to the search that produced them.

What it does
------------
For every target x fold: read `grid_{target}_fold{N}.csv`, keep only the rows
with the fixed design, take the best `SUBCV_R2`, recover that configuration's
id (hence its seed), refit on the fold's training rows and predict the held-out
block. Then pool.

Outputs (<run>/../GLSRF_reported_ANISOexp/):
  predictions_long_GLSRF.csv   row,x,y,z,fold,target,method,y_true,y_pred,
                               y_pred_mean,y_pred_krig,y_unc
  fold_metrics.csv             per target x fold, both prediction rules
  pooled_summary.csv           the headline table
  selected_configs.csv         the configuration and seed used in each fold
  run_config.json              provenance, including the source run

Run:  python glsrf_report.py [run_folder]
      (blank = the most recent GLSRF run that has grid tables)
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    _HERE = Path(__file__).resolve().parent
except NameError:                       # notebook cell
    _HERE = Path.cwd()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import GLSRF as G


# ---- the reported specification ----------------------------------------
DESIGN_MODE = "ANISO"        # vs the "ISO" control
DESIGN_KERNEL = "exp"        # vs "matern32"
RANK = "SUBCV_R2"            # mean-rule sub-CV; "SUBCV_R2_K" would rank kriged
OUT_NAME = "GLSRF_reported_ANISOexp"


def latest_run() -> Path:
    runs = sorted(p for p in G.RUNS_BASE.glob("GLSRF_*") if p.is_dir()
                  and (p / f"grid_{G.TARGET_COLS[0]}_fold1.csv").exists())
    if not runs:
        raise FileNotFoundError(
            f"No GLSRF run folders with grid tables in {G.RUNS_BASE}. "
            "Run GLSRF.py first.")
    return runs[-1]


def choose_design(run_dir: Path, n_folds: int):
    """Mean sub-CV score of each (Mode, Kernel) across folds and targets.

    Printed so the fixed design is visibly the one the selection metric picks,
    rather than an assertion in a docstring.
    """
    rows = []
    for t in G.TARGET_COLS:
        for f in range(1, n_folds + 1):
            g = pd.read_csv(run_dir / f"grid_{t}_fold{f}.csv").dropna(subset=[RANK])
            for (mode, kern), sl in g.groupby(["Mode", "Kernel"]):
                rows.append({"Mode": mode, "Kernel": kern, "target": t,
                             "fold": f, RANK: float(sl[RANK].max())})
    tab = (pd.DataFrame(rows).groupby(["Mode", "Kernel"])[RANK]
           .mean().reset_index().sort_values(RANK, ascending=False))
    print(f"\nDesign ranking by mean {RANK} across folds and targets:")
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    top = tab.iloc[0]
    if (top["Mode"], top["Kernel"]) != (DESIGN_MODE, DESIGN_KERNEL):
        print(f"\n[NOTE] the best design here is {top['Mode']}/{top['Kernel']}, "
              f"but this script exports the declared {DESIGN_MODE}/{DESIGN_KERNEL}. "
              "Change DESIGN_MODE/DESIGN_KERNEL, or report what the metric picked.")
    return tab


def main(run_dir=None):
    run_dir = Path(run_dir) if run_dir else latest_run()
    out_dir = run_dir.parent / OUT_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"REPORTED GLS-RF MODEL: {DESIGN_MODE}/{DESIGN_KERNEL}, "
          f"mean-function prediction")
    print(f"source run: {run_dir.name}")
    print("=" * 72)

    df = (pd.read_excel(G.DATA_PATH) if G.DATA_PATH.suffix.lower() in (".xlsx", ".xls")
          else pd.read_csv(G.DATA_PATH, low_memory=False))
    for c in G.COORD_COLS + G.FEATURE_COLS + G.TARGET_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    roles = np.load(G.ROLES_PATH)
    if roles.shape[0] != len(df):
        raise ValueError(f"roles rows ({roles.shape[0]}) != data rows ({len(df)}).")
    n_folds = roles.shape[1]
    coords_all = df[G.COORD_COLS].values.astype(np.float64)
    configs = list(G.iter_configs())

    choose_design(run_dir, n_folds)

    long_rows, metric_rows, sel_rows = [], [], []
    for target in G.TARGET_COLS:
        y_all = df[target].values.astype(np.float64)
        for f in range(n_folds):
            fold_no = f + 1
            g = pd.read_csv(run_dir / f"grid_{target}_fold{fold_no}.csv")
            g = g[(g.Mode == DESIGN_MODE) & (g.Kernel == DESIGN_KERNEL)]
            g = g.dropna(subset=[RANK])
            if g.empty:
                raise RuntimeError(
                    f"No {DESIGN_MODE}/{DESIGN_KERNEL} rows scored in "
                    f"grid_{target}_fold{fold_no}.csv — was the run made with "
                    "COMPARE_DESIGNS enabled?")
            best = g.loc[g[RANK].idxmax()]
            cfg_id = int(best["config_id"])
            # the seed formula of run_folds(), so the refit matches the search
            seed = G.SEED + f * 10_000 + cfg_id * 10
            cfg = configs[cfg_id]
            if cfg["Mode"] != DESIGN_MODE or cfg["Kernel"] != DESIGN_KERNEL:
                raise RuntimeError(
                    f"config_id {cfg_id} is {cfg['Mode']}/{cfg['Kernel']} in the "
                    f"current enumeration but {DESIGN_MODE}/{DESIGN_KERNEL} in the "
                    "run's grid table. Section 2/3 of GLSRF.py changed since the "
                    "run — re-run instead of exporting.")

            tr = (roles[:, f] == 0) & ~np.isnan(y_all)
            te = roles[:, f] == 1
            med = df.loc[tr, G.FEATURE_COLS].median(numeric_only=True)
            X_all = df[G.FEATURE_COLS].fillna(med).values.astype(np.float64)

            model = G.build_model(cfg, seed)
            model.fit(X_all[tr], y_all[tr], coords_all[tr])
            oob = model.oob_r2(y_all[tr])
            tree_preds = model.predict_all(X_all[te])
            y_mean = tree_preds.mean(axis=0)
            y_unc = tree_preds.std(axis=0)
            y_krig = model.predict_kriged(X_all[te], coords_all[te],
                                          mean_pred=y_mean)

            y_true_te = y_all[te]
            for i, r_id in enumerate(np.where(te)[0]):
                long_rows.append({
                    "row": int(r_id), "x": coords_all[r_id, 0],
                    "y": coords_all[r_id, 1], "z": coords_all[r_id, 2],
                    "fold": fold_no, "target": target, "method": G.METHOD_NAME,
                    "y_true": y_true_te[i],
                    "y_pred": y_mean[i],          # the reported rule
                    "y_pred_mean": y_mean[i], "y_pred_krig": y_krig[i],
                    "y_unc": float(y_unc[i])})

            ok = ~np.isnan(y_true_te)
            r2, rmse, mae = G.regression_metrics(y_true_te[ok], y_mean[ok])
            r2k, rmsek, maek = G.regression_metrics(y_true_te[ok], y_krig[ok])
            print(f"  {target:9s} fold {fold_no} | cfg {cfg_id:3d} seed {seed:6d} "
                  f"| depth={cfg['max_depth']:2d} leaf={cfg['min_leaf']} "
                  f"mf={cfg['max_features']} | subCV={float(best[RANK]):.4f} "
                  f"TEST={r2:.4f} (kriged {r2k:.4f})")

            metric_rows.append({
                "target": target, "fold": fold_no, "method": G.METHOD_NAME,
                "n_train": int(tr.sum()), "n_test": int(ok.sum()),
                "Mode": DESIGN_MODE, "Kernel": DESIGN_KERNEL,
                "nugget": cfg["nugget"],
                **{k: G._to_py(v) for k, v in oob.items()},
                "TEST_R2": r2, "TEST_RMSE": rmse, "TEST_MAE": mae,
                "TEST_R2_K": r2k, "TEST_RMSE_K": rmsek, "TEST_MAE_K": maek,
                f"BEST_{RANK}": float(best[RANK])})
            sel_rows.append({"target": target, "fold": fold_no,
                             **{k: G._to_py(v) for k, v in cfg.items()},
                             "seed": seed})

    df_long = pd.DataFrame(long_rows)
    df_long.to_csv(out_dir / f"predictions_long_{G.METHOD_NAME}.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(out_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(sel_rows).to_csv(out_dir / "selected_configs.csv", index=False)

    # ---- pooled: every sample predicted exactly once ----
    pooled = []
    for target in G.TARGET_COLS:
        sl = df_long[df_long["target"] == target].dropna(subset=["y_true"])
        if len(sl) <= 2:
            continue
        yt = sl["y_true"].values.astype(np.float64)
        row = {"target": target, "method": G.METHOD_NAME, "n": len(sl)}
        for tag, col in (("", "y_pred_mean"), ("_K", "y_pred_krig")):
            yp = sl[col].values.astype(np.float64)
            ss_res = float(((yt - yp) ** 2).sum())
            ss_tot = float(((yt - yt.mean()) ** 2).sum())
            cov = float(((yt - yt.mean()) * (yp - yp.mean())).mean())
            row[f"POOLED_R2{tag}"] = 1.0 - ss_res / max(ss_tot, 1e-300)
            row[f"POOLED_RMSE{tag}"] = float(np.sqrt(ss_res / yt.size))
            row[f"POOLED_MAE{tag}"] = float(np.abs(yt - yp).mean())
            row[f"POOLED_CCC{tag}"] = float(
                2.0 * cov / (yt.var() + yp.var() + (yt.mean() - yp.mean()) ** 2))
        pooled.append(row)
    df_pool = pd.DataFrame(pooled)
    df_pool.to_csv(out_dir / "pooled_summary.csv", index=False)

    src_cfg = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    src_cfg.update({
        "created_at": datetime.now().isoformat(),
        "derived_from": run_dir.name,
        "reported_model": True,
        "design": f"{DESIGN_MODE}/{DESIGN_KERNEL} fixed across folds",
        "design_selected_on": f"mean {RANK} across folds",
        "prediction_rule": "mean function (Eq. 11 kriging computed but not used)",
        "residual_kriging": False,
        "rank_col": RANK,
    })
    (out_dir / "run_config.json").write_text(
        json.dumps(src_cfg, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print("REPORTED MODEL - pooled (headline is POOLED_R2, the mean rule)")
    print(df_pool[["target", "n", "POOLED_R2", "POOLED_RMSE", "POOLED_MAE",
                   "POOLED_CCC", "POOLED_R2_K"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nWritten to: {out_dir}")
    print("=" * 72)
    return out_dir


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(args[0] if args else None)
