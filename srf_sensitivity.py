# -*- coding: utf-8 -*-
r"""
SRF hyperparameter SENSITIVITY ANALYSIS (local_aniso, augmentation off, k=3).

Why this exists
---------------
The production grid in srf_train.py is deliberately narrow. The worry is that
its ranges are centred in the wrong place — that the real optimum lies at, or
beyond, an edge of the grid, so per-fold tuning keeps landing in the same local
region. This script answers that directly: it sweeps each hyperparameter across
a WIDER range than the production grid and reports where accuracy actually
peaks, and whether the peak sits on a boundary.

Note on PROD below: those are the production levels AS THEY WERE when this
analysis was run (216 configs). The grid now in srf_train.py is the 27-config
result of this study, so re-running the script draws the old grid as its
reference band. That is deliberate — it is the comparison the study makes.

What is measured
----------------
For every config, the signal is POOLED spatial-CV R^2: train on each outer
fold's training blocks, predict its held-out block, pool all 456 predictions,
score once. That is the honest generalisation number the paper reports — every
sample predicted by a model that never saw its spatial neighbourhood. It is used
here only to CHARACTERISE the response surface, not to select the deployed model
(the pipeline still tunes by nested sub-fold CV inside each fold), so there is no
selection leakage: nothing downstream consumes these numbers.

Design
------
Factorial sweep at fixed n_estimators (past the saturation point), over ranges
that extend beyond the production grid on both sides:

    max_depth     2  3  4  6  8  10 14      (production: 4 6 10)
    min_leaf      1  2  4  8  16 30 50       (production: 4 10 20)
    max_features  sqrt third 0.5 0.75 all    (production: sqrt third 0.5)
    max_samples   0.5 0.7 1.0                (production: 0.7 1.0)

Plus a separate 1-D n_estimators curve at each target's best factorial config,
to show the saturation point on THIS data rather than citing a side-study.

Anisotropy features are recomputed per fold from that fold's training rows only
(leakage-safe), exactly as in srf_train.py. Augmentation is off.

Outputs (<root>\SRF_Sensitivity\SENS_<timestamp>\):
  sensitivity_grid.csv         config x target -> pooled R2/RMSE/MAE + per-fold R2
  marginal_effects.csv         per hyperparameter level: median/IQR/max pooled R2
  n_estimators_curve.csv       pooled R2 vs n_estimators at the best config
  best_configs.csv             the single best config per target + boundary flags
  fig_marginal_{TARGET}.png    box+strip of pooled R2 by level, one panel/factor
  fig_depth_leaf_{TARGET}.png  max_depth x min_leaf heatmap (regularisation pair)
  fig_nestimators.png          saturation curve
  run_config.json

Run:  python -u srf_sensitivity.py
"""

import json
import sys
import time
from datetime import datetime
from itertools import product
from pathlib import Path

try:
    _HERE = Path(__file__).resolve().parent
except NameError:                       # notebook cell
    _HERE = Path.cwd()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

import srf_train as st

# ---------------- configuration ----------------
PATTERN = "local_aniso"     # anisotropy on, augmentation off (the better model)
KERNEL = "k3"
N_JOBS = 16
N_EST_FIXED = 200           # comfortably past saturation for the factorial

# ranges extended beyond the production grid on both sides
SWEEP = {
    "max_depth":    [2, 3, 4, 6, 8, 10, 14],
    "min_leaf":     [1, 2, 4, 8, 16, 30, 50],
    "max_features": ["sqrt", "third", 0.5, 0.75, "all"],
    "max_samples":  [0.5, 0.7, 1.0],
}
N_EST_CURVE = [30, 50, 75, 100, 150, 200, 300, 500, 750]

# production grid, so the plots can mark where the current search looks
PROD = {
    "max_depth":    [4, 6, 10],
    "min_leaf":     [4, 10, 20],
    "max_features": ["sqrt", "third", 0.5],
    "max_samples":  [0.7, 1.0],
    "n_estimators": [100, 150, 300, 500],
}

FIXED = {"split_quantiles": 64, "bootstrap": True}
OUT_BASE = st.PROJECT_ROOT / "SRF_Sensitivity"


def pooled_metrics(yt, yp):
    yt = np.asarray(yt, np.float64)
    yp = np.asarray(yp, np.float64)
    ss_res = float(((yt - yp) ** 2).sum())
    ss_tot = float(((yt - yt.mean()) ** 2).sum())
    return (1.0 - ss_res / max(ss_tot, 1e-300),
            float(np.sqrt(ss_res / yt.size)),
            float(np.abs(yt - yp).mean()))


def mf_label(v):
    return v if isinstance(v, str) else f"{v:g}"


def fit_predict_fold(cfg, y_tr, ext_tr, ext_te0, rot_maps, aniso_pack, run_cfg, seed):
    """One config on one fold: fit on train views, predict the held-out block."""
    model = st.build_forest_from_config(cfg, run_cfg, rot_maps, aniso_pack, seed)
    model.fit(None, y_tr, ext_rots=ext_tr)
    return model.predict(None, ext0=ext_te0)


def main():
    st.set_variant(pattern=PATTERN, kernel=KERNEL)
    assert st.USE_AUGMENTATION is False and st.USE_ANISO_FEATS is True

    run_dir = OUT_BASE / ("SENS_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 74)
    print(f"SRF SENSITIVITY - {PATTERN}, {KERNEL}, aug={st.USE_AUGMENTATION}")
    print(f"run dir: {run_dir}")
    print("=" * 74)

    X = np.load(st.X_path).astype(np.float32)
    y_all = np.load(st.y_path)
    roles = np.load(st.CV_ROLES_PATH)
    N, D = X.shape
    n_folds = roles.shape[1]
    n_props_total = len(st.PROPS_TOTAL)
    n_props_aniso = len(st.PROPS_COV)
    print(f"X {X.shape} | roles {roles.shape} | folds {n_folds}")

    rotations = [((0, 1, 2), (1, 1, 1))]        # augmentation off
    rot_maps = st.build_rotation_index_maps(st.KERNEL_SIZE, n_props_total, rotations)
    rot_maps_vox = st.voxel_maps_from_feature_maps(rot_maps, n_props_total)
    run_cfg = {"task": "regression", "kernel_size": int(st.KERNEL_SIZE),
               "n_props_total": n_props_total, "n_props_aniso": n_props_aniso,
               "n_static_features": 0, "use_augmentation": False,
               "use_aniso_feats": True, "aniso_var_log1p": st.ANISO_VAR_LOG1P,
               "aniso_var_zscore": st.ANISO_VAR_ZSCORE}

    # ---- config list (factorial) ----
    keys = list(SWEEP)
    factorial = [dict(zip(keys, vals), n_estimators=N_EST_FIXED, **FIXED)
                 for vals in product(*SWEEP.values())]
    print(f"factorial configs: {len(factorial)} (n_estimators fixed {N_EST_FIXED})")
    Dfeat = D + (n_props_aniso * st.ANISO_FEATS_PER_PROP)
    print(f"design width ~ {Dfeat} features -> "
          f"sqrt={int(np.sqrt(Dfeat))}, third={int(np.ceil(Dfeat/3))}, "
          f"0.5={int(np.ceil(0.5*Dfeat))}, 0.75={int(np.ceil(0.75*Dfeat))}, all={Dfeat}")

    def sweep_configs(cfg_list, targets):
        """Pooled spatial-CV predictions for every config in cfg_list."""
        # pred[target][cfg_id] accumulates (row, y_pred) across folds
        preds = {t: {i: [] for i in range(len(cfg_list))} for t in targets}
        truth = {t: {} for t in targets}
        for f in range(n_folds):
            tr = roles[:, f] == 0
            te = roles[:, f] == 1
            te_rows = np.where(te)[0]
            aniso_pack = st.compute_aniso_pack(X[tr], rot_maps, rot_maps_vox,
                                               n_props_total, n_props_aniso)
            proto = st.build_forest_from_config(cfg_list[0], run_cfg, rot_maps,
                                                aniso_pack, 0)
            ext_full = proto.build_design_matrices(X)
            ext_tr = [E[tr] for E in ext_full]
            ext_te0 = ext_full[0][te]
            for t in targets:
                col = st.TARGET_COLS[t]
                y_tr = y_all[tr, col].astype(np.float32)
                y_te = y_all[te, col].astype(np.float64)
                for i, r in enumerate(te_rows):
                    truth[t][int(r)] = y_te[i]
                t0 = time.perf_counter()
                fold_preds = Parallel(n_jobs=N_JOBS)(
                    delayed(fit_predict_fold)(
                        cfg, y_tr, ext_tr, ext_te0, rot_maps, aniso_pack,
                        run_cfg, st.REFIT_SEED + cid)
                    for cid, cfg in enumerate(cfg_list))
                for cid, yp in enumerate(fold_preds):
                    for i, r in enumerate(te_rows):
                        preds[t][cid].append((int(r), float(yp[i])))
                print(f"  fold {f+1} {t}: {len(cfg_list)} configs "
                      f"in {time.perf_counter()-t0:.1f}s")
        # pool
        rows = []
        for t in targets:
            for cid, cfg in enumerate(cfg_list):
                pr = preds[t][cid]
                r_ids = np.array([r for r, _ in pr])
                yp = np.array([v for _, v in pr])
                yt = np.array([truth[t][int(r)] for r in r_ids])
                r2, rmse, mae = pooled_metrics(yt, yp)
                # per-fold R2 too
                perfold = {}
                for f in range(n_folds):
                    te_rows = set(np.where(roles[:, f] == 1)[0].tolist())
                    m = np.array([r in te_rows for r in r_ids])
                    if m.sum() > 2:
                        perfold[f"R2_f{f+1}"] = pooled_metrics(yt[m], yp[m])[0]
                rows.append({"target": t, "config_id": cid,
                             **{k: mf_label(v) if k == "max_features" else v
                                for k, v in cfg.items()},
                             "pooled_R2": r2, "pooled_RMSE": rmse,
                             "pooled_MAE": mae, **perfold})
        return pd.DataFrame(rows)

    targets = list(st.TARGET_COLS)      # DTR, MAGNETIC
    print("\n--- factorial sweep ---")
    t_start = time.time()
    grid = sweep_configs(factorial, targets)
    grid.to_csv(run_dir / "sensitivity_grid.csv", index=False)
    print(f"factorial done in {(time.time()-t_start)/60:.1f} min")

    # ---- marginal effects per hyperparameter level ----
    marg_rows = []
    for t in targets:
        g = grid[grid.target == t]
        for hp in keys:
            for lv in SWEEP[hp]:
                key = mf_label(lv) if hp == "max_features" else lv
                sel = g[g[hp] == key]
                if len(sel):
                    marg_rows.append({
                        "target": t, "hyperparameter": hp, "level": str(key),
                        "n_configs": len(sel),
                        "R2_median": float(sel.pooled_R2.median()),
                        "R2_q25": float(sel.pooled_R2.quantile(.25)),
                        "R2_q75": float(sel.pooled_R2.quantile(.75)),
                        "R2_max": float(sel.pooled_R2.max()),
                        "in_production_grid": key in [
                            mf_label(v) if hp == "max_features" else v
                            for v in PROD[hp]]})
    marg = pd.DataFrame(marg_rows)
    marg.to_csv(run_dir / "marginal_effects.csv", index=False)

    # ---- best config per target + boundary check ----
    best_rows = []
    best_cfgs = {}
    for t in targets:
        g = grid[grid.target == t].sort_values("pooled_R2", ascending=False)
        b = g.iloc[0]
        best_cfgs[t] = b
        flags = []
        for hp in keys:
            lv = b[hp]
            levels = [mf_label(v) if hp == "max_features" else v for v in SWEEP[hp]]
            # boundary only meaningful for the ordered numeric factors
            if hp in ("max_depth", "min_leaf", "max_samples"):
                if lv == levels[0]:
                    flags.append(f"{hp}@LOW_edge({lv})")
                elif lv == levels[-1]:
                    flags.append(f"{hp}@HIGH_edge({lv})")
        best_rows.append({"target": t, "pooled_R2": float(b.pooled_R2),
                          "pooled_RMSE": float(b.pooled_RMSE),
                          **{k: b[k] for k in keys},
                          "n_estimators": int(b.n_estimators),
                          "boundary_flags": "; ".join(flags) if flags else "interior"})
    pd.DataFrame(best_rows).to_csv(run_dir / "best_configs.csv", index=False)

    # ---- n_estimators curve at each target's best config ----
    print("\n--- n_estimators curve ---")
    curve_frames = []
    for t in targets:
        b = best_cfgs[t]
        base = {k: (b[k] if k != "max_features"
                    else (b[k] if b[k] in ("sqrt", "third", "all") else float(b[k])))
                for k in keys}
        cfgs = [dict(base, n_estimators=ne, **FIXED) for ne in N_EST_CURVE]
        sub = sweep_configs(cfgs, [t])
        sub["n_estimators"] = [c["n_estimators"] for c in cfgs]
        curve_frames.append(sub)
    curve = pd.concat(curve_frames, ignore_index=True)
    curve.to_csv(run_dir / "n_estimators_curve.csv", index=False)

    # ---- figures ----
    make_figures(run_dir, grid, marg, curve, best_cfgs, targets)

    with open(run_dir / "run_config.json", "w", encoding="utf-8") as fj:
        json.dump({"created_at": datetime.now().isoformat(),
                   "pattern": PATTERN, "kernel": KERNEL,
                   "n_estimators_fixed": N_EST_FIXED, "sweep": SWEEP,
                   "n_est_curve": N_EST_CURVE, "fixed": FIXED,
                   "production_grid": PROD,
                   "signal": "pooled spatial-CV R2 over 5 outer folds",
                   "n_samples": int(N), "elapsed_min": (time.time()-t_start)/60},
                  fj, indent=2)

    # ---- console summary ----
    print("\n" + "=" * 74)
    print("BEST CONFIG PER TARGET (widened grid)")
    print("=" * 74)
    print(pd.DataFrame(best_rows).to_string(index=False))
    print("\nMARGINAL EFFECT - median pooled R2 by level "
          "(* = level also in production grid):")
    for t in targets:
        print(f"\n  --- {t} ---")
        for hp in keys:
            row = marg[(marg.target == t) & (marg.hyperparameter == hp)]
            cells = "  ".join(
                f"{r.level}{'*' if r.in_production_grid else ' '}:{r.R2_median:.3f}"
                for _, r in row.iterrows())
            print(f"    {hp:13s} {cells}")
    print(f"\nSaved to {run_dir}")
    print("=" * 74)
    return run_dir


def make_figures(run_dir, grid, marg, curve, best_cfgs, targets):
    keys = list(SWEEP)
    for t in targets:
        g = grid[grid.target == t]
        fig, axes = plt.subplots(1, len(keys), figsize=(4.0 * len(keys), 4.2))
        for ax, hp in zip(axes, keys):
            levels = [mf_label(v) if hp == "max_features" else v for v in SWEEP[hp]]
            data = [g[g[hp] == lv].pooled_R2.values for lv in levels]
            bp = ax.boxplot(data, showfliers=False, widths=0.6,
                            medianprops=dict(color="black"))
            for xi, (lv, arr) in enumerate(zip(levels, data), start=1):
                ax.scatter(np.full(len(arr), xi) + np.random.uniform(-.12, .12, len(arr)),
                           arr, s=8, alpha=0.35, color="#4C78A8")
                inprod = lv in [mf_label(v) if hp == "max_features" else v
                                for v in PROD[hp]]
                ax.get_xticklabels()
            ax.set_xticklabels([f"{lv}\n{'(prod)' if lv in [mf_label(v) if hp=='max_features' else v for v in PROD[hp]] else ''}"
                                for lv in levels], fontsize=8)
            ax.set_title(hp, fontsize=10)
            ax.axhline(g.pooled_R2.max(), ls=":", lw=0.8, color="crimson")
            ax.grid(axis="y", alpha=0.25)
        axes[0].set_ylabel("pooled spatial-CV $R^2$")
        fig.suptitle(f"SRF hyperparameter sensitivity — {t}  "
                     f"(local_aniso, aug off, k=3; dotted = best config)",
                     fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(run_dir / f"fig_marginal_{t}.png", dpi=150)
        plt.close(fig)

        # depth x min_leaf heatmap (mean over max_features, max_samples)
        piv = (g.pivot_table(index="min_leaf", columns="max_depth",
                             values="pooled_R2", aggfunc="mean")
               .reindex(index=SWEEP["min_leaf"], columns=SWEEP["max_depth"]))
        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(piv.values, origin="lower", aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(SWEEP["max_depth"]))); ax.set_xticklabels(SWEEP["max_depth"])
        ax.set_yticks(range(len(SWEEP["min_leaf"]))); ax.set_yticklabels(SWEEP["min_leaf"])
        ax.set_xlabel("max_depth"); ax.set_ylabel("min_leaf")
        for (yy, xx), v in np.ndenumerate(piv.values):
            if np.isfinite(v):
                ax.text(xx, yy, f"{v:.3f}", ha="center", va="center",
                        color="white" if v < np.nanmean(piv.values) else "black",
                        fontsize=7)
        b = best_cfgs[t]
        ax.scatter(SWEEP["max_depth"].index(int(b.max_depth)),
                   SWEEP["min_leaf"].index(int(b.min_leaf)),
                   marker="*", s=300, edgecolor="red", facecolor="none", linewidths=2)
        fig.colorbar(im, ax=ax, label="mean pooled $R^2$")
        ax.set_title(f"depth x min_leaf response surface — {t}\n"
                     "(mean over max_features, max_samples; star = best config)")
        fig.tight_layout()
        fig.savefig(run_dir / f"fig_depth_leaf_{t}.png", dpi=150)
        plt.close(fig)

    # n_estimators saturation
    fig, ax = plt.subplots(figsize=(7, 5))
    for t in targets:
        c = curve[curve.target == t].sort_values("n_estimators")
        ax.plot(c.n_estimators, c.pooled_R2, "o-", label=t)
    ax.set_xlabel("n_estimators"); ax.set_ylabel("pooled spatial-CV $R^2$")
    ax.axvspan(min(PROD["n_estimators"]), max(PROD["n_estimators"]),
               alpha=0.12, color="green", label="production range")
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("n_estimators saturation at each target's best config")
    fig.tight_layout()
    fig.savefig(run_dir / "fig_nestimators.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
