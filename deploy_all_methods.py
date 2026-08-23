# -*- coding: utf-8 -*-
r"""
DEPLOY ALL METHODS on the final block model, into ONE file.

Continuity validated on held-out CV predictions describes the honest test; the
product the mine uses is the DEPLOYED map — every method refit on ALL of its
training data at its frozen CV-selected config (best_configs.json), predicting
every block-model node. This script produces that map for all seven methods on
one common node support, so the continuity of the final fields can be compared
like-for-like (continuity_block.py consumes the output).

Supports and routes (they genuinely differ, as the run history established):
  - Tree baselines + GLS-RF train on the classic table (551 samples) and predict
    from node covariates in Vector\grade_full_D3.csv (case-insensitive column
    mapping, rows kept if coords + >=1 covariate; train-median imputation happens
    inside classic_final.fit_predict / explicitly for GLS-RF).
  - SRF trains on the 456 k3 voxel vectors (local_aniso, aug off) and predicts
    the k3 patterns in Vector\X_infer_27.npy at centers_infer_27.npy.
  - The two node sets are joined 1:1 by coordinate (0.5 m tolerance, the proven
    threshold; ambiguous 0.5-1.0 m matches are a hard error). Only the
    intersection is written, so every row of the output carries ALL methods.

Output (BlockModel\):
  block_predictions_all.csv   x, y, z + <METHOD>_<TARGET> columns (14 fields)
  block_deploy_config.json    configs, sources, n rows, join stats

Run:  python -u deploy_all_methods.py
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# The sibling modules (srf_train, GLSRF, classic_final) are imported below,
# so this file's own folder has to be importable no matter where python
# was launched from.
try:
    _HERE = Path(__file__).resolve().parent
except NameError:                       # notebook cell
    _HERE = Path.cwd()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# Every input and output hangs off one root. Defaults to the repository folder;
# set the SRF_PROJECT_ROOT environment variable to run against data elsewhere.
RF = Path(os.environ.get("SRF_PROJECT_ROOT", _HERE))
OUT = RF / "BlockModel"
CFG_PATH = RF / "best_configs.json"
FULL_GRID = RF / "Vector" / "grade_full_D3.csv"
X_INFER = RF / "Vector" / "X_infer_27.npy"
CENTERS_INFER = RF / "Vector" / "centers_infer_27.npy"

MATCH_TOL_M = 0.5
REJECT_BAND_M = 1.0
SRF_BATCH = 8192

# Reported benchmark only: one representative per family.
CLASSIC_METHODS = ["RF", "RF_XYZ", "XGB"]
TARGETS = ["DTR", "Magnetic"]
SRF_TARGET_COL = {"DTR": 0, "Magnetic": 1}
GLSRF_CFG_COLS = ["Mode", "Kernel", "Azimuth", "Dip", "Tilt",
                  "A_major", "A_semi", "A_minor", "nugget", "n_estimators",
                  "max_depth", "min_leaf", "max_features", "max_samples",
                  "ridge", "jitter"]
DEPLOY_SEED = 20260724


def load_full_grid(coord_cols, feature_cols):
    df = pd.read_csv(FULL_GRID, low_memory=False)
    lower = {c.lower(): c for c in df.columns}
    ren = {}
    for c in coord_cols + feature_cols:
        if c not in df.columns:
            if c.lower() in lower:
                ren[lower[c.lower()]] = c
            elif c in ("XC", "YC", "ZC") and c[0].lower() in lower:
                ren[lower[c[0].lower()]] = c        # X -> XC fallback
            else:
                raise KeyError(f"full grid missing column {c}: "
                               f"{list(df.columns)}")
    if ren:
        df = df.rename(columns=ren)
        print(f"grid column mapping: {ren}")
    for c in coord_cols + feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    keep = df[coord_cols].notna().all(axis=1) & \
        (df[feature_cols].notna().sum(axis=1) >= 1)
    dropped = int((~keep).sum())
    df = df[keep].reset_index(drop=True)
    print(f"full grid: {len(df)} usable nodes ({dropped} dropped: no coords "
          "or no covariates)")
    return df


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfgs = json.loads(CFG_PATH.read_text(encoding="utf-8"))
    t0 = time.time()

    import classic_final as cf
    import GLSRF as gf
    import srf_train as st
    st.set_variant(pattern="local_aniso", kernel="k3")

    # ---- training table (551) ----
    df_tr = pd.read_excel(cf.DATA_PATH)
    for c in cf.COORD_COLS + cf.FEATURE_COLS + cf.TARGET_COLS:
        df_tr[c] = pd.to_numeric(df_tr[c].replace(cf.BAD_TOKENS, np.nan),
                                 errors="coerce")
    feats_tr = df_tr[cf.FEATURE_COLS].values.astype(np.float64)
    coords_tr = df_tr[cf.COORD_COLS].values.astype(np.float64)
    print(f"training table: {len(df_tr)} rows ({cf.DATA_PATH.name})")

    # ---- full grid + SRF inference support, joined 1:1 ----
    grid = load_full_grid(cf.COORD_COLS, cf.FEATURE_COLS)
    gcoords = grid[cf.COORD_COLS].values.astype(np.float64)
    X_inf = np.load(X_INFER).astype(np.float32)
    cen_inf = np.load(CENTERS_INFER).astype(np.float64)
    if len(X_inf) != len(cen_inf):
        raise RuntimeError("X_infer / centers_infer length mismatch")
    d, gi = cKDTree(gcoords).query(cen_inf)
    ok = d <= MATCH_TOL_M
    ambiguous = int(((d > MATCH_TOL_M) & (d < REJECT_BAND_M)).sum())
    if ambiguous:
        raise RuntimeError(f"{ambiguous} node joins in the {MATCH_TOL_M}-"
                           f"{REJECT_BAND_M} m band — join untrustworthy")
    idx_grid = gi[ok]
    if len(np.unique(idx_grid)) != len(idx_grid):
        raise RuntimeError("node join is not one-to-one")
    n_nodes = int(ok.sum())
    print(f"common node support: {n_nodes} of {len(cen_inf)} SRF nodes matched "
          f"to grid rows (max offset {d[ok].max():.3f} m; "
          f"{int((~ok).sum())} SRF nodes without grid covariates dropped)")

    feats_nodes = grid[cf.FEATURE_COLS].values.astype(np.float64)[idx_grid]
    coords_nodes = gcoords[idx_grid]
    X_inf = X_inf[ok]

    out = pd.DataFrame(coords_nodes, columns=["x", "y", "z"])

    # ---- SRF setup (trained on ALL 456 vectors) ----
    Xv = np.load(st.X_path).astype(np.float32)
    yv = np.load(st.y_path)
    n_pt, n_pa = len(st.PROPS_TOTAL), len(st.PROPS_COV)
    rotations = [((0, 1, 2), (1, 1, 1))]
    rot_maps = st.build_rotation_index_maps(st.KERNEL_SIZE, n_pt, rotations)
    rot_maps_vox = st.voxel_maps_from_feature_maps(rot_maps, n_pt)
    srf_run_cfg = {"task": "regression", "kernel_size": int(st.KERNEL_SIZE),
                   "n_props_total": n_pt, "n_props_aniso": n_pa,
                   "n_static_features": 0, "use_augmentation": False,
                   "use_aniso_feats": True,
                   "aniso_var_log1p": st.ANISO_VAR_LOG1P,
                   "aniso_var_zscore": st.ANISO_VAR_ZSCORE}
    pack = st.compute_aniso_pack(Xv, rot_maps, rot_maps_vox, n_pt, n_pa)

    for target in TARGETS:
        y_all = df_tr[target].values.astype(np.float64)
        ok_tr = ~np.isnan(y_all)
        print(f"\n=== {target} (train n={int(ok_tr.sum())}) ===")

        for m in CLASSIC_METHODS:
            t1 = time.time()
            p, _, _, _ = cf.fit_predict(
                m, feats_tr[ok_tr], coords_tr[ok_tr], y_all[ok_tr],
                feats_nodes, coords_nodes, DEPLOY_SEED,
                cfgs[m][target]["params"])
            out[f"{m}_{target}"] = p
            print(f"  {m:7s} deployed ({time.time()-t1:5.1f}s) "
                  f"mean={p.mean():.2f} sd={p.std():.2f}")

        # GLS-RF (median-impute features, coords appended)
        t1 = time.time()
        gcfg = {k: cfgs["GLSRF"][target]["params"][k] for k in GLSRF_CFG_COLS}
        med = np.nanmedian(feats_tr[ok_tr], axis=0)
        fe_tr = feats_tr.copy(); fe_nd = feats_nodes.copy()
        for j in range(fe_tr.shape[1]):
            fe_tr[np.isnan(fe_tr[:, j]), j] = med[j]
            fe_nd[np.isnan(fe_nd[:, j]), j] = med[j]
        Xg_tr = fe_tr   # coordinates are not GLS-RF features
        Xg_nd = fe_nd
        gm = gf.build_model(gcfg, DEPLOY_SEED)
        gm.fit(Xg_tr[ok_tr], y_all[ok_tr], coords_tr[ok_tr])
        p = np.asarray(gm.predict(Xg_nd), dtype=np.float64)
        out[f"GLSRF_{target}"] = p
        print(f"  GLSRF   deployed ({time.time()-t1:5.1f}s) "
              f"mean={p.mean():.2f} sd={p.std():.2f}")

        # SRF (refit on all 456 vectors, batched pattern inference)
        t1 = time.time()
        y_k = yv[:, SRF_TARGET_COL[target]].astype(np.float32)
        sm = st.build_forest_from_config(cfgs["SRF"][target]["params"],
                                         srf_run_cfg, rot_maps, pack,
                                         st.REFIT_SEED)
        ext_tr = sm.build_design_matrices(Xv)
        sm.fit(None, y_k, ext_rots=ext_tr)
        p = np.empty(n_nodes, dtype=np.float64)
        for s0 in range(0, n_nodes, SRF_BATCH):
            s1 = min(s0 + SRF_BATCH, n_nodes)
            E = sm.build_design_matrix(X_inf[s0:s1])
            p[s0:s1] = np.asarray(sm.predict(None, ext0=E), dtype=np.float64)
        out[f"SRF_{target}"] = p
        print(f"  SRF     deployed ({time.time()-t1:5.1f}s) "
              f"mean={p.mean():.2f} sd={p.std():.2f}")

    out_path = OUT / "block_predictions_all.csv"
    out.to_csv(out_path, index=False, float_format="%.4f")
    cfg_out = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "n_nodes": n_nodes,
        "training_table": str(cf.DATA_PATH), "n_train_classic": len(df_tr),
        "srf_vectors": str(st.X_path), "n_train_srf": len(Xv),
        "srf_variant": "local_aniso/k3/aug-off",
        "x_infer": str(X_INFER), "full_grid": str(FULL_GRID),
        "join_max_offset_m": float(d[ok].max()),
        "deploy_seed": DEPLOY_SEED, "srf_refit_seed": int(st.REFIT_SEED),
        "configs": cfgs,
    }
    (OUT / "block_deploy_config.json").write_text(
        json.dumps(cfg_out, indent=2), encoding="utf-8")
    print(f"\nDONE in {(time.time()-t0)/60:.1f} min -> {out_path} "
          f"({n_nodes} nodes x {len(out.columns)} cols)")


if __name__ == "__main__":
    main()
