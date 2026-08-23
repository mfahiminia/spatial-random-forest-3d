# -*- coding: utf-8 -*-
r"""
SRF design-choice ablation: the 2x2 of kernel size x pattern variant.

    kernel size   k3 (3x3x3, 27 voxels)   vs   k5 (5x5x5, 125 voxels)
    pattern       local (augmentation + anisotropy summaries)
                  vs local_plain (same X, neither)

Six runs of srf_train.py on the same spatial folds, with the same grid and the
same nested sub-fold tuning; only the design factors change. Augmentation and
kernel size alter the design matrix itself, so they cannot be grid columns
inside one run the way the GLS-RF design choices are — each arm is a full run.

Every arm is declared BEFORE the runs — nothing is selected after seeing the
scores — and all of them are reported whatever they show.

Outputs (<root>\SRF_Ablation\):
  srf_ablation_runs.csv      one row per arm x target: pooled TEST + mean OOB
  srf_ablation_folds.csv     one row per arm x target x fold: OOB vs TEST
  srf_ablation_cells.csv     the design cells themselves (compare these)
  srf_ablation_effects.csv   each factor's effect, paired by fold, WITHIN each
                             level of the other factors

Run:  python srf_ablation_run.py
      --smoke   check the plumbing first with a cut-down grid
      --force   re-run arms already recorded in the manifest
Time: roughly 20 min per k3 arm, 60-90 min per k5 arm on 16 cores.
"""

import json
import sys
import time
from pathlib import Path

try:
    _HERE = Path(__file__).resolve().parent
except NameError:                       # notebook cell
    _HERE = Path.cwd()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import numpy as np
import pandas as pd

import srf_train as _st        # shares this run's PROJECT_ROOT resolution


# ============================================================
# CONFIG — the arms, declared up front
# ============================================================

ARMS = [
    # augmentation x anisotropy, full 2x2 at the reported kernel size
    {"arm": "local_k3",        "pattern": "local",       "kernel": "k3"},
    {"arm": "local_aug_k3",    "pattern": "local_aug",   "kernel": "k3"},
    {"arm": "local_aniso_k3",  "pattern": "local_aniso", "kernel": "k3"},
    {"arm": "local_plain_k3",  "pattern": "local_plain", "kernel": "k3"},
    # kernel size, at both corners of the pattern factor
    {"arm": "local_k5",        "pattern": "local",       "kernel": "k5"},
    {"arm": "local_plain_k5",  "pattern": "local_plain", "kernel": "k5"},
]

OUT_DIR = _st.PROJECT_ROOT / "SRF_Ablation"
MANIFEST = OUT_DIR / "srf_ablation_manifest.json"

# Which factor each arm carries, so effects can be reported WITHIN each level of
# the other factor. A marginal average over non-exclusive subsets is not a
# comparison (the "k3" and "aug+aniso" averages share an arm), so it is not
# produced here — cells and paired within-level effects only.
AUG = {"local": True, "local_aug": True, "local_aniso": False,
       "local_plain": False}
ANI = {"local": True, "local_aug": False, "local_aniso": True,
       "local_plain": False}


# ============================================================

def run_arm(arm: dict, smoke: bool) -> Path:
    import srf_train as st
    st.set_variant(pattern=arm["pattern"], kernel=arm["kernel"])
    st.SPLIT_MODE = "folds"
    st.TASK = "regression"
    st.IMPORTANCE_ENABLE = False
    if smoke:
        st.PARAM_GRID = {k: v[:2] if k in ("n_estimators", "max_depth") else v[:1]
                         for k, v in st.PARAM_GRID.items()}
        st.PARAM_GRID["n_estimators"] = [40]
    print("\n" + "#" * 70)
    print(f"# ARM {arm['arm']}  |  pattern={st.PATTERN_VARIANT} "
          f"kernel={st.KERNEL_VARIANT} method={st.METHOD_NAME}")
    print(f"# X = {st.X_path.name}   roles = {st.CV_ROLES_PATH}")
    print("#" * 70)
    return st.run_training()


def collect(arm: dict, run_dir: Path):
    """Pull the per-fold OOB/TEST table and the pooled summary out of a run."""
    folds = pd.read_csv(run_dir / "03_test_evaluation" / "test_metrics.csv")
    summ = pd.read_csv(run_dir / "summary.csv")

    keep = ["TARGET", "fold", "n_train", "n_test", "BEST_SUBCV_R2",
            "OOB_R2", "OOB_RMSE", "OOB_MAE",
            "TEST_R2", "TEST_RMSE", "TEST_MAE",
            "n_estimators", "max_depth", "min_leaf", "max_features",
            "max_samples"]
    fold_rows = folds[[c for c in keep if c in folds.columns]].copy()
    for k, v in (("arm", arm["arm"]), ("pattern", arm["pattern"]),
                 ("kernel", arm["kernel"])):
        fold_rows.insert(0, k, v)

    run_rows = []
    for _, r in summ.iterrows():
        run_rows.append({
            "arm": arm["arm"], "pattern": arm["pattern"],
            "kernel": arm["kernel"], "target": str(r["TARGET"]).upper(),
            "OOB_R2_mean": r.get("OOB_R2_mean", np.nan),
            "OOB_RMSE_mean": r.get("OOB_RMSE_mean", np.nan),
            "OOB_MAE_mean": r.get("OOB_MAE_mean", np.nan),
            "TEST_R2_mean": r.get("TEST_R2_mean", np.nan),
            "POOLED_R2": r.get("POOLED_R2", np.nan),
            "POOLED_RMSE": r.get("POOLED_RMSE", np.nan),
            "POOLED_MAE": r.get("POOLED_MAE", np.nan),
            "POOLED_CCC": r.get("POOLED_CCC", np.nan),
            "run_dir": str(run_dir),
        })
    return fold_rows, pd.DataFrame(run_rows)


def cell_table(runs: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    """One row per arm per target: the actual cells of the design, not averages."""
    out = []
    for _, r in runs.iterrows():
        f = folds[(folds.arm == r.arm) & (folds.TARGET.str.upper() == r.target)]
        out.append({
            "arm": r.arm, "kernel": r.kernel, "pattern": r.pattern,
            "augmentation": AUG[r.pattern], "anisotropy": ANI[r.pattern],
            "target": r.target, "n": int(f.n_train.iloc[0] + f.n_test.iloc[0]),
            "SUBCV_R2": f["BEST_SUBCV_R2"].mean(),
            "OOB_R2": f["OOB_R2"].mean(), "OOB_RMSE": f["OOB_RMSE"].mean(),
            "OOB_MAE": f["OOB_MAE"].mean(),
            "POOLED_R2": r.POOLED_R2, "POOLED_RMSE": r.POOLED_RMSE,
            "POOLED_MAE": r.POOLED_MAE, "POOLED_CCC": r.POOLED_CCC,
        })
    return pd.DataFrame(out)


def effect_table(folds: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    """Effect of each factor WITHIN each level of the others, paired by fold.

    This is what replaces the marginal table: 'augmentation on vs off, with
    anisotropy held at X, kernel held at Y' is a real comparison; an average
    over arms that differ in more than one factor is not.
    """
    out = []
    a2p = {a: p for a, p in zip(cells.arm, cells.pattern)}
    for arm_a, arm_b, factor, held in [
        ("local_aug_k3", "local_plain_k3", "augmentation", "aniso OFF, k=3"),
        ("local_k3", "local_aniso_k3", "augmentation", "aniso ON,  k=3"),
        ("local_aniso_k3", "local_plain_k3", "anisotropy", "aug OFF,   k=3"),
        ("local_k3", "local_aug_k3", "anisotropy", "aug ON,    k=3"),
        ("local_plain_k3", "local_plain_k5", "kernel k3-k5", "aug OFF, aniso OFF"),
        ("local_k3", "local_k5", "kernel k3-k5", "aug ON,  aniso ON"),
    ]:
        if arm_a not in a2p or arm_b not in a2p:
            continue
        for target in sorted(folds.TARGET.str.upper().unique()):
            A = folds[(folds.arm == arm_a) &
                      (folds.TARGET.str.upper() == target)].set_index("fold")
            B = folds[(folds.arm == arm_b) &
                      (folds.TARGET.str.upper() == target)].set_index("fold")
            idx = A.index.intersection(B.index)
            if not len(idx):
                continue
            row = {"factor": factor, "held at": held, "target": target,
                   "comparison": f"{arm_a} - {arm_b}", "n_folds": len(idx)}
            for m in ("BEST_SUBCV_R2", "OOB_R2", "TEST_R2"):
                d = A.loc[idx, m] - B.loc[idx, m]
                row[f"d_{m}"] = d.mean()
                row[f"wins_{m}"] = int((d > 0).sum())
            out.append(row)
    return pd.DataFrame(out)


def main():
    smoke = "--smoke" in sys.argv
    force = "--force" in sys.argv          # re-run arms already in the manifest
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done = {}
    if MANIFEST.exists() and not force:
        done = json.loads(MANIFEST.read_text(encoding="utf-8"))
        done = {k: v for k, v in done.items()
                if (Path(v) / "summary.csv").exists()}

    todo = [a for a in ARMS if a["arm"] not in done]
    print("=" * 70)
    print(f"SRF DESIGN ABLATION | {len(ARMS)} arms"
          + ("  [SMOKE]" if smoke else ""))
    for a in ARMS:
        state = "reuse" if a["arm"] in done else "RUN"
        print(f"   [{state:5s}] {a['arm']:18s} pattern={a['pattern']:12s} "
              f"kernel={a['kernel']}")
    print("=" * 70)

    t0 = time.time()
    for i, arm in enumerate(todo, 1):
        done[arm["arm"]] = str(run_arm(arm, smoke))
        MANIFEST.write_text(json.dumps(done, indent=2), encoding="utf-8")
        print(f"[{arm['arm']}] done ({i}/{len(todo)}) | "
              f"elapsed {(time.time() - t0) / 60:.1f} min")

    fold_frames, run_frames = [], []
    for arm in ARMS:
        fr, rr = collect(arm, Path(done[arm["arm"]]))
        fold_frames.append(fr)
        run_frames.append(rr)
    folds = pd.concat(fold_frames, ignore_index=True)
    runs = pd.concat(run_frames, ignore_index=True)
    folds.to_csv(OUT_DIR / "srf_ablation_folds.csv", index=False)
    runs.to_csv(OUT_DIR / "srf_ablation_runs.csv", index=False)

    cells = cell_table(runs, folds)
    effects = effect_table(folds, cells)
    cells.to_csv(OUT_DIR / "srf_ablation_cells.csv", index=False)
    effects.to_csv(OUT_DIR / "srf_ablation_effects.csv", index=False)

    fmt = lambda v: f"{v:.4f}"
    print("\n" + "=" * 70)
    print("PER-ARM RESULTS (pooled TEST vs mean OOB-train)")
    print("=" * 70)
    print(runs[["arm", "target", "OOB_R2_mean", "OOB_RMSE_mean",
                "POOLED_R2", "POOLED_RMSE", "POOLED_MAE", "POOLED_CCC"]]
          .to_string(index=False, float_format=fmt))

    print("\n" + "=" * 70)
    print("DESIGN CELLS (the arms themselves - compare these, not averages)")
    print("=" * 70)
    for target in sorted(cells.target.unique()):
        print(f"\n  --- {target} ---")
        print(cells[cells.target == target]
              .drop(columns=["target", "arm"])
              .to_string(index=False, float_format=fmt))

    print("\n" + "=" * 70)
    print("FACTOR EFFECTS (paired by fold, WITHIN each level of the others)")
    print("=" * 70)
    for target in sorted(effects.target.unique()):
        print(f"\n  --- {target} ---")
        print(effects[effects.target == target]
              .drop(columns=["target", "comparison"])
              .to_string(index=False, float_format=fmt))

    print("\n" + "=" * 70)
    print("OOB-train vs honest TEST gap (mean over folds):")
    for target in sorted(runs.target.unique()):
        s = runs[runs.target == target]
        print(f"  {target:9s} OOB R2={s.OOB_R2_mean.mean():.3f}  ->  "
              f"pooled TEST R2={s.POOLED_R2.mean():.3f}   "
              f"(optimism {s.OOB_R2_mean.mean() - s.POOLED_R2.mean():+.3f})")
    print("=" * 70)
    for name in ("srf_ablation_runs.csv", "srf_ablation_folds.csv",
                 "srf_ablation_cells.csv", "srf_ablation_effects.csv"):
        print(f"Saved: {OUT_DIR / name}")
    print(f"Total time: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
