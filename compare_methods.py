# -*- coding: utf-8 -*-
r"""
Like-for-like comparison of every method on the SAME spatial folds.

Supports differ. The tree baselines and GLS-RF are fitted on the classic point
table (551 samples); the SRF needs a k^3 voxel window around each sample and
build_voxels.py rejected the ones near the model boundary for insufficient
coverage (456 samples at k=3). Comparing a 551-row pooled metric against a
456-row pooled metric measures the difference in support, not in method. So two
tables are produced:

  FULL   : every method on its own native support (what each achieves given all
           the data it can use)
  COMMON : every method restricted to the rows the SRF can also predict — the
           only table in which the numbers are directly comparable

Joining the SRF to the classic table
------------------------------------
Voxel centres are stored as float32, whose ULP at these eastings is 0.25 m, so
a genuine pair sits 0.12-0.25 m apart. The NEXT block up is 2.0 m away in z and
carries a completely different grade. A tolerance loose enough to absorb the
rounding is therefore also loose enough to grab the wrong block: at the 2.5 m
tolerance this script used previously, 6 of 101 matches were the neighbouring
block, off by up to 33 DTR units. The tolerance is now 0.5 m, and the join is
rejected outright if any candidate lands in the ambiguous 0.5-1.0 m band.

Every join is additionally verified to be one-to-one, to agree on y_true, and
to agree on fold, before a single metric is computed.

Outputs (Comparison\):
  compare_full.csv      pooled metrics, native support
  compare_common.csv    pooled metrics, common support
  compare_paired.csv    paired Wilcoxon, every method pair, common support
  compare_perfold.csv   per-fold test R2, common support

Run:  python -u compare_methods.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import wilcoxon

# Every input and output hangs off one root. Defaults to the repository folder;
# set the SRF_PROJECT_ROOT environment variable to run against data elsewhere.
try:
    _HERE = Path(__file__).resolve().parent
except NameError:                       # notebook cell
    _HERE = Path.cwd()
RF = Path(os.environ.get("SRF_PROJECT_ROOT", _HERE))
OUT_DIR = RF / "Comparison"

# Runs to compare. All must sit on the same fold DESIGN; the classic-table runs
# must additionally share a roles fingerprint (checked below). Resolved to the
# newest matching run rather than hard-coded, because a stale path here is
# invisible: it produces a complete, plausible table computed on folds that no
# longer exist. Override on the command line to pin a specific run.
def _newest(root: Path, pattern: str, marker: str) -> Path:
    hits = sorted((p for p in root.glob(pattern)
                   if p.is_dir() and any(p.glob(marker))),
                  key=lambda p: p.stat().st_mtime)
    if not hits:
        raise FileNotFoundError(f"No run matching {pattern}/{marker} under {root}")
    return hits[-1]


TREES_RUN = _newest(RF / "Classic_runs", "TREES_*", "predictions_long_TREES.csv")
GLSRF_RUN = _newest(RF / "GLSRF_runs", "GLSRF_*", "predictions_long_GLSRF.csv")
# The reported SRF model is the anisotropic k3 variant, not whichever ran last.
SRF_RUN = _newest(RF / "SRF_runs", "SRF_run_*",
                  "03_test_evaluation/predictions_long_SRF_LOCAL_ANISO_k3.csv")

if len(sys.argv) > 3:
    TREES_RUN, GLSRF_RUN, SRF_RUN = (Path(a) for a in sys.argv[1:4])

MATCH_TOL_M = 0.5        # genuine pairs are at 0.12-0.25 m
REJECT_BAND_M = 1.0      # nothing may match between MATCH_TOL_M and this

TARGETS = ["DTR", "Magnetic"]
DISPLAY = {"RF": "Ordinary RF", "RF_XYZ": "RF + XYZ", "BAG": "Bagged trees",
           "GBM": "Gradient boosting", "XGB": "XGBoost",
           "GLSRF": "GLS-RF_3D", "SRF": "SRF_3D (local_aniso, k3)"}
# One representative per family: a plain RF, an RF with coordinates
# (the RF-Loc baseline), one boosting method, and the two spatially
# explicit learners. Bagged trees only ever isolated the RF-BAG
# feature-subsampling contrast, and GBM duplicates XGBoost.
ORDER = ["RF", "RF_XYZ", "XGB", "GLSRF", "SRF"]


def load_long():
    """One tidy frame: method, target, x, y, z, fold, y_true, y_pred."""
    frames = []

    t = pd.read_csv(TREES_RUN / "predictions_long_TREES.csv")
    if "selection" in t.columns:          # keep the cv-selected models only
        t = t[t.selection == "cv"]
    frames.append(t[["method", "target", "x", "y", "z", "fold", "y_true", "y_pred"]])
    print(f"trees : {TREES_RUN.name}  methods={sorted(t.method.unique())}  n={len(t)}")

    g = pd.read_csv(GLSRF_RUN / "predictions_long_GLSRF.csv")
    g["method"] = "GLSRF"
    frames.append(g[["method", "target", "x", "y", "z", "fold", "y_true", "y_pred"]])
    print(f"gls-rf: {GLSRF_RUN.name}  n={len(g)}")

    s = pd.read_csv(next((SRF_RUN / "03_test_evaluation").glob("predictions_long_*.csv")))
    s["target"] = s.target.map({"DTR": "DTR", "MAGNETIC": "Magnetic"})
    s["method"] = "SRF"
    frames.append(s[["method", "target", "x", "y", "z", "fold", "y_true", "y_pred"]])
    print(f"srf   : {SRF_RUN.name}  n={len(s)}")

    df = pd.concat(frames, ignore_index=True)
    df["target"] = df.target.str.strip()
    return df


def check_provenance():
    """The classic-table runs must have been fitted on identical rows+folds."""
    a = json.loads((TREES_RUN / "run_config.json").read_text(encoding="utf-8"))
    b = json.loads((GLSRF_RUN / "run_config.json").read_text(encoding="utf-8"))
    for key in ("coord_fingerprint_sha1", "roles_fingerprint_sha1", "n_rows"):
        if a.get(key) != b.get(key):
            raise RuntimeError(
                f"TREES and GLSRF disagree on {key}: {a.get(key)} vs {b.get(key)} — "
                "they were not fitted on the same data/folds.")
    print(f"provenance OK: both classic-table runs on {a['n_rows']} rows, "
          f"coords {a['coord_fingerprint_sha1']}, roles {a['roles_fingerprint_sha1']}")


def metrics(g):
    yt = g.y_true.values.astype(np.float64)
    yp = g.y_pred.values.astype(np.float64)
    ss_res = float(((yt - yp) ** 2).sum())
    ss_tot = float(((yt - yt.mean()) ** 2).sum())
    cov = float(((yt - yt.mean()) * (yp - yp.mean())).mean())
    return pd.Series({
        "n": len(g),
        "R2": 1.0 - ss_res / ss_tot,
        "RMSE": float(np.sqrt(ss_res / yt.size)),
        "MAE": float(np.abs(yt - yp).mean()),
        "CCC": float(2.0 * cov / (yt.var() + yp.var()
                                  + (yt.mean() - yp.mean()) ** 2)),
    })


def common_support(df):
    """Map every method's rows onto the SRF rows, proving the join each time."""
    keep = []
    for target in TARGETS:
        srf = df[(df.method == "SRF") & (df.target == target)].reset_index(drop=True)
        anchor = srf[["x", "y", "z"]].values
        for method in ORDER:
            m = df[(df.method == method) & (df.target == target)].reset_index(drop=True)
            if not len(m):
                continue
            d, i = cKDTree(m[["x", "y", "z"]].values).query(anchor)
            ok = d <= MATCH_TOL_M
            ambiguous = int(((d > MATCH_TOL_M) & (d < REJECT_BAND_M)).sum())
            if ambiguous:
                raise RuntimeError(
                    f"{method}/{target}: {ambiguous} anchors match between "
                    f"{MATCH_TOL_M} and {REJECT_BAND_M} m — the distance "
                    "populations are not cleanly separated; join not trustworthy.")
            idx = i[ok]
            if len(np.unique(idx)) != len(idx):
                raise RuntimeError(f"{method}/{target}: join is not one-to-one.")
            sel = m.iloc[idx].reset_index(drop=True)
            ref = srf[ok].reset_index(drop=True)
            dy = np.abs(sel.y_true.values - ref.y_true.values).max()
            if dy > 1e-3:
                raise RuntimeError(
                    f"{method}/{target}: matched rows disagree on y_true by "
                    f"{dy:.4g} — the join is wrong.")
            agree = float(np.mean(sel.fold.values == ref.fold.values))
            if agree < 1.0:
                raise RuntimeError(
                    f"{method}/{target}: fold agreement only {agree:.1%} — the "
                    "two fold files do not describe the same partition.")
            keep.append(sel)
            print(f"  {method:7s}/{target:9s} matched {int(ok.sum())}/{len(anchor)} "
                  f"| max offset {d[ok].max():.3f} m | folds agree 100%")
    return pd.concat(keep, ignore_index=True)


def table(df, label):
    t = (df.groupby(["method", "target"]).apply(metrics, include_groups=False)
         .reset_index())
    t["order"] = t.method.map({m: i for i, m in enumerate(ORDER)})
    t = t.sort_values(["target", "order"]).drop(columns="order")
    t.insert(0, "support", label)
    return t


def paired_all(df):
    """Wilcoxon on per-sample squared errors, every method pair."""
    out = []
    for target in TARGETS:
        err = {}
        for method in ORDER:
            m = df[(df.method == method) & (df.target == target)]
            if len(m):
                m = m.sort_values(["x", "y", "z"])
                err[method] = (m.y_true.values - m.y_pred.values) ** 2
        present = [m for m in ORDER if m in err]
        for i, a in enumerate(present):
            for b in present[i + 1:]:
                ea, eb = err[a], err[b]
                if len(ea) != len(eb):
                    raise RuntimeError(f"{a} vs {b}/{target}: length mismatch.")
                stat, p = wilcoxon(ea, eb)
                out.append({"target": target, "method_a": a, "method_b": b,
                            "mse_a": ea.mean(), "mse_b": eb.mean(),
                            "wilcoxon_p": p,
                            "verdict": (f"{a} better" if p < 0.05 and ea.mean() < eb.mean()
                                        else f"{b} better" if p < 0.05 else "tie")})
    return pd.DataFrame(out)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    check_provenance()
    df = load_long()
    # Restrict to the reported benchmark. common_support() already filters by
    # ORDER; the native table did not, so a dropped method silently survived
    # in one table and vanished from the other.
    df = df[df.method.isin(ORDER)].reset_index(drop=True)
    print()

    full = table(df, "native")
    print("verifying the common-support join:")
    com_df = common_support(df)
    com = table(com_df, "common")
    paired = paired_all(com_df)
    perfold = (com_df.groupby(["method", "target", "fold"])
               .apply(metrics, include_groups=False).reset_index())

    full.to_csv(OUT_DIR / "compare_full.csv", index=False)
    com.to_csv(OUT_DIR / "compare_common.csv", index=False)
    # The per-sample common-support frame, so downstream analyses reuse THIS
    # join rather than reimplementing it. A second implementation of a join
    # this safety-critical is a second chance to get it wrong.
    com_df.to_csv(OUT_DIR / "compare_common_long.csv", index=False)
    paired.to_csv(OUT_DIR / "compare_paired.csv", index=False)
    perfold.to_csv(OUT_DIR / "compare_perfold.csv", index=False)

    fmt = lambda v: f"{v:.3f}"
    for label, t in (("NATIVE SUPPORT (each method on all rows it can use)", full),
                     ("COMMON SUPPORT (the comparable table)", com)):
        print("\n" + "=" * 78)
        print(label)
        print("=" * 78)
        for target in TARGETS:
            s = t[t.target == target].copy()
            s["method"] = s.method.map(DISPLAY)
            print(f"\n  --- {target} ---")
            print(s[["method", "n", "R2", "RMSE", "MAE", "CCC"]]
                  .sort_values("R2", ascending=False)
                  .to_string(index=False, float_format=fmt))

    print("\n" + "=" * 78)
    print("PAIRED WILCOXON on per-sample squared errors (common support)")
    print("=" * 78)
    p = paired.copy()
    p["method_a"] = p.method_a.map(DISPLAY)
    p["method_b"] = p.method_b.map(DISPLAY)
    print(p.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n" + "=" * 78)
    print("PER-FOLD TEST R2 (common support)")
    print("=" * 78)
    for target in TARGETS:
        piv = (perfold[perfold.target == target]
               .pivot(index="method", columns="fold", values="R2")
               .reindex([m for m in ORDER if m in perfold.method.values]))
        print(f"\n  --- {target} ---")
        print(piv.to_string(float_format=fmt))

    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
