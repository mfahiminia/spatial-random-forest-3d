# -*- coding: utf-8 -*-
r"""
Distributional fidelity and selectivity of the spatial-CV test predictions.

Point-support accuracy says how close each prediction is; it says nothing about
whether the predicted POPULATION resembles the sampled one. Every conditional-
expectation estimator shrinks toward the local mean, so all of them lose
variance — the question is how much, and whether the loss is concentrated in
the tails that drive cut-off decisions.

Consumes Comparison\compare_common_long.csv, i.e. the SAME verified 456-sample
join used by the point-support comparison (one-to-one, 100% fold agreement), so
every method's distribution is computed on identical material.

Reported per method x target:
  smoothing ratio   var(pred)/var(obs)          1.0 = variance preserved
  conditional bias  slope of obs regressed on pred; 1.0 = unbiased (Krige)
  KS statistic      max ECDF gap (descriptor, not a test - see note below)
  tail quantiles    q50/q75/q90/q95/q99, observed vs predicted
  selectivity       proportion above cut-off, and mean grade above cut-off

The KS p-value is deliberately NOT reported. Smoothing guarantees the two
distributions differ, so the test rejects for every method at this n and
answers a question nobody asked; the statistic is kept as an effect size.

NOTE ON SUPPORT: these are cross-validated predictions at sample locations,
each from a model that never saw the surrounding block. That is not the same
quantity as smoothing in a deployed block model, where the fitted model has
seen all the data and the targets are cell averages. The two stages are
related but not comparable numerically.

Run:  python distributional_fidelity.py
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

# Every input and output hangs off one root. Defaults to the repository folder;
# set the SRF_PROJECT_ROOT environment variable to run against data elsewhere.
try:
    _HERE = Path(__file__).resolve().parent
except NameError:                       # notebook cell
    _HERE = Path.cwd()
RF = Path(os.environ.get("SRF_PROJECT_ROOT", _HERE))
SRC = RF / "Comparison" / "compare_common_long.csv"
OUT = RF / "Distributional"

TARGETS = ["DTR", "Magnetic"]
# One representative per family: a plain RF, an RF with coordinates (the
# RF-Loc baseline), one boosting method, and the two spatially explicit
# learners. Bagged trees existed only to isolate the feature-subsampling
# contrast against RF, and GBM duplicates XGBoost; both are dropped from the
# reported benchmark. They remain in compare_common_long.csv if needed.
ORDER = ["RF", "RF_XYZ", "XGB", "GLSRF", "SRF"]
DISPLAY = {"RF": "Ordinary RF", "RF_XYZ": "RF + XYZ", "BAG": "Bagged trees",
           "GBM": "Gradient boosting", "XGB": "XGBoost",
           "GLSRF": "GLS-RF_3D", "SRF": "SRF_3D"}
SPATIAL = {"GLSRF", "SRF"}          # drawn heavier: the methods under test

# Okabe-Ito, CVD-safe
COLOR = {"RF": "#E69F00", "RF_XYZ": "#56B4E9", "BAG": "#999999",
         "GBM": "#D55E00", "XGB": "#CC79A7", "GLSRF": "#0072B2",
         "SRF": "#009E73"}
OBS = "#000000"

QUANTILES = [0.50, 0.75, 0.90, 0.95, 0.99]
N_CUTOFFS = 40


def load():
    if not SRC.exists():
        raise FileNotFoundError(
            f"{SRC} not found — run compare_methods.py first; it writes the "
            "verified common-support join this analysis consumes.")
    d = pd.read_csv(SRC).dropna(subset=["y_true", "y_pred"])
    missing = set(ORDER) - set(d.method.unique())
    if missing:
        raise RuntimeError(f"methods absent from the join: {sorted(missing)}")
    return d


def metrics(yt, yp):
    """Distributional descriptors for one method x target."""
    slope, intercept = np.polyfit(yp, yt, 1)      # observed ~ estimated
    row = {
        "n": len(yt),
        "mean_obs": yt.mean(), "mean_pred": yp.mean(),
        "ME": yp.mean() - yt.mean(),
        "sd_obs": yt.std(ddof=1), "sd_pred": yp.std(ddof=1),
        "smoothing_ratio": yp.var(ddof=1) / yt.var(ddof=1),
        "cond_slope": float(slope), "cond_intercept": float(intercept),
        "KS_D": float(ks_2samp(yt, yp).statistic),
    }
    for q in QUANTILES:
        qo, qp = np.quantile(yt, q), np.quantile(yp, q)
        row[f"q{int(q*100)}_obs"] = qo
        row[f"q{int(q*100)}_pred"] = qp
        row[f"q{int(q*100)}_err"] = qp - qo
    upper = [q for q in QUANTILES if q >= 0.75]
    row["tail_MAE"] = float(np.mean(
        [abs(np.quantile(yp, q) - np.quantile(yt, q)) for q in upper]))
    return row


def selectivity(yt, yp, cuts):
    """Proportion above cut-off and mean grade above cut-off, obs vs pred."""
    rows = []
    for c in cuts:
        mo, mp = yt >= c, yp >= c
        rows.append({
            "cutoff": c,
            "prop_obs": mo.mean(), "prop_pred": mp.mean(),
            "grade_above_obs": yt[mo].mean() if mo.any() else np.nan,
            "grade_above_pred": yp[mp].mean() if mp.any() else np.nan,
        })
    return pd.DataFrame(rows)


def ecdf(v):
    s = np.sort(v)
    return s, np.arange(1, len(s) + 1) / len(s)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = load()
    print(f"source: {SRC.name}  |  {len(d)} rows  |  methods {ORDER}")

    met_rows, sel_frames = [], []
    for t in TARGETS:
        obs = d[(d.target == t) & (d.method == ORDER[0])]
        yt_ref = obs.y_true.values.astype(float)
        cuts = np.linspace(np.quantile(yt_ref, 0.02),
                           np.quantile(yt_ref, 0.98), N_CUTOFFS)
        for m in ORDER:
            s = d[(d.method == m) & (d.target == t)]
            yt = s.y_true.values.astype(float)
            yp = s.y_pred.values.astype(float)
            met_rows.append({"target": t, "method": m, **metrics(yt, yp)})
            sf = selectivity(yt, yp, cuts)
            sf.insert(0, "method", m)
            sf.insert(0, "target", t)
            sel_frames.append(sf)

    M = pd.DataFrame(met_rows)
    S = pd.concat(sel_frames, ignore_index=True)
    M.to_csv(OUT / "dist_metrics.csv", index=False)
    S.to_csv(OUT / "selectivity.csv", index=False)

    # ---- console summary ----
    for t in TARGETS:
        print("\n" + "=" * 78)
        print(f"{t} - distributional fidelity (common 456-sample support)")
        print("=" * 78)
        sub = M[M.target == t].set_index("method").reindex(ORDER)
        cols = ["smoothing_ratio", "cond_slope", "KS_D", "sd_obs", "sd_pred",
                "q90_err", "q95_err", "tail_MAE"]
        print(sub[cols].rename(index=DISPLAY)
              .to_string(float_format=lambda v: f"{v:.3f}"))

    # ---- figure ----
    fig, axes = plt.subplots(3, 2, figsize=(12.0, 13.0))
    for j, t in enumerate(TARGETS):
        sub = d[d.target == t]
        yt_ref = sub[sub.method == ORDER[0]].y_true.values.astype(float)

        # row 1: QQ
        ax = axes[0][j]
        qs = np.linspace(0.01, 0.99, 99)
        qo = np.quantile(yt_ref, qs)
        lim = [min(qo.min(), 0), qo.max() * 1.05]
        ax.plot(lim, lim, "--", color=OBS, lw=1.2, zorder=1, label="observed (1:1)")
        for m in ORDER:
            yp = sub[sub.method == m].y_pred.values.astype(float)
            ax.plot(qo, np.quantile(yp, qs), color=COLOR[m],
                    lw=2.4 if m in SPATIAL else 1.3, zorder=3 if m in SPATIAL else 2,
                    label=DISPLAY[m])
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel(f"observed {t} quantile")
        ax.set_ylabel("predicted quantile")
        ax.set_title(f"{t} — quantile–quantile", fontsize=11, loc="left")

        # row 2: ECDF
        ax = axes[1][j]
        x, y = ecdf(yt_ref)
        ax.step(x, y, where="post", color=OBS, lw=2.0, ls="--", label="observed")
        for m in ORDER:
            x, y = ecdf(sub[sub.method == m].y_pred.values.astype(float))
            ax.step(x, y, where="post", color=COLOR[m],
                    lw=2.2 if m in SPATIAL else 1.2)
        ax.set_xlabel(t); ax.set_ylabel("cumulative proportion")
        ax.set_title(f"{t} — empirical CDF", fontsize=11, loc="left")

        # row 3: selectivity
        ax = axes[2][j]
        ss = S[S.target == t]
        ref = ss[ss.method == ORDER[0]]
        ax.plot(ref.cutoff, ref.prop_obs * 100, "--", color=OBS, lw=2.0,
                label="observed")
        for m in ORDER:
            g = ss[ss.method == m]
            ax.plot(g.cutoff, g.prop_pred * 100, color=COLOR[m],
                    lw=2.2 if m in SPATIAL else 1.2)
        ax.set_xlabel(f"{t} cut-off"); ax.set_ylabel("% of samples above cut-off")
        ax.set_title(f"{t} — selectivity", fontsize=11, loc="left")

        for ax in axes[:, j]:
            ax.grid(color="#e3e3e0", lw=0.7)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)

    axes[0][0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle("Distributional fidelity of spatial-CV predictions "
                 "(common 456-sample support)\n"
                 "curves below the 1:1 line and to the left of the observed "
                 "ECDF indicate variance loss", fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.945])
    fig.savefig(OUT / "fig_distributional_fidelity.png", dpi=200,
                facecolor="white")
    plt.close(fig)

    print(f"\nWrote {OUT / 'dist_metrics.csv'}")
    print(f"Wrote {OUT / 'selectivity.csv'}")
    print(f"Wrote {OUT / 'fig_distributional_fidelity.png'}")


if __name__ == "__main__":
    main()
