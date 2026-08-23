# -*- coding: utf-8 -*-
r"""
Sensitivity of the domain-continuity result to its two real parameters.

WHY THIS EXISTS
    Section 3.7.1 reports components per 1000 assigned blocks and the share of
    the domain lying in components below one SMU (12 blocks). The "per 1000" is
    a unit, not a parameter: any other constant rescales every method
    identically. The parameters that can actually change the conclusion are

      1. MIN_SIZE  - the smallest component that still counts as a workable
                     domain. One SMU is the mining limit; a domain that has to
                     sustain a processing campaign plausibly needs to be larger,
                     so the threshold is swept from well below one SMU to two
                     orders of magnitude above it.
      2. CONNECTIVITY - 6 (face) vs 26 (face/edge/corner). 6 is conservative;
                     26 merges blocks touching only at a corner, which no
                     excavator can mine as a unit but which a domain boundary
                     might legitimately cross.

    If the ranking of the methods is stable across both, the result in 3.7.1 is
    a property of the fields. If it is not, 3.7.1 must report the dependence.

Run:  python -u coherence_sensitivity.py
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.stats import kendalltau

# Every input and output hangs off one root. Defaults to the repository folder;
# set the SRF_PROJECT_ROOT environment variable to run against data elsewhere.
try:
    _HERE = Path(__file__).resolve().parent
except NameError:                       # notebook cell
    _HERE = Path.cwd()
RF = Path(os.environ.get("SRF_PROJECT_ROOT", _HERE))
BLOCK_CSV = RF / "BlockModel" / "block_predictions_all.csv"
SAMPLES = RF / "data_classic.xlsx"
OUT = RF / "BlockModel" / "Coherence"

TARGETS = ["DTR", "Magnetic"]
ORDER = ["RF", "RF_XYZ", "XGB", "GLSRF", "SRF"]
DISPLAY = {"RF": "Ordinary RF", "RF_XYZ": "RF + XYZ", "XGB": "XGBoost",
           "GLSRF": "GLS-RF_3D", "SRF": "SRF_3D"}
COLOR = {"RF": "#E69F00", "RF_XYZ": "#56B4E9", "XGB": "#CC79A7",
         "GLSRF": "#0072B2", "SRF": "#009E73"}

BLOCK_M3 = 5.0 * 5.0 * 2.0                       # 50 m3 per block
MIN_SIZES = [2, 4, 8, 12, 24, 50, 100, 200, 400]  # blocks
CUTOFF_PCT = [50, 65, 90]
PRIMARY_PCT = 65
STRUCT = {6: ndimage.generate_binary_structure(3, 1),
          26: ndimage.generate_binary_structure(3, 3)}


def to_grid(df, col):
    xs = np.unique(np.round(df.x.values, 3))
    ys = np.unique(np.round(df.y.values, 3))
    zs = np.unique(np.round(df.z.values, 3))
    ix = np.searchsorted(xs, np.round(df.x.values, 3))
    iy = np.searchsorted(ys, np.round(df.y.values, 3))
    iz = np.searchsorted(zs, np.round(df.z.values, 3))
    g = np.full((len(zs), len(ys), len(xs)), np.nan)
    g[iz, iy, ix] = df[col].values
    return g


def component_sizes(g, cut, conn):
    mask = np.isfinite(g) & (g >= cut)
    if not mask.any():
        return np.array([], dtype=int), 0
    lab, n = ndimage.label(mask, structure=STRUCT[conn])
    return np.bincount(lab.ravel())[1:], int(mask.sum())


def main():
    blk = pd.read_csv(BLOCK_CSV)
    smp = pd.read_excel(SAMPLES)

    rows = []
    for t in TARGETS:
        obs = pd.to_numeric(smp[t], errors="coerce").dropna()
        cuts = {p: float(np.percentile(obs, p)) for p in CUTOFF_PCT}
        for conn in (6, 26):
            for m in ORDER:
                g = to_grid(blk, f"{m}_{t}")
                for p, cv in cuts.items():
                    sizes, n_above = component_sizes(g, cv, conn)
                    if n_above == 0:
                        continue
                    for ms in MIN_SIZES:
                        rows.append({
                            "target": t, "method": m, "conn": conn,
                            "cutoff_pct": p, "min_size": ms,
                            "n_above": n_above,
                            "n_components": len(sizes),
                            "n_viable": int((sizes >= ms).sum()),
                            "frac_below": float(sizes[sizes < ms].sum() / n_above),
                            "largest_frac": float(sizes.max() / n_above)})
    S = pd.DataFrame(rows)
    S.to_csv(OUT / "sensitivity_min_size.csv", index=False)

    # ---------------- console: threshold sweep at 6-connectivity ------------
    for t in TARGETS:
        for p in (PRIMARY_PCT, 90):
            sub = S[(S.target == t) & (S.conn == 6) & (S.cutoff_pct == p)]
            if sub.empty:
                continue
            print("\n" + "=" * 86)
            print(f"{t}  cut-off P{p}  -  % of the domain below the size "
                  f"threshold (6-connectivity)")
            print("=" * 86)
            hdr = "  ".join(f"{ms:>5d}" for ms in MIN_SIZES)
            print(f"{'threshold (blocks)':22s} {hdr}")
            print(f"{'  = volume (m3)':22s} "
                  + "  ".join(f"{int(ms*BLOCK_M3):>5d}" for ms in MIN_SIZES))
            print("-" * 86)
            piv = sub.pivot_table(index="method", columns="min_size",
                                  values="frac_below")
            for m in ORDER:
                if m not in piv.index:
                    print(f"{DISPLAY[m]:22s} {'domain empty':>5s}")
                    continue
                print(f"{DISPLAY[m]:22s} "
                      + "  ".join(f"{100*piv.loc[m, ms]:5.1f}"
                                  for ms in MIN_SIZES))

            # ranking stability across thresholds
            rank = piv.reindex([m for m in ORDER if m in piv.index]).rank(axis=0)
            base = rank[MIN_SIZES[0]]
            taus = [kendalltau(base, rank[ms]).statistic for ms in MIN_SIZES[1:]]
            print(f"\n  ranking (1 = most continuous), by threshold:")
            for m in ORDER:
                if m not in rank.index:
                    continue
                print(f"  {DISPLAY[m]:22s} "
                      + "  ".join(f"{int(rank.loc[m, ms]):5d}"
                                  for ms in MIN_SIZES))
            print(f"  Kendall tau vs the {MIN_SIZES[0]}-block threshold: "
                  + ", ".join(f"{v:.2f}" for v in taus))

            # number of workable domains
            pv = sub.pivot_table(index="method", columns="min_size",
                                 values="n_viable")
            print(f"\n  number of components at or above the threshold:")
            for m in ORDER:
                if m not in pv.index:
                    continue
                print(f"  {DISPLAY[m]:22s} "
                      + "  ".join(f"{int(pv.loc[m, ms]):5d}"
                                  for ms in MIN_SIZES))

    # ---------------- console: 6 vs 26 connectivity -------------------------
    print("\n" + "=" * 86)
    print(f"CONNECTIVITY - components and % below one SMU (12 blocks), "
          f"P{PRIMARY_PCT}")
    print("=" * 86)
    print(f"{'':22s} {'DTR 6':>8} {'DTR 26':>8} {'Mag 6':>8} {'Mag 26':>8}")
    for m in ORDER:
        cells = []
        for t in TARGETS:
            for conn in (6, 26):
                r = S[(S.target == t) & (S.conn == conn)
                      & (S.cutoff_pct == PRIMARY_PCT)
                      & (S.min_size == 12) & (S.method == m)]
                cells.append(f"{int(r.n_components.iloc[0])}/"
                             f"{100*r.frac_below.iloc[0]:.1f}%")
        print(f"{DISPLAY[m]:22s} " + " ".join(f"{c:>8}" for c in cells))
    print("  (components / % of domain below one SMU)")

    # ---------------- figure ------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.8))
    for i, t in enumerate(TARGETS):
        for j, p in enumerate((PRIMARY_PCT, 90)):
            ax = axes[i][j]
            sub = S[(S.target == t) & (S.conn == 6) & (S.cutoff_pct == p)]
            for m in ORDER:
                s = sub[sub.method == m].sort_values("min_size")
                if s.empty:
                    continue
                ax.plot(s.min_size * BLOCK_M3, 100 * s.frac_below, "o-",
                        color=COLOR[m], label=DISPLAY[m], lw=1.8, ms=4)
            ax.axvline(12 * BLOCK_M3, color="#888", ls="--", lw=1.0)
            ax.text(12 * BLOCK_M3 * 1.08, 2, "1 SMU", fontsize=7.5,
                    color="#666", rotation=90, va="bottom")
            ax.set_xscale("log")
            ax.set_xlabel("minimum workable domain volume (m³)")
            ax.set_ylabel("% of domain below threshold")
            ax.set_title(f"{t} — cut-off P{p}", fontsize=10.5, loc="left")
            ax.grid(color="#e6e6e3", lw=0.7)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            if i == 0 and j == 0:
                ax.legend(fontsize=8, frameon=False)
    fig.suptitle("Sensitivity of the domain-continuity result to the minimum "
                 "workable domain size\n6-connectivity; block = 50 m³",
                 fontsize=11.5, x=0.008, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT / "fig_sensitivity.png", dpi=300, facecolor="white")
    plt.close(fig)
    print(f"\nWrote {OUT/'fig_sensitivity.png'} and "
          f"{OUT/'sensitivity_min_size.csv'}")


if __name__ == "__main__":
    main()
