# -*- coding: utf-8 -*-
r"""
Paper figure + paper tables for the local-coherence diagnostics.

Consumes the outputs of spatial_coherence.py (connectivity.csv, spikes.csv) so
the numbers in the figure and in the manuscript tables cannot drift apart from
the numbers in the console log.

Two things are normalised here that the diagnostic script reports raw:
  - component count is divided by the number of above-cut-off blocks, because
    the methods do not select the same tonnage at a common cut-off and a raw
    count therefore penalises whichever method reports MORE ore;
  - the P90 panel carries the above-cut-off block count, because one method
    places a single block above P90 and a percentage computed on n = 1 must not
    be read as a rate.

Run:  python -u coherence_figure.py
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Every input and output hangs off one root. Defaults to the repository folder;
# set the SRF_PROJECT_ROOT environment variable to run against data elsewhere.
try:
    _HERE = Path(__file__).resolve().parent
except NameError:                       # notebook cell
    _HERE = Path.cwd()
RF = Path(os.environ.get("SRF_PROJECT_ROOT", _HERE))
OUT = RF / "BlockModel" / "Coherence"

TARGETS = ["DTR", "Magnetic"]
ORDER = ["RF", "RF_XYZ", "XGB", "GLSRF", "SRF"]
DISPLAY = {"RF": "Ordinary RF", "RF_XYZ": "RF + XYZ", "XGB": "XGBoost",
           "GLSRF": "GLS-RF_3D", "SRF": "SRF_3D"}
COLOR = {"RF": "#E69F00", "RF_XYZ": "#56B4E9", "XGB": "#CC79A7",
         "GLSRF": "#0072B2", "SRF": "#009E73"}
N_NODES = 39504
SMU_BLOCKS = 12
PRIMARY_PCT = 65          # domain boundary carried into the tables and figure
SWEEP_PCT = [50, 60, 65, 70, 75, 80]   # reported to show the result is not
                                       # an artefact of where the line is drawn


def style(ax):
    ax.grid(axis="y", color="#e6e6e3", lw=0.7)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def ticks(ax, xs):
    ax.set_xticks(xs)
    ax.set_xticklabels([DISPLAY[m] for m in ORDER], rotation=18,
                       ha="right", fontsize=8.5)


def main():
    C = pd.read_csv(OUT / "connectivity.csv")
    S = pd.read_csv(OUT / "spikes.csv")
    C["pct_of_nodes"] = 100.0 * C.n_above / N_NODES

    # ------------------------------------------------------------------ tables
    t75 = (C[C.cutoff_pct == PRIMARY_PCT]  # primary cut-off table
           .assign(largest_pct=lambda d: 100 * d.largest_frac,
                   small_pct=lambda d: 100 * d.frac_in_small)
           [["target", "method", "cutoff", "n_above", "pct_of_nodes",
             "n_components", "n_viable", "largest_pct", "n_singletons",
             "small_pct"]])
    t90 = (C[C.cutoff_pct == 90]
           .assign(largest_pct=lambda d: 100 * d.largest_frac,
                   small_pct=lambda d: 100 * d.frac_in_small)
           [["target", "method", "cutoff", "n_above", "pct_of_nodes",
             "n_components", "largest_pct", "n_singletons", "small_pct"]])
    t75.to_csv(OUT / f"table_connectivity_P{PRIMARY_PCT}.csv", index=False)
    t90.to_csv(OUT / "table_connectivity_P90.csv", index=False)
    S.to_csv(OUT / "table_texture.csv", index=False)

    for t in TARGETS:
        print("\n" + "=" * 74)
        print(f"{t}  -  TABLE A: continuity of the domain above P{PRIMARY_PCT}")
        print("=" * 74)
        sub = t75[t75.target == t].set_index("method").reindex(ORDER)
        print(f"{'method':12s} {'n>cut':>7} {'%nodes':>7} {'comp':>5} "
              f"{'>=SMU':>6} {'largest%':>9} {'single':>7} {'%<SMU':>7}")
        for m in ORDER:
            r = sub.loc[m]
            print(f"{DISPLAY[m]:12s} {int(r.n_above):>7} {r.pct_of_nodes:>7.1f} "
                  f"{int(r.n_components):>5} {int(r.n_viable):>6} "
                  f"{r.largest_pct:>8.1f}% {int(r.n_singletons):>7} "
                  f"{r.small_pct:>6.2f}%")

        print(f"\n{t}  -  TABLE B: the above-P90 set")
        sub = t90[t90.target == t].set_index("method").reindex(ORDER)
        print(f"{'method':12s} {'n>cut':>7} {'%nodes':>7} {'comp':>5} "
              f"{'largest%':>9} {'%<SMU':>7}")
        for m in ORDER:
            r = sub.loc[m]
            lp = "  n/a" if not np.isfinite(r.largest_pct) else f"{r.largest_pct:7.1f}%"
            sp = "  n/a" if not np.isfinite(r.small_pct) else f"{r.small_pct:5.1f}%"
            print(f"{DISPLAY[m]:12s} {int(r.n_above):>7} {r.pct_of_nodes:>7.2f} "
                  f"{int(r.n_components):>5} {lp:>9} {sp:>7}")

        print(f"\n{t}  -  TABLE C: local texture (whole field)")
        sub = S[S.target == t].set_index("method").reindex(ORDER)
        print(f"{'method':12s} {'roughness':>10} {'|z|>2 %':>9} {'|z|>3 %':>9} "
              f"{'LISA %':>8}")
        for m in ORDER:
            r = sub.loc[m]
            print(f"{DISPLAY[m]:12s} {r.roughness:>10.4f} "
                  f"{r['spike_rate_z2']:>9.4f} {r['spike_rate_z3']:>9.4f} "
                  f"{r.outlier_rate:>8.4f}")

    # ------------------------------------------------- cut-off sweep --------
    # The domain boundary is a convention. If the ranking moved with it, the
    # primary cut-off would be a choice rather than a reporting decision.
    for t in TARGETS:
        print("\n" + "=" * 74)
        print(f"{t}  -  CUT-OFF SWEEP: % of assigned volume below one SMU")
        print("=" * 74)
        sub = C[(C.target == t) & (C.cutoff_pct.isin(SWEEP_PCT))]
        piv = sub.pivot_table(index="method", columns="cutoff_pct",
                              values="frac_in_small").reindex(ORDER) * 100
        print(f"{'method':12s} " + " ".join(f"{'P'+str(p):>7}"
                                            for p in SWEEP_PCT))
        for m in ORDER:
            print(f"{DISPLAY[m]:12s} "
                  + " ".join(f"{piv.loc[m, p]:7.2f}" for p in SWEEP_PCT))
        rank = piv.rank(axis=0)
        print(f"\n{'rank (1=best)':12s} " + " ".join(f"{'P'+str(p):>7}"
                                                     for p in SWEEP_PCT))
        for m in ORDER:
            print(f"{DISPLAY[m]:12s} "
                  + " ".join(f"{int(rank.loc[m, p]):7d}" for p in SWEEP_PCT))

    # ----------------------------------------------------------------- figure
    # Two rows only: the local-texture panel moved to fig_class_speckle.png,
    # because speckle is a property of the CLASS assignment and the value-based
    # texture measures cannot resolve it (see class_coherence.py).
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.4))
    xs = np.arange(len(ORDER))

    for j, t in enumerate(TARGETS):
        # (a) how many pieces the domain breaks into, and how many are workable
        ax = axes[0][j]
        sub = C[(C.target == t) & (C.cutoff_pct == PRIMARY_PCT)].set_index("method")
        tot = np.array([sub.loc[m].n_components for m in ORDER], float)
        via = np.array([sub.loc[m].n_viable for m in ORDER], float)
        w = 0.38
        ax.bar(xs - 0.5 * w, tot, width=w, label="all components",
               color=[COLOR[m] for m in ORDER])
        ax.bar(xs + 0.5 * w, via, width=w, label=f"components ≥ 1 SMU",
               color=[COLOR[m] for m in ORDER], alpha=0.45,
               edgecolor="#444", lw=0.7)
        for k, m in enumerate(ORDER):
            ax.text(xs[k] - 0.5 * w, tot[k] + 0.02 * tot.max(),
                    f"{int(tot[k])}", ha="center", va="bottom", fontsize=7.6,
                    color="#333")
            ax.text(xs[k] + 0.5 * w, via[k] + 0.02 * tot.max(),
                    f"{int(via[k])}", ha="center", va="bottom", fontsize=7.6,
                    color="#333")
        ax.set_ylim(0, tot.max() * 1.22)
        ticks(ax, xs)
        ax.set_ylabel("number of components")
        ax.set_title(f"(a) {t} — how the domain above P{PRIMARY_PCT} breaks up\n"
                     "solid = every piece; pale = pieces workable as a domain",
                     fontsize=10.5, loc="left")
        if j == 0:
            ax.legend(fontsize=8, frameon=False, loc="upper left")

        # (b) volume stranded in sub-SMU fragments, at the primary cut-off only.
        # P90 is deliberately not drawn: with 42-317 blocks above that cut-off
        # the same statistic is unstable, and a bar chart would invite the
        # ranking the text declines to make.
        ax = axes[1][j]
        sb = C[(C.target == t) & (C.cutoff_pct == PRIMARY_PCT)].set_index("method")
        vals = np.array([100 * sb.loc[m].frac_in_small for m in ORDER])
        ax.bar(xs, vals, color=[COLOR[m] for m in ORDER], width=0.62)
        for k, m in enumerate(ORDER):
            ax.text(xs[k], vals[k] + 0.02 * vals.max(),
                    f"{vals[k]:.2f}%\nn={int(sb.loc[m].n_above)}",
                    ha="center", va="bottom", fontsize=7.4, color="#333")
        ax.set_ylim(0, vals.max() * 1.32)
        ticks(ax, xs)
        ax.set_ylabel("% of assigned volume in fragments < SMU")
        ax.set_title(f"(b) {t} — domain volume that cannot be mined or routed\n"
                     f"cut-off P{PRIMARY_PCT}; SMU = {SMU_BLOCKS} blocks "
                     "(10 x 10 x 6 m); n = blocks assigned",
                     fontsize=10.5, loc="left")

        for ax in axes[:, j]:
            style(ax)

    fig.suptitle("Continuity of the geometallurgical domains delineated by the "
                 f"deployed block models ({N_NODES:,} nodes, 5 x 5 x 2 m)\n"
                 "properties that the histogram and the variogram, being "
                 "averages, cannot resolve",
                 fontsize=11.5, x=0.008, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.945])
    p = OUT / "fig_coherence_paper.png"
    fig.savefig(p, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"\nWrote {p}")
    print(f"Wrote {OUT/f'table_connectivity_P{PRIMARY_PCT}.csv'}, "
          f"{OUT/'table_connectivity_P90.csv'}, {OUT/'table_texture.csv'}")


if __name__ == "__main__":
    main()
