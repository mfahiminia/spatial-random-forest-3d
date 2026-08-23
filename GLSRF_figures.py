# -*- coding: utf-8 -*-
r"""Publication figure for the GLS-RF design comparison.

Fig 1 (glsrf_design_choices.png): paired dumbbell plots of the best sub-fold
CV R2 attained by each competing option, with the other design factor held
fixed. Dumbbells rather than bars, so a focused x-axis is legitimate: the
marks encode position, not length from zero.

  row 1  isotropic vs anisotropic Sigma   (kernel held fixed, both shown)
  row 2  exponential vs Matern-3/2 kernel (geometry held fixed, both shown)

Every paired comparison behind the reported Wilcoxon statistics appears as
one dumbbell, so the figure and the text cannot disagree.

Run:  python GLSRF_figures.py [run_folder]
"""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import wilcoxon

try:
    _HERE = Path(__file__).resolve().parent
except NameError:                       # notebook cell
    _HERE = Path.cwd()
PROJECT_ROOT = Path(os.environ.get("SRF_PROJECT_ROOT", _HERE))
RUNS_BASE = PROJECT_ROOT / "GLSRF_runs"
TARGETS = ["DTR", "Magnetic"]
FOLDS = [1, 2, 3, 4, 5]
METRIC = "SUBCV_R2"          # the selection metric of the reported model

# Okabe-Ito, CVD-safe; matches the project's other diagnostics figures.
SEL = "#0072B2"              # selected option (anisotropic / exponential)
ALT = "#E69F00"              # alternative
GREY = "#7F7F7F"


def latest_run() -> Path:
    runs = sorted(p for p in RUNS_BASE.glob("GLSRF_*") if p.is_dir()
                  and (p / f"grid_{TARGETS[0]}_fold1.csv").exists())
    if not runs:
        raise FileNotFoundError(f"No GLSRF run folders with grid tables in {RUNS_BASE}")
    return runs[-1]


def load(run_dir: Path):
    return {(t, f): pd.read_csv(run_dir / f"grid_{t}_fold{f}.csv")
            for t in TARGETS for f in FOLDS}


def best(grids, t, f, mode, kern):
    g = grids[(t, f)].dropna(subset=[METRIC])
    sl = g[(g.Mode == mode) & (g.Kernel == kern)]
    return np.nan if sl.empty else float(sl[METRIC].max())


def panel(ax, rows, title):
    """rows: list of (row_label, value_selected, value_alternative).
    The paired statistics go in the title, not in a floating box: an inset
    text box inevitably collides with a dumbbell once the data change."""
    ys = np.arange(len(rows))[::-1]
    d = []
    for y, (lab, v_sel, v_alt) in zip(ys, rows):
        ax.plot([v_alt, v_sel], [y, y], "-", color=GREY, lw=1.4, zorder=1,
                solid_capstyle="round")
        ax.scatter([v_alt], [y], s=52, color=ALT, zorder=3,
                   edgecolor="white", linewidth=0.8)
        ax.scatter([v_sel], [y], s=52, color=SEL, zorder=3,
                   edgecolor="white", linewidth=0.8)
        d.append(v_sel - v_alt)
    d = np.array(d)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_ylim(-0.8, len(rows) - 0.2)
    ax.margins(x=0.10)
    p = wilcoxon(d).pvalue
    ax.set_title(f"{title}\nmean $\\Delta$ = {d.mean():+.3f}   "
                 f"{int((d > 0).sum())}/{len(d)} folds   p = {p:.3f}",
                 fontsize=9.5, pad=8)
    ax.grid(axis="x", color="#DDDDDD", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    # separate the two held-fixed blocks
    ax.axhline(len(rows) / 2 - 0.5, color="#BBBBBB", lw=0.8, ls=":", zorder=0)
    return d


def make_figure(run_dir: Path) -> Path:
    grids = load(run_dir)
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.8))

    # ---- row 1: geometry, kernel held fixed ----
    for j, t in enumerate(TARGETS):
        rows = []
        for kern, ktag in (("exp", "exp"), ("matern32", "Mat\u00e9rn")):
            for f in FOLDS:
                rows.append((f"fold {f} \u00b7 {ktag}",
                             best(grids, t, f, "ANISO", kern),
                             best(grids, t, f, "ISO", kern)))
        panel(axes[0, j], rows, f"{t} \u2014 covariance geometry")

    # ---- row 2: kernel, geometry held fixed ----
    for j, t in enumerate(TARGETS):
        rows = []
        for mode, mtag in (("ANISO", "aniso"), ("ISO", "iso")):
            for f in FOLDS:
                rows.append((f"fold {f} \u00b7 {mtag}",
                             best(grids, t, f, mode, "exp"),
                             best(grids, t, f, mode, "matern32")))
        panel(axes[1, j], rows, f"{t} \u2014 correlation function")

    for ax in axes[1, :]:
        ax.set_xlabel("best sub-fold CV $R^2$  (higher is better)", fontsize=9)

    def leg(ax, sel, alt):
        ax.legend(handles=[Line2D([], [], marker="o", ls="", color=SEL, ms=8,
                                  label=sel),
                           Line2D([], [], marker="o", ls="", color=ALT, ms=8,
                                  label=alt)],
                  loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8.5,
                  frameon=False, handletextpad=0.4)

    leg(axes[0, 1], "anisotropic", "isotropic")
    leg(axes[1, 1], "exponential", "Mat\u00e9rn-3/2")

    fig.suptitle("GLS-RF design choices \u2014 paired within fold, "
                 "other factor held fixed", fontsize=11.5, y=0.985)
    fig.tight_layout(rect=(0, 0, 0.89, 0.95))

    out = run_dir / "glsrf_design_choices.png"
    fig.savefig(out, dpi=300, facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    return out


if __name__ == "__main__":
    rd = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_run()
    print(f"run: {rd}")
    print(f"wrote: {make_figure(rd)}")
