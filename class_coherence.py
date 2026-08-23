# -*- coding: utf-8 -*-
r"""
Salt-and-pepper in the DOMAIN model, separated from class boundaries.

THE CRITERION
    For each block, take the 3 x 3 x 3 = 27-block neighbourhood and count how
    many of those 27 carry the SAME class as the centre block. Call this the
    support of the block.

        support < 5     the class is held by the centre block and almost
                        nothing around it -> salt-and-pepper speck
        support ~ 10-14 roughly half the neighbourhood agrees -> the block sits
                        on a boundary between two domains, which is ordinary
                        geology and not a defect
        support ~ 27    interior of a homogeneous domain

    This is the distinction that a "class differs from the neighbourhood mode"
    test cannot make: that test flags every boundary block as well as every
    speck, and boundary blocks vastly outnumber specks, so it measures surface
    area rather than speckle.

SUPPORT IS AN ABSOLUTE COUNT
    A neighbourhood need not be complete for the criterion to apply: a block
    whose class is shared by 3 of the 20 modelled blocks around it is as
    isolated as one sharing it with 3 of 27. Support is therefore counted as it
    stands, not rescaled by neighbourhood size.

    The one exception is the extreme edge of the envelope, where a block with
    only four modelled neighbours would score support < 5 by geometry alone. A
    floor of 8 modelled neighbours removes that case and retains 99.6% of the
    model. The interior-only subset (all 26 neighbours modelled, 54.6% of
    blocks) is reported as a robustness check.

CLASSES (absolute cut-offs, as used in the section plots)
    low  < 35        mid  35-65        high >= 65

Run:  python -u class_coherence.py
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
BLOCK_CSV = RF / "BlockModel" / "block_predictions_all.csv"
OUT = RF / "BlockModel" / "Coherence"

TARGETS = ["DTR", "Magnetic"]
ORDER = ["RF", "RF_XYZ", "XGB", "GLSRF", "SRF"]
DISPLAY = {"RF": "Ordinary RF", "RF_XYZ": "RF + XYZ", "XGB": "XGBoost",
           "GLSRF": "GLS-RF_3D", "SRF": "SRF_3D"}
COLOR = {"RF": "#E69F00", "RF_XYZ": "#56B4E9", "XGB": "#CC79A7",
         "GLSRF": "#0072B2", "SRF": "#009E73"}

EDGES = [35.0, 65.0]
CLASS_NAME = ["low (<35)", "mid (35-65)", "high (>=65)"]
N_CLASS = 3

SPECK_SUPPORT = 5          # support < 5  ->  speck
SUPPORT_SWEEP = [2, 3, 4, 5, 6, 8, 10, 13]
MIN_NB_PRIMARY = 8         # floor, so support < 5 is not forced by geometry
MIN_NB_ROBUST = 26         # interior blocks only
# support bands for the stacked figure
BANDS = [(1, 4, "speck (support < 5)", "#D55E00"),
         (5, 13, "boundary (5-13)", "#E8C36A"),
         (14, 27, "domain interior (>= 14)", "#4C9F70")]


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


def shift(a, dz, dy, dx, fill):
    out = np.full_like(a, fill)
    src = a[max(0, -dz):a.shape[0] - max(0, dz),
            max(0, -dy):a.shape[1] - max(0, dy),
            max(0, -dx):a.shape[2] - max(0, dx)]
    out[max(0, dz):a.shape[0] - max(0, -dz),
        max(0, dy):a.shape[1] - max(0, -dy),
        max(0, dx):a.shape[2] - max(0, -dx)] = src
    return out


def support_field(g):
    """Returns (cls, support, n_valid_neighbours, valid)."""
    valid = np.isfinite(g)
    cls = np.full(g.shape, -1, dtype=np.int8)
    cls[valid] = np.digitize(g[valid], EDGES, right=False)

    same = np.zeros(g.shape, dtype=np.int16)     # neighbours of the same class
    n_nb = np.zeros(g.shape, dtype=np.int16)     # modelled neighbours
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dz == dy == dx == 0:
                    continue
                s = shift(cls, dz, dy, dx, -1)
                n_nb += (s >= 0)
                same += (s == cls) & (s >= 0)
    support = same + 1                            # include the centre block
    return cls, support, n_nb, valid


def analyse(g):
    cls, support, n_nb, valid = support_field(g)
    res = {}
    for tag, minnb in (("", MIN_NB_PRIMARY), ("_r", MIN_NB_ROBUST)):
        m = valid & (n_nb >= minnb)
        n = int(m.sum())
        res[f"n_blocks{tag}"] = n
        res[f"speck_pct{tag}"] = 100 * ((support < SPECK_SUPPORT) & m).sum() / n
    m = valid & (n_nb >= MIN_NB_PRIMARY)
    for s in SUPPORT_SWEEP:
        res[f"pct_lt{s}"] = 100 * ((support < s) & m).sum() / int(m.sum())
    for c in range(N_CLASS):
        mc = m & (cls == c)
        n_c = int(mc.sum())
        res[f"n_class{c}"] = n_c
        if n_c:
            sup = support[mc]
            res[f"speck_class{c}"] = 100 * (sup < SPECK_SUPPORT).mean()
            res[f"medsup_class{c}"] = float(np.median(sup))
            for lo, hi, lab, _ in BANDS:
                res[f"band{lo}_class{c}"] = 100 * ((sup >= lo) &
                                                   (sup <= hi)).mean()
        else:
            res[f"speck_class{c}"] = np.nan
            res[f"medsup_class{c}"] = np.nan
            for lo, _, _, _ in BANDS:
                res[f"band{lo}_class{c}"] = np.nan
    return res


def main():
    blk = pd.read_csv(BLOCK_CSV)
    rows = []
    for t in TARGETS:
        for m in ORDER:
            rows.append({"target": t, "method": m, **analyse(to_grid(blk, f"{m}_{t}"))})
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "class_speckle_support.csv", index=False)

    for t in TARGETS:
        sub = D[D.target == t].set_index("method").reindex(ORDER)
        print("\n" + "=" * 80)
        print(f"{t}  -  SALT-AND-PEPPER BY 27-BLOCK SUPPORT  "
              f"(speck = support < {SPECK_SUPPORT})")
        print(f"blocks with >= {MIN_NB_PRIMARY} modelled neighbours: "
              f"n = {int(sub.n_blocks.iloc[0])}")
        print("=" * 80)
        print(f"{'method':12s} {'speck %':>8} {'speck % (interior)':>19} "
              f"{'low':>8} {'mid':>8} {'high':>8}   (speck % within class)")
        for m in ORDER:
            r = sub.loc[m]
            print(f"{DISPLAY[m]:12s} {r.speck_pct:>8.3f} {r.speck_pct_r:>19.3f} "
                  f"{r.speck_class0:>8.2f} {r.speck_class1:>8.2f} "
                  f"{r.speck_class2:>8.2f}")

        print(f"\n  median support by class (27 = fully homogeneous cube)")
        print(f"  {'method':12s} {'low':>6} {'mid':>6} {'high':>6} "
              f"{'n high':>8}")
        for m in ORDER:
            r = sub.loc[m]
            print(f"  {DISPLAY[m]:12s} {r.medsup_class0:>6.0f} "
                  f"{r.medsup_class1:>6.0f} {r.medsup_class2:>6.0f} "
                  f"{int(r.n_class2):>8}")

        print(f"\n  high class, share of blocks by support band")
        print(f"  {'method':12s} " + " ".join(f"{lab:>22}"
                                              for _, _, lab, _ in BANDS))
        for m in ORDER:
            r = sub.loc[m]
            print(f"  {DISPLAY[m]:12s} "
                  + " ".join(f"{r[f'band{lo}_class2']:>21.1f}%"
                             for lo, _, _, _ in BANDS))

        print(f"\n  sensitivity of the speck threshold (% of all blocks)")
        print(f"  {'method':12s} " + " ".join(f"{'<'+str(s):>7}"
                                              for s in SUPPORT_SWEEP))
        for m in ORDER:
            r = sub.loc[m]
            print(f"  {DISPLAY[m]:12s} "
                  + " ".join(f"{r[f'pct_lt{s}']:>7.3f}" for s in SUPPORT_SWEEP))

    # ---------------------------------------------------------------- figure
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.6))
    xs = np.arange(len(ORDER))
    for j, t in enumerate(TARGETS):
        sub = D[D.target == t].set_index("method").reindex(ORDER)

        ax = axes[0][j]
        v = sub.speck_class2.values
        # A field may place no block at all in the high class (it happens
        # on small or synthetic datasets), leaving v all-NaN. Fall back to
        # a unit axis rather than letting matplotlib reject a NaN limit.
        vmax = np.nanmax(v) if np.isfinite(v).any() else 1.0
        ax.bar(xs, v, color=[COLOR[m] for m in ORDER], width=0.62)
        for k, m in enumerate(ORDER):
            if not np.isfinite(v[k]):
                continue
            ax.text(xs[k], v[k] + 0.02 * vmax,
                    f"{v[k]:.2f}%\n{int(sub.loc[m].n_class2)} blk",
                    ha="center", va="bottom", fontsize=7.4, color="#333")
        ax.set_ylim(0, max(vmax, 1e-9) * 1.32)
        ax.set_ylabel("% of high-class blocks that are specks")
        ax.set_title(f"(a) {t} — speckle in the high domain (≥ 65)\n"
                     f"speck = fewer than {SPECK_SUPPORT} of the 27 blocks "
                     "share the class", fontsize=10.5, loc="left")

        ax = axes[1][j]
        bottom = np.zeros(len(ORDER))
        for lo, hi, lab, col in BANDS:
            vals = np.array([sub.loc[m][f"band{lo}_class2"] for m in ORDER])
            ax.bar(xs, vals, bottom=bottom, width=0.62, label=lab, color=col)
            bottom += vals
        ax.set_ylim(0, 100)
        ax.set_ylabel("% of high-class blocks")
        ax.set_title(f"(b) {t} — high-class blocks by support band\n"
                     "speck, boundary and interior separated",
                     fontsize=10.5, loc="left")
        if j == 0:
            ax.legend(fontsize=8, frameon=False, loc="lower left")

        for ax in axes[:, j]:
            ax.set_xticks(xs)
            ax.set_xticklabels([DISPLAY[m] for m in ORDER], rotation=18,
                               ha="right", fontsize=8.5)
            ax.grid(axis="y", color="#e6e6e3", lw=0.7)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)

    fig.suptitle("Salt-and-pepper separated from domain boundaries by "
                 "27-block support\ninterior blocks only; a speck holds its "
                 "class almost alone, a boundary block shares it with about "
                 "half its neighbourhood",
                 fontsize=11.5, x=0.008, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT / "fig_class_speckle.png", dpi=300, facecolor="white")
    plt.close(fig)
    print(f"\nWrote {OUT/'fig_class_speckle.png'} and "
          f"{OUT/'class_speckle_support.csv'}")


if __name__ == "__main__":
    main()
