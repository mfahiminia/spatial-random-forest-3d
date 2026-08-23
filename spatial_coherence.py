# -*- coding: utf-8 -*-
r"""
Local spatial coherence of the deployed block models.

THE PROBLEM THIS ADDRESSES
    A model can reproduce the histogram and the variogram and still be
    geologically implausible, because both are AVERAGES. The variogram is a
    two-point statistic: a few hundred incoherent blocks among 39,504 barely
    move it, and two fields with identical variograms can have completely
    different connectivity. What a mine planner sees instead is isolated
    blocks disagreeing with all their neighbours ("salt and pepper") and ore
    that breaks into specks too small to extract.

    The block model here is a regular 5 x 5 x 2 m lattice (58 x 43 x 114,
    13.9% filled), so 3D image analysis applies directly. All operations are
    mask-aware: only cells that carry a prediction take part.

WHAT IS MEASURED
  A. CONNECTIVITY AT CUT-OFF (scipy.ndimage.label, 6-connectivity = face
     adjacency, the conservative choice). Above a cut-off, report the number
     of connected components, the share of above-cut-off blocks in the largest
     one, the number of single-block components, and — the operational number
     — the share of above-cut-off blocks sitting in components smaller than a
     selective mining unit. That tonnage cannot be mined regardless of how
     good the global statistics look.

  B. SPIKE INDEX (median-filter residual). A median filter suppresses isolated
     spikes while preserving true edges, so r_i = v_i - median(26 neighbours)
     isolates the salt-and-pepper component. Standardised by the field sd,
     |z| > 2 and > 3 give spike rates directly comparable across methods.

  C. SPATIAL OUTLIERS (local Moran). A block is discordant when its own
     deviation from the mean and its neighbourhood's mean deviation have
     OPPOSITE signs and its own deviation is material (|z| > 1): a high block
     in a low neighbourhood or the reverse. This is the classic High-Low /
     Low-High LISA quadrant, counted rather than permutation-tested, because
     at 39k cells x 10 fields the permutation cost buys nothing the rate does
     not already show.

Run:  python spatial_coherence.py
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage

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

# Cut-offs as percentiles of the SAMPLED distribution, so they mean the same
# thing for both targets. The whole range is computed rather than a single
# value, so the reported result can be shown not to depend on where the domain
# boundary is drawn; PRIMARY_PCT is the one carried into the tables and figure.
CUTOFF_PCT = [50, 60, 65, 70, 75, 80, 90]
PRIMARY_PCT = 65
# 5 x 5 x 2 m blocks; an SMU of roughly 10 x 10 x 6 m is 2 x 2 x 3 = 12 blocks.
SMU_BLOCKS = 12
SPIKE_Z = [2.0, 3.0]


def to_grid(df, col):
    """Regular lattice -> 3D array with NaN outside the modelled envelope."""
    xs = np.unique(np.round(df.x.values, 3))
    ys = np.unique(np.round(df.y.values, 3))
    zs = np.unique(np.round(df.z.values, 3))
    ix = np.searchsorted(xs, np.round(df.x.values, 3))
    iy = np.searchsorted(ys, np.round(df.y.values, 3))
    iz = np.searchsorted(zs, np.round(df.z.values, 3))
    g = np.full((len(zs), len(ys), len(xs)), np.nan)
    g[iz, iy, ix] = df[col].values
    return g


def neighbour_stack(g):
    """(26, nz, ny, nx) of the 26 face/edge/corner shifts, NaN-padded."""
    out = []
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dz == dy == dx == 0:
                    continue
                s = np.full_like(g, np.nan)
                src = g[max(0, -dz):g.shape[0] - max(0, dz),
                        max(0, -dy):g.shape[1] - max(0, dy),
                        max(0, -dx):g.shape[2] - max(0, dx)]
                s[max(0, dz):g.shape[0] - max(0, -dz),
                  max(0, dy):g.shape[1] - max(0, -dy),
                  max(0, dx):g.shape[2] - max(0, -dx)] = src
                out.append(s)
    return np.stack(out)


def connectivity(g, cut):
    """Components of the above-cut-off set, 6-connectivity."""
    mask = np.isfinite(g) & (g >= cut)
    n_above = int(mask.sum())
    if n_above == 0:
        return dict(n_above=0, n_components=0, n_viable=0, largest_frac=np.nan,
                    n_singletons=0, frac_in_small=np.nan, median_size=np.nan)
    lab, n = ndimage.label(mask)                 # default = face adjacency
    sizes = np.bincount(lab.ravel())[1:]
    # n_viable is the count of components large enough to work as a domain; it
    # is the quantity a planner reads, and unlike a normalised component rate it
    # needs no reference constant to interpret.
    return dict(n_above=n_above, n_components=int(n),
                n_viable=int((sizes >= SMU_BLOCKS).sum()),
                largest_frac=float(sizes.max() / n_above),
                n_singletons=int((sizes == 1).sum()),
                frac_in_small=float(sizes[sizes < SMU_BLOCKS].sum() / n_above),
                median_size=float(np.median(sizes)))


def spikes(g):
    """Median-filter residual and local-Moran discordance."""
    stack = neighbour_stack(g)
    with np.errstate(invalid="ignore"):
        med = np.nanmedian(stack, axis=0)
        nmean = np.nanmean(stack, axis=0)
    valid = np.isfinite(g) & np.isfinite(med)
    r = np.where(valid, g - med, np.nan)
    sd = np.nanstd(g)
    z = r / sd
    # Rates are over the MODELLED cells only. The lattice is 13.9% filled, so
    # a denominator of nz*ny*nx would dilute every rate by a factor of ~7.
    az = np.abs(z)
    fin = np.isfinite(az)
    out = {"roughness": float(az[fin].mean()), "sd": float(sd),
           "n_valid": int(fin.sum())}
    for t in SPIKE_Z:
        out[f"spike_rate_z{t:g}"] = float((az[fin] > t).mean() * 100)

    gm = np.nanmean(g)
    zi = (g - gm) / sd
    zn = (nmean - gm) / sd
    ok = np.isfinite(zi) & np.isfinite(zn)
    disc = ok & (np.sign(zi) != np.sign(zn)) & (np.abs(zi) > 1.0)
    out["outlier_rate"] = float(disc.sum() / ok.sum() * 100)
    return out, z


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    blk = pd.read_csv(BLOCK_CSV)
    smp = pd.read_excel(SAMPLES)

    conn_rows, spike_rows = [], []
    zfields = {}
    for t in TARGETS:
        obs = pd.to_numeric(smp[t], errors="coerce").dropna()
        cuts = {p: float(np.percentile(obs, p)) for p in CUTOFF_PCT}
        print(f"\n[{t}] cut-offs from the sampled distribution: "
              + ", ".join(f"P{p}={v:.1f}" for p, v in cuts.items()))
        for m in ORDER:
            g = to_grid(blk, f"{m}_{t}")
            s, z = spikes(g)
            spike_rows.append({"target": t, "method": m, **s})
            zfields[(t, m)] = z
            for p, cv in cuts.items():
                conn_rows.append({"target": t, "method": m, "cutoff_pct": p,
                                  "cutoff": cv, **connectivity(g, cv)})

    C = pd.DataFrame(conn_rows); C.to_csv(OUT / "connectivity.csv", index=False)
    S = pd.DataFrame(spike_rows); S.to_csv(OUT / "spikes.csv", index=False)

    for t in TARGETS:
        print("\n" + "=" * 78)
        print(f"{t} - LOCAL COHERENCE")
        print("=" * 78)
        print("A. connectivity of the above-cut-off set (6-connectivity)")
        for p in CUTOFF_PCT:
            sub = C[(C.target == t) & (C.cutoff_pct == p)].set_index("method")
            print(f"\n   cut-off P{p} = {sub.cutoff.iloc[0]:.1f}   "
                  f"(SMU = {SMU_BLOCKS} blocks)")
            print(f"   {'method':12s} {'n_above':>8} {'n_comp':>7} "
                  f"{'largest%':>9} {'singles':>8} {'%<SMU':>7}")
            for m in ORDER:
                r = sub.loc[m]
                print(f"   {DISPLAY[m]:12s} {int(r.n_above):>8} "
                      f"{int(r.n_components):>7} {100*r.largest_frac:>8.1f}% "
                      f"{int(r.n_singletons):>8} {100*r.frac_in_small:>6.2f}%")
        print("\nB. spike / outlier rates (whole field)")
        sub = S[S.target == t].set_index("method")
        print(f"   {'method':12s} {'roughness':>10} {'|z|>2 %':>9} "
              f"{'|z|>3 %':>9} {'outlier %':>10}")
        for m in ORDER:
            r = sub.loc[m]
            print(f"   {DISPLAY[m]:12s} {r.roughness:>10.4f} "
                  f"{r['spike_rate_z2']:>9.2f} {r['spike_rate_z3']:>9.2f} "
                  f"{r.outlier_rate:>10.2f}")

    # ---- figure ----
    fig, axes = plt.subplots(3, 2, figsize=(12.0, 12.0))
    xs = np.arange(len(ORDER))
    for j, t in enumerate(TARGETS):
        ax = axes[0][j]
        sub = C[(C.target == t) & (C.cutoff_pct == PRIMARY_PCT)].set_index("method")
        ax.bar(xs, [sub.loc[m].n_components for m in ORDER],
               color=[COLOR[m] for m in ORDER], width=0.6)
        ax.set_xticks(xs); ax.set_xticklabels([DISPLAY[m] for m in ORDER],
                                              rotation=20, ha="right", fontsize=8.5)
        ax.set_ylabel("number of connected components")
        ax.set_title(f"{t} — domains above P{PRIMARY_PCT} "
                     "(fewer = more coherent)", fontsize=11, loc="left")

        ax = axes[1][j]
        w = 0.38
        for k, p in enumerate([PRIMARY_PCT, 90]):
            sub = C[(C.target == t) & (C.cutoff_pct == p)].set_index("method")
            ax.bar(xs + (k - 0.5) * w,
                   [100 * sub.loc[m].frac_in_small for m in ORDER],
                   width=w, label=f"cut-off P{p}",
                   color=["#0072B2", "#D55E00"][k])
        ax.set_xticks(xs); ax.set_xticklabels([DISPLAY[m] for m in ORDER],
                                              rotation=20, ha="right", fontsize=8.5)
        ax.set_ylabel("% of ore in sub-SMU specks")
        ax.set_title(f"{t} — unmineable fragmented ore", fontsize=11, loc="left")
        if j == 0:
            ax.legend(fontsize=8, frameon=False)

        ax = axes[2][j]
        sub = S[S.target == t].set_index("method")
        for k, (col, lab) in enumerate([("spike_rate_z2", "|z| > 2"),
                                        ("outlier_rate", "spatial outlier")]):
            ax.bar(xs + (k - 0.5) * w, [sub.loc[m][col] for m in ORDER],
                   width=w, label=lab, color=["#0072B2", "#D55E00"][k])
        ax.set_xticks(xs); ax.set_xticklabels([DISPLAY[m] for m in ORDER],
                                              rotation=20, ha="right", fontsize=8.5)
        ax.set_ylabel("% of blocks")
        ax.set_title(f"{t} — salt-and-pepper rate", fontsize=11, loc="left")
        if j == 0:
            ax.legend(fontsize=8, frameon=False)

        for ax in axes[:, j]:
            ax.grid(axis="y", color="#e3e3e0", lw=0.7)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)

    fig.suptitle("Local spatial coherence of the deployed block models\n"
                 "what the variogram and histogram cannot see",
                 fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "fig_coherence.png", dpi=200, facecolor="white")
    plt.close(fig)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
