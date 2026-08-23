# -*- coding: utf-8 -*-
r"""
Synthetic demo dataset — makes this repository runnable without the case-study data.

The deposit data behind the paper is not redistributable, so this script
fabricates a stand-in with the SAME FILE SCHEMA, the same grid geometry and the
same qualitative spatial structure. Every downstream script then runs unchanged:

    python make_demo_data.py
    python build_voxels.py
    python make_folds.py
    python srf_train.py

*** THE NUMBERS THIS PRODUCES ARE NOT THE PAPER'S NUMBERS. ***
This is a simulated orebody. Use it to check that the pipeline runs, to read the
code against a concrete example, or as a template for substituting your own data
(see DATA.md for the exact schema). Do not cite results obtained from it.

What is simulated, and why
--------------------------
The point of the demo is that the parts of the method that matter are actually
exercised, not that the geology is realistic:

* Anisotropic continuity. Covariates are drawn as Gaussian random fields with
  an ellipsoidal correlation structure — major/semi/minor ranges 120/50/25 m at
  azimuth 110 deg, dip 25 deg, the geometry the SRF's anisotropy features assume.
  An isotropic demo would leave that whole feature block untested.
* Depth zonation. Grade is depth-dependent, with a low-grade cap over the top of
  the model. This is what makes spatial block CV hard and random CV easy in the
  real deposit, and it reproduces here: expect one weak fold.
* Sampling along drillhole traces. Responses exist only on vertical runs of
  nodes at scattered collars, not everywhere — so the fold builder sees the same
  kind of clustered, unevenly covered support it sees in practice.
* A mineralised envelope. Covariates exist only inside an irregular body
  covering about 15 % of the bounding box, as in a real estimated block model.
  This is what keeps the inference array a sane size and what makes boundary
  windows fail the coverage rule.
* Missing covariates. Inside the envelope a few per cent of nodes are voided at
  random, exercising the coverage rule and the within-window IDW imputation.
* An unobservable component. The responses depend on random fields that appear
  in no covariate, so they cannot be predicted perfectly from the voxel window.
  Without it the demo scores R^2 ~ 0.9 and says nothing about the method.

One consequence to be aware of when reading demo results: the low-grade cap here
is an explicit, smooth function of depth, so the z coordinate alone carries a lot
of signal. RF+XYZ therefore beats plain RF by a wide margin on the demo, whereas
on the case-study data the two are statistically tied. Do not read the demo as
evidence about coordinates-as-features either way.

What it does NOT reproduce: the calibration between the CV test-to-train distance
and the deployment distance. In the case study those two distributions were
matched by choosing the block layout (median 16.0 m against 16.2 m); here the
simulated drilling pattern is denser than that, so make_folds.py prints a CV
median about half the deployment median. On your own data, check that print-out
and adjust N_DIV — see DATA.md.

Outputs
-------
  Vector/grade_full_D3.csv   the estimated grid table (input to build_voxels.py)
  data_classic.xlsx          the point-support sample table (input to make_folds.py)

Run:  python make_demo_data.py [--small]
      --small  a 4x smaller grid; the whole pipeline then runs in well under a
               minute, at the cost of fewer samples per fold.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import fftconvolve

try:
    _HERE = Path(__file__).resolve().parent
except NameError:                       # notebook cell
    _HERE = Path.cwd()

PROJECT_ROOT = Path(os.environ.get("SRF_PROJECT_ROOT", _HERE))

# ---------------- geometry (mirrors the case study) ----------------
GRID = (60, 47, 117)            # nodes (Nx, Ny, Nz)
GRID_SMALL = (34, 26, 60)
CELL_SIZE = (5.0, 5.0, 2.0)     # metres
ORIGIN = (0.0, 0.0, 0.0)        # arbitrary local frame, not a real location

# continuity ellipsoid — the same frame srf_train.py assumes
A_MAJOR, A_SEMI, A_MINOR = 120.0, 50.0, 25.0
AZIMUTH_DEG, DIP_DEG = 110.0, 25.0

N_COLLARS = 52                  # drillhole collars carrying responses
HOLE_LEN_NODES = (8, 18)        # samples per hole, drawn uniformly in this range
MIN_COLLAR_SPACING_NODES = 3    # Chebyshev spacing between collars, in nodes
ENVELOPE_FRACTION = 0.15        # share of the box that carries covariates
FRAC_MISSING = 0.04             # of those, voided at random

SEED = 20260814


# ============================================================
# Anisotropic Gaussian random field
# ============================================================

def rotation_matrix(azimuth_deg, dip_deg):
    """Rows = major, semi, minor axes. Same convention as srf_core_final."""
    az, dip = np.deg2rad(azimuth_deg), np.deg2rad(dip_deg)
    v1 = np.array([np.sin(az) * np.cos(dip),
                   np.cos(az) * np.cos(dip),
                   -np.sin(dip)])
    v1 /= np.linalg.norm(v1)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(ref @ v1) > 0.99:
        ref = np.array([0.0, 1.0, 0.0])
    v2 = np.cross(ref, v1); v2 /= np.linalg.norm(v2)
    v3 = np.cross(v1, v2);  v3 /= np.linalg.norm(v3)
    return np.vstack([v1, v2, v3])


def _kernel(cell, ranges, azimuth_deg, dip_deg):
    """Anisotropic Gaussian smoothing kernel, trimmed to its support.

    Convolving white noise with this gives a field whose covariance is the
    kernel's autocorrelation, so the kernel is built at range/sqrt(2) to land on
    the requested practical ranges.
    """
    sig = np.array(ranges, dtype=float) / np.sqrt(2.0)
    half = [int(np.ceil(max(ranges) / c)) for c in cell]
    axes = [np.arange(-h, h + 1) * c for h, c in zip(half, cell)]
    hx, hy, hz = np.meshgrid(*axes, indexing="ij")
    h = np.stack([hx, hy, hz], axis=-1).reshape(-1, 3).T

    R = rotation_matrix(azimuth_deg, dip_deg)
    u = (np.diag(1.0 / sig) @ (R @ h)).T
    k = np.exp(-0.5 * np.sum(u * u, axis=1)).reshape(hx.shape)

    # trim slices that contribute nothing, so the FFT stays small
    keep = [np.where(k.max(axis=tuple(a for a in range(3) if a != ax)) > 1e-4)[0]
            for ax in range(3)]
    k = k[np.ix_(keep[0], keep[1], keep[2])]
    return k / np.sqrt((k ** 2).sum())


def gaussian_field(shape, cell, ranges, azimuth_deg, dip_deg, rng):
    """Unit-variance, zero-mean anisotropic Gaussian random field."""
    k = _kernel(cell, ranges, azimuth_deg, dip_deg)
    pad = [s // 2 for s in k.shape]
    noise = rng.standard_normal([s + 2 * p for s, p in zip(shape, pad)])
    f = fftconvolve(noise, k, mode="same")
    f = f[pad[0]:pad[0] + shape[0],
          pad[1]:pad[1] + shape[1],
          pad[2]:pad[2] + shape[2]]
    return (f - f.mean()) / f.std()


# ============================================================
# Synthetic deposit
# ============================================================

def build(shape, rng):
    nx, ny, nz = shape
    dx, dy, dz = CELL_SIZE

    geom = dict(cell=CELL_SIZE, ranges=(A_MAJOR, A_SEMI, A_MINOR),
                azimuth_deg=AZIMUTH_DEG, dip_deg=DIP_DEG)
    print("simulating random fields ...")
    z1 = gaussian_field(shape, rng=rng, **geom)      # the main grade field
    z2 = gaussian_field(shape, rng=rng, **geom)      # a second, independent one
    z3 = gaussian_field(shape, cell=CELL_SIZE, ranges=(60.0, 30.0, 15.0),
                        azimuth_deg=AZIMUTH_DEG, dip_deg=DIP_DEG, rng=rng)
    # z4/z5 drive the RESPONSES ONLY and appear in no covariate. Without an
    # unobservable component like this the demo is far too easy — the responses
    # would be a near-deterministic function of the covariate window and every
    # method would score R^2 ~ 0.9. Real deposits have metallurgical variation
    # the covariates do not see, and it is what caps achievable accuracy.
    z4 = gaussian_field(shape, rng=rng, **geom)
    z5 = gaussian_field(shape, cell=CELL_SIZE, ranges=(80.0, 40.0, 20.0),
                        azimuth_deg=AZIMUTH_DEG, dip_deg=DIP_DEG, rng=rng)

    # Depth zonation: a low-grade cap over the top third of the model. This is
    # the feature that makes one spatial block genuinely hard to predict from
    # the others, and it is deliberately reproduced here.
    zi = np.arange(nz)[None, None, :] * np.ones((nx, ny, 1))
    depth = zi / max(nz - 1, 1)                       # 0 = bottom, 1 = top
    cap = 1.0 / (1.0 + np.exp((depth - 0.72) / 0.06))  # ~1 below, ~0 in the cap

    # ---- covariates ----
    # clipped where the Gaussian tails would run negative: these are assays
    fe = np.maximum(30.0 + 8.0 * z1 + 9.0 * cap + 1.0 * z3, 1.0)
    feo = np.maximum(0.34 * fe + 1.6 * z2 + 2.0, 0.1)
    magsus = np.maximum(60.0 + 34.0 * z1 + 55.0 * cap + 12.0 * z2, 1.0)
    # S tracks pyrite, which dilutes magnetite: anti-correlated with grade
    s = np.exp(0.35 - 0.9 * z1 - 1.1 * cap + 0.45 * z3)

    # ---- responses ----
    # Scaled so that both exceedance thresholds used downstream (55 and 80) sit
    # inside the distribution: a threshold nothing ever crosses yields a
    # degenerate all-zero probability column.
    dtr = (22.0 + 1.30 * (fe - 30.0) + 0.14 * (magsus - 60.0)
           - 7.0 * s + 34.0 * cap + 6.0 * z4 + 3.0 * z5)
    mag = (20.0 + 1.10 * (fe - 30.0) + 0.11 * (magsus - 60.0)
           - 5.5 * s + 30.0 * cap + 7.0 * z4 + 4.0 * z2)
    dtr = np.clip(dtr + 2.5 * rng.standard_normal(shape), 5.0, 98.0)
    mag = np.clip(mag + 2.5 * rng.standard_normal(shape), 5.0, 98.0)

    # ---- coordinates ----
    xs = ORIGIN[0] + np.arange(nx) * dx
    ys = ORIGIN[1] + np.arange(ny) * dy
    zs = ORIGIN[2] + np.arange(nz) * dz
    XC, YC, ZC = np.meshgrid(xs, ys, zs, indexing="ij")

    # ---- mineralised envelope ----
    # A real estimated block model only carries covariates inside the domain
    # that was estimated, not over the whole bounding box. Reproducing that
    # matters: it is what keeps the inference array a manageable size, and it
    # is what makes boundary windows fail the coverage rule.
    # Shape: a broad body spanning the model in X and Y, thinner in Z, with an
    # irregular boundary from a smooth random field. A purely random blob would
    # cluster in one corner and leave the fold blocks badly unbalanced.
    env = gaussian_field(shape, cell=CELL_SIZE, ranges=(200.0, 140.0, 60.0),
                         azimuth_deg=AZIMUTH_DEG, dip_deg=DIP_DEG, rng=rng)
    gx, gy, gz = np.meshgrid(*[(np.arange(n) - (n - 1) / 2) / ((n - 1) / 2)
                               for n in shape], indexing="ij")
    body = 1.0 - (0.35 * gx ** 2 + 0.55 * gy ** 2 + 1.00 * gz ** 2) + 0.45 * env
    inside = body > np.quantile(body, 1.0 - ENVELOPE_FRACTION)

    # ---- missing covariates ----
    void = (~inside) | (rng.random(shape) < FRAC_MISSING)
    for a in (fe, feo, magsus, s):
        a[void] = np.nan

    # ---- responses only along drillhole traces, inside the envelope ----
    have = np.zeros(shape, dtype=bool)
    margin = 3                                   # keep collars off the boundary
    core = inside.copy()
    core[:margin] = core[-margin:] = False
    core[:, :margin] = core[:, -margin:] = False
    cand = np.argwhere(core.sum(axis=2) >= HOLE_LEN_NODES[1])
    if len(cand) < N_COLLARS:
        raise RuntimeError("envelope too thin for the requested collars")

    # Space the collars out. Drillholes packed close together make the CV
    # test-to-train distance far shorter than the deployment distance, i.e. an
    # unrealistically easy problem; the whole point of the fold design is that
    # those two distributions match.
    order = rng.permutation(len(cand))
    picked = []
    for c in order:
        p = cand[c]
        if all(np.abs(p - q).max() >= MIN_COLLAR_SPACING_NODES for q in picked):
            picked.append(p)
        if len(picked) == N_COLLARS:
            break
    print(f"collars placed: {len(picked)} (requested {N_COLLARS})")

    for i, j in picked:
        ks = np.where(core[i, j])[0]
        n_samp = int(rng.integers(*HOLE_LEN_NODES))
        start = int(rng.integers(0, len(ks) - n_samp + 1))
        have[i, j, ks[start:start + n_samp]] = True
    have &= ~void                                # a sample needs its covariates

    dtr_out = np.where(have, dtr, np.nan)
    mag_out = np.where(have, mag, np.nan)

    df = pd.DataFrame({
        "XC": XC.ravel(), "YC": YC.ravel(), "ZC": ZC.ravel(),
        "Fe": fe.ravel(), "FeO": feo.ravel(),
        "MagSus": magsus.ravel(), "S": s.ravel(),
        "DTR": dtr_out.ravel(), "Magnetic": mag_out.ravel(),
    })
    return df, int(have.sum())


def main():
    shape = GRID_SMALL if "--small" in sys.argv else GRID
    rng = np.random.default_rng(SEED)

    print("=" * 70)
    print("SYNTHETIC DEMO DATA - not the case-study deposit, not paper numbers")
    print(f"grid {shape} nodes | cell {CELL_SIZE} m | seed {SEED}")
    print("=" * 70)

    df, n_samples = build(shape, rng)

    vec_dir = PROJECT_ROOT / "Vector"
    vec_dir.mkdir(parents=True, exist_ok=True)
    grid_csv = vec_dir / "grade_full_D3.csv"
    df.to_csv(grid_csv, index=False, float_format="%.4f")

    # Point-support table: the sampled nodes only. NOTE the covariate columns
    # are spelled differently here than in the grid CSV (FE/FEO vs Fe/FeO) —
    # that is how the case-study files are, and GLSRF.FEATURE_COLS expects this
    # spelling, so the demo mirrors it rather than quietly tidying it up.
    samples = (df.dropna(subset=["DTR", "Magnetic"])
                 .rename(columns={"Fe": "FE", "FeO": "FEO", "MagSus": "Magsus"})
                 .reset_index(drop=True))
    classic_xlsx = PROJECT_ROOT / "data_classic.xlsx"
    samples.to_excel(classic_xlsx, index=False)

    print(f"\ngrid nodes        : {len(df)}")
    print(f"sampled nodes     : {n_samples} (responses present)")
    print(f"covariates missing: {int(df['Fe'].isna().sum())} nodes "
          f"({100 * df['Fe'].isna().mean():.1f}%)")
    for c in ("Fe", "MagSus", "S", "DTR", "Magnetic"):
        v = df[c].dropna()
        print(f"  {c:9s} min={v.min():7.2f}  median={v.median():7.2f}  max={v.max():7.2f}")
    print(f"\nSaved: {grid_csv}")
    print(f"Saved: {classic_xlsx}")
    print("\nNext: python build_voxels.py -> make_folds.py -> srf_train.py")


if __name__ == "__main__":
    main()
