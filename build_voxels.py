# -*- coding: utf-8 -*-
"""
Build Voxels — step 1 of the SRF pipeline.

Turns the estimated grid table (grade_full_D3.csv: one row per grid node with
Fe, FeO, MagSus, S and the two responses) into the 3D voxel patterns the SRF
consumes: for every grid node that passes the coverage rule, a k x k x k window
centred on it is vectorised into one feature row.

Both kernel sizes (k=3 and k=5) are built in one run. k=3 is the reported SRF
variant; k=5 is retained for the kernel-size ablation.

FEATURE LAYOUT (the contract srf_core_final.py assumes):
    the missingness mask is INTERLEAVED as the 5th channel of every voxel
        [vox0: Fe,FeO,MagSus,S,MASK, vox1: Fe,FeO,MagSus,S,MASK, ...]
    with voxel index vox = i*k^2 + j*k + kk, offsets (i-half, j-half, kk-half).
    Appending the mask as a separate k^3 block at the end instead would leave
    plain tree splits unaffected but silently scramble the Z-rotation
    augmentation maps, the anisotropy features and the zone-of-influence maps,
    because those index features by voxel.

MISSING DATA:
    A voxel counts as valid only if ALL FOUR covariates are present there, so
    the mask channel means "any covariate missing" and stays correct whether or
    not the covariates are co-present. At load time the script reports how each
    covariate's missingness agrees with Fe's, so the assumption is verified
    rather than assumed. A window is accepted when at least MIN_COVERAGE of its
    voxels are valid; the remaining voxels are filled by inverse-distance
    weighting (power 2) over the valid voxels of the same window, using physical
    voxel-centre distances from CELL_SIZE.

After running this script, re-run make_folds.py (row counts and centres change)
and then srf_train.py.

Outputs (<root>/Vector):
  X_train_{k3}.npy            (N, k^3*5)  interleaved layout (see above)
  y_train_{k3}.npy            (N, 2) -> [DTR, Magnetic]
  centers_train_{k3}.npy      (N, 3)
  centers_index_train_{k3}.npy
  block_params_train_{k3}.npy (N, 4) window means of Fe,FeO,MagSus,S
  X_infer_{k3}.npy, centers_infer_{k3}.npy, centers_index_infer_{k3}.npy,
  block_params_infer_{k3}.npy
  Grid_cov.npy, Grid_dtr.npy, Grid_mag.npy
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------- Config ----------------
try:
    _HERE = Path(__file__).resolve().parent
except NameError:                       # notebook cell
    _HERE = Path.cwd()

# Defaults to the repository folder; set SRF_PROJECT_ROOT to run against data
# kept elsewhere. Input and output both live under <root>/Vector.
PROJECT_ROOT = Path(os.environ.get("SRF_PROJECT_ROOT", _HERE))

csv_path = PROJECT_ROOT / "Vector" / "grade_full_D3.csv"
out_dir = PROJECT_ROOT / "Vector"

# Internal prop names (kept consistent with the SRF trainer) -> CSV columns
COLUMN_MAP = {
    "FE": "Fe",
    "FEO": "FeO",
    "Magsus": "MagSus",
    "S": "S",
}
props = list(COLUMN_MAP.keys())          # ["FE", "FEO", "Magsus", "S"]
n_props = len(props)

resp_cols = ["DTR", "Magnetic"]          # y saved as [DTR, Magnetic]

VOXEL_SIZES = [3, 5]                     # both datasets built in one run

# Coverage policy: accept a window if >= 70% of its voxels are fully observed
MIN_COVERAGE = 0.70

# IDW settings
CELL_SIZE = (5.0, 5.0, 2.0)              # (dx, dy, dz) metres
IDW_POWER = 2.0
IDW_EPS = 1e-8


# ---------------- Utilities ----------------
def print_progress(prefix, current, total):
    if total <= 0:
        return
    pct = 100.0 * current / float(total)
    bar_len = 30
    filled = int(bar_len * pct / 100.0)
    sys.stdout.write(f"\r{prefix} [{'=' * filled}{'.' * (bar_len - filled)}] {pct:5.1f}%")
    sys.stdout.flush()


def parse_ijk_cell(val):
    if pd.isna(val):
        return None
    if isinstance(val, (int, np.integer)):
        return None
    s = str(val)
    for sep in ["_", "-", ",", ";", " "]:
        if sep in s:
            parts = s.split(sep)
            if len(parts) == 3:
                try:
                    return tuple(int(p) for p in parts)
                except Exception:
                    return None
    return None


def idx_to_coord(i, j, k, x_unique, y_unique, z_unique):
    if x_unique is not None:
        return float(x_unique[i]), float(y_unique[j]), float(z_unique[k])
    return float(i), float(j), float(k)


def build_voxel_coords(k, cell_size):
    """Centered physical coordinates for all k^3 voxels, C-order
    (i outer, j middle, kk inner) — matches the reshape/ravel order used
    everywhere below AND the SRF trainer contract (vox = i*k^2 + j*k + kk)."""
    half = k // 2
    dx, dy, dz = cell_size
    coords = np.zeros((k ** 3, 3), dtype=np.float32)
    t = 0
    for i in range(k):
        for j in range(k):
            for kk in range(k):
                coords[t] = ((i - half) * dx, (j - half) * dy, (kk - half) * dz)
                t += 1
    return coords


def idw_impute_block(block_4d, valid_mask_3d, voxel_coords, power=2.0, eps=1e-8):
    """IDW-impute missing voxels for ALL properties using the shared validity
    mask. block_4d: (k,k,k,n_props) with NaNs; valid_mask_3d: (k,k,k) bool."""
    k = block_4d.shape[0]
    k3 = k ** 3

    flat = block_4d.reshape(k3, n_props).astype(np.float32, copy=False)
    valid_flat = valid_mask_3d.reshape(k3)
    miss_flat = ~valid_flat
    if not np.any(miss_flat):
        return block_4d

    valid_idx = np.where(valid_flat)[0]
    miss_idx = np.where(miss_flat)[0]
    if valid_idx.size == 0:
        return block_4d

    C_valid = voxel_coords[valid_idx]
    C_miss = voxel_coords[miss_idx]

    diff = C_miss[:, None, :] - C_valid[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2)).astype(np.float32)

    w = 1.0 / np.power(dist + eps, power)
    w_sum = np.sum(w, axis=1, keepdims=True)

    X_valid = flat[valid_idx, :]
    X_hat = (w @ X_valid) / np.maximum(w_sum, eps)

    flat_filled = flat.copy()
    flat_filled[miss_idx, :] = X_hat.astype(np.float32)
    return flat_filled.reshape(k, k, k, n_props)


def make_feature_vector(cov_filled, valid_mask, k3):
    """Interleave the mask as the 5th channel of every voxel.
    Output order: [vox0: FE,FEO,Magsus,S,MASK, vox1: ...] — SRF contract."""
    cov_flat = cov_filled.reshape(k3, n_props)                       # (k3, 4)
    mask_col = (~valid_mask.reshape(k3)).astype(np.float32)[:, None] # 1 = missing
    return np.concatenate([cov_flat, mask_col], axis=1).ravel(order="C")


# ---------------- Load CSV ----------------
print(f"Loading {csv_path} ...")
df = pd.read_csv(csv_path, low_memory=False)

missing_csv_cols = [c for c in ["XC", "YC", "ZC"] + list(COLUMN_MAP.values()) + resp_cols
                    if c not in df.columns]
if missing_csv_cols:
    raise ValueError(f"CSV is missing required columns: {missing_csv_cols}")

# '-' placeholders -> NaN, everything numeric
for c in ["XC", "YC", "ZC"] + list(COLUMN_MAP.values()) + resp_cols:
    df[c] = pd.to_numeric(df[c].replace("-", np.nan), errors="coerce")

df = df.dropna(subset=["XC", "YC", "ZC"]).copy()

# ---------------- Covariate co-presence check ----------------
print("\n=== Covariate missingness concordance (vs Fe) ===")
fe_ok = df[COLUMN_MAP["FE"]].notna()
n_fe = int(fe_ok.sum())
worst_mismatch = 0
for name, col in COLUMN_MAP.items():
    if name == "FE":
        continue
    ok = df[col].notna()
    fe_only = int((fe_ok & ~ok).sum())      # Fe present, this prop missing
    other_only = int((~fe_ok & ok).sum())   # this prop present, Fe missing
    worst_mismatch = max(worst_mismatch, fe_only, other_only)
    print(f"  {name:7s}: present={int(ok.sum()):7d} | Fe-only={fe_only:6d} | {name}-only={other_only:6d}")
print(f"  Fe present at {n_fe} of {len(df)} nodes.")
if worst_mismatch == 0:
    print("  -> perfectly homogeneous: Fe presence == all-covariate presence.")
else:
    print(f"  -> NOT homogeneous (worst mismatch {worst_mismatch} nodes).")
print("  Validity mask below uses ALL covariates (correct either way).\n")

# ---------------- Indexing: IJK triple if parsable, else unique coords ----------------
if "IJK" in df.columns:
    ijk_parsed = df["IJK"].apply(parse_ijk_cell)
else:
    ijk_parsed = pd.Series([None] * len(df))

use_ijk = ijk_parsed.notna().all() and len(df) > 0

if use_ijk:
    ijk = np.vstack(ijk_parsed.values)
    i_idx, j_idx, k_idx = ijk[:, 0], ijk[:, 1], ijk[:, 2]
    i_idx -= i_idx.min(); j_idx -= j_idx.min(); k_idx -= k_idx.min()
    Nx, Ny, Nz = int(i_idx.max() + 1), int(j_idx.max() + 1), int(k_idx.max() + 1)
    x_unique = y_unique = z_unique = None
    print("Indexing: parsed IJK triples.")
else:
    x_unique = np.sort(df["XC"].unique())
    y_unique = np.sort(df["YC"].unique())
    z_unique = np.sort(df["ZC"].unique())
    Nx, Ny, Nz = len(x_unique), len(y_unique), len(z_unique)
    x_index = {x: i for i, x in enumerate(x_unique)}
    y_index = {y: j for j, y in enumerate(y_unique)}
    z_index = {z: k for k, z in enumerate(z_unique)}
    i_idx = df["XC"].map(x_index).values.astype(int)
    j_idx = df["YC"].map(y_index).values.astype(int)
    k_idx = df["ZC"].map(z_index).values.astype(int)
    print("Indexing: unique-coordinate grid.")

# ---------------- Build grids ----------------
Grid_cov = np.full((Nx, Ny, Nz, n_props), np.nan, dtype=np.float32)
for p, name in enumerate(props):
    Grid_cov[i_idx, j_idx, k_idx, p] = df[COLUMN_MAP[name]].values.astype(np.float32)

Grid_dtr = np.full((Nx, Ny, Nz), np.nan, dtype=np.float32)
Grid_dtr[i_idx, j_idx, k_idx] = df["DTR"].values.astype(np.float32)

Grid_mag = np.full((Nx, Ny, Nz), np.nan, dtype=np.float32)
Grid_mag[i_idx, j_idx, k_idx] = df["Magnetic"].values.astype(np.float32)

out_dir.mkdir(parents=True, exist_ok=True)
np.save(out_dir / "Grid_cov.npy", Grid_cov)
np.save(out_dir / "Grid_dtr.npy", Grid_dtr)
np.save(out_dir / "Grid_mag.npy", Grid_mag)


# ---------------- Per-kernel build ----------------
def build_for_kernel(voxel_size: int):
    if voxel_size % 2 != 1 or voxel_size < 1:
        raise ValueError("voxel_size must be a positive odd integer.")
    half = voxel_size // 2
    k3 = voxel_size ** 3
    voxel_coords = build_voxel_coords(voxel_size, CELL_SIZE)

    i_min, i_max = half, Nx - half
    j_min, j_max = half, Ny - half
    k_min, k_max = half, Nz - half
    n_total = max((i_max - i_min) * (j_max - j_min) * (k_max - k_min), 1)

    print(f"\n===== kernel {voxel_size}^3 (k3={k3}) | coverage >= {MIN_COVERAGE:.2f} =====")

    X_train, y_train, centers_train, cidx_train, params_train = [], [], [], [], []
    X_infer, centers_infer, cidx_infer, params_infer = [], [], [], []

    counter = 0
    for i in range(i_min, i_max):
        for j in range(j_min, j_max):
            for k in range(k_min, k_max):
                counter += 1
                if counter % 2000 == 0 or counter == n_total:
                    print_progress(f"k={voxel_size} progress", counter, n_total)

                cov_block = Grid_cov[i - half:i + half + 1,
                                     j - half:j + half + 1,
                                     k - half:k + half + 1, :]

                # validity from ALL covariates, not Fe alone
                valid_mask = ~np.isnan(cov_block).any(axis=-1)
                coverage = float(np.count_nonzero(valid_mask)) / float(k3)
                if coverage < MIN_COVERAGE:
                    continue

                cov_filled = idw_impute_block(cov_block, valid_mask, voxel_coords,
                                              power=IDW_POWER, eps=IDW_EPS)
                vec = make_feature_vector(cov_filled, valid_mask, k3)   # interleaved layout
                means = np.mean(cov_filled.reshape(-1, n_props), axis=0).astype(np.float32)
                coord = idx_to_coord(i, j, k, x_unique, y_unique, z_unique)

                # INFER: every accepted window
                X_infer.append(vec)
                centers_infer.append(coord)
                cidx_infer.append((i, j, k))
                params_infer.append(means)

                # TRAIN: additionally needs BOTH responses at the center
                center_dtr = Grid_dtr[i, j, k]
                center_mag = Grid_mag[i, j, k]
                if np.isnan(center_dtr) or np.isnan(center_mag):
                    continue
                X_train.append(vec)
                y_train.append((center_dtr, center_mag))
                centers_train.append(coord)
                cidx_train.append((i, j, k))
                params_train.append(means)
    print()

    def stack(lst, width, dtype=np.float32):
        return (np.vstack(lst).astype(dtype) if lst
                else np.empty((0, width), dtype=dtype))

    D = k3 * (n_props + 1)
    X_tr = stack(X_train, D)
    y_tr = stack(y_train, 2)
    c_tr = stack(centers_train, 3)
    ci_tr = stack(cidx_train, 3, np.int32)
    p_tr = stack(params_train, n_props)
    X_in = stack(X_infer, D)
    c_in = stack(centers_infer, 3)
    ci_in = stack(cidx_infer, 3, np.int32)
    p_in = stack(params_infer, n_props)

    np.save(out_dir / f"X_train_{k3}.npy", X_tr)
    np.save(out_dir / f"y_train_{k3}.npy", y_tr)
    np.save(out_dir / f"centers_train_{k3}.npy", c_tr)
    np.save(out_dir / f"centers_index_train_{k3}.npy", ci_tr)
    np.save(out_dir / f"block_params_train_{k3}.npy", p_tr)
    np.save(out_dir / f"X_infer_{k3}.npy", X_in)
    np.save(out_dir / f"centers_infer_{k3}.npy", c_in)
    np.save(out_dir / f"centers_index_infer_{k3}.npy", ci_in)
    np.save(out_dir / f"block_params_infer_{k3}.npy", p_in)

    # layout self-check: every 5th column must be the binary mask channel
    if X_tr.shape[0] > 0:
        mask_cols = X_tr[:, n_props::n_props + 1]
        assert np.isin(mask_cols, [0.0, 1.0]).all(), "mask channel not interleaved!"

    print(f"TRAIN: X {X_tr.shape} | y {y_tr.shape}")
    print(f"INFER: X {X_in.shape}")
    print(f"X layout: [vox: FE,FEO,Magsus,S,MASK] x {k3}  (D={D})  -- SRF contract OK")
    return X_tr.shape[0], X_in.shape[0]


# ---------------- Run ----------------
summary = {}
for vs in VOXEL_SIZES:
    summary[vs] = build_for_kernel(vs)

print("\n=== SUMMARY ===")
print(f"Grid size (Nx,Ny,Nz): {Nx}, {Ny}, {Nz}")
print(f"Coverage threshold  : {MIN_COVERAGE:.2f} (max missing = {1.0 - MIN_COVERAGE:.2f})")
for vs, (ntr, nin) in summary.items():
    print(f"k={vs}: TRAIN={ntr} | INFER={nin}")
print(f"Files saved under   : {out_dir}")
print("\nNEXT STEPS (required):")
print("  1) re-run make_folds.py  (row counts/centers changed)")
print("  2) re-run srf_train.py   (train on the regenerated vectors)")
