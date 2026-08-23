# -*- coding: utf-8 -*-
r"""
GLSRF — RF-GLS for 3D block-model prediction (single-file implementation).

Implements Saha, Basu & Datta (2023, JASA), "Random Forests for Spatially
Dependent Data", with the 3D anisotropic extension used in this project:

  Y = m(X) + eps,   eps(s) = omega(s) + eps*,   omega ~ GP(0, sigma^2 rho(d))

  * Eq. 5  global DART split criterion: a split is scored on ALL n rows
           against the FULL current partition, not inside the node.
  * Eq. 6  node representatives beta = (Z'QZ)^-1 Z'QY, solved JOINTLY for
           every node of the tree once growth stops.
  * Eq. 10 contrast resampling: trees resample rows of the decorrelated
           contrasts Ytil = W Y (W'W = Sigma^-1), not rows of Y.
  * Eq. 11 kriging prediction: yhat(x0,s0) = mhat(x0) + v0' Sigma^-1 (Y - mhat).
  * Sec 2.7 Sigma is a *working* covariance chosen by the user; consistency
           does not require it to be correct.

Deviations from the published algorithm, deliberate and recorded:
  - Dense exact Cholesky instead of the NNGP sparse approximation. At n ~ 500
    the exact factor is cheap and strictly more accurate; NNGP exists for
    scalability, not fidelity.
  - Sigma's parameters are IMPOSED from the deposit's experimental variogram
    rather than estimated by maximum likelihood from RF residuals (the
    paper's Sec. 2.7 recipe). Section 2 below is where they are declared.

Coordinates are NOT model features. Adding them is the paper's separate
"RF-Loc" baseline; doing both lets the forest absorb the spatial signal
through m(X), leaving nothing for Sigma to correct, which forces the nugget
toward 1 (Sigma -> I) and silently turns RF-GLS back into plain RF. Spatial
information enters this model through Sigma alone.

Run:  python GLSRF.py
      --smoke   tiny grid and tiny forests, to check the plumbing in minutes
                rather than hours. NOT paper numbers.

A full run is expensive — the reported one took 5.8 h for 32 configs x 3
sub-folds x 5 folds x 2 targets. Use --smoke first, and FOLDS_TO_RUN in
Section 5 to work on a single fold.

The paper's headline GLS-RF model is not this run's per-fold selection: it is
the ANISO/exp design held fixed across folds, under the mean-function
prediction rule. Run glsrf_report.py on the finished run folder to produce it.
"""

import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.linalg import cho_solve, solve_triangular

GLSRF_VERSION = "3.0"

SMOKE = "--smoke" in sys.argv

try:
    _HERE = Path(__file__).resolve().parent
except NameError:                       # notebook cell
    _HERE = Path.cwd()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


# ##############################################################
# #                        CONFIGURATION                       #
# ##############################################################

# ==============================================================
# SECTION 1 — DATA
# ==============================================================

# Every input and output hangs off one root. Defaults to the repository folder;
# set the SRF_PROJECT_ROOT environment variable to run against data elsewhere.
PROJECT_ROOT = Path(os.environ.get("SRF_PROJECT_ROOT", _HERE))

FOLDS_ROOT = PROJECT_ROOT / "CV_folds"
ROLES_PATH = FOLDS_ROOT / "classic" / "roles_classic.npy"
FOLDS_CSV = FOLDS_ROOT / "classic" / "folds_classic.csv"

# The table the folds were built from is resolved from fold_config.json so the
# data and the fold assignment cannot silently disagree. Set to a path to pin.
DATA_PATH_OVERRIDE = None

COORD_COLS = ["XC", "YC", "ZC"]
FEATURE_COLS = ["FE", "FEO", "Magsus", "S"]     # coordinates excluded by design
TARGET_COLS = ["DTR", "Magnetic"]

METHOD_NAME = "GLSRF"
RUNS_BASE = PROJECT_ROOT / "GLSRF_runs"

# Only used by predict_full_block() (Section E); ignored by the fold run.
FULL_GRID_PATH = PROJECT_ROOT / "Vector" / "grade_full_D3.csv"
FULL_GRID_BATCH = 100_000


# ==============================================================
# SECTION 2 — SPATIAL COVARIANCE MODEL  (the variogram)
#             >>> everything geostatistical is declared here <<<
# ==============================================================
# Sigma = (1 - NUGGET) * rho(h') + NUGGET * I, where h' is the lag rotated
# into the ellipsoid frame and divided by the range along each axis.

# --- nugget effect -------------------------------------------------------
# NUGGET = C0 / (C0 + C1) = tau^2 / (sigma^2 + tau^2), the share of variance
# that is NOT spatially structured. Read straight off the experimental
# variogram of the deposit. It is the single most influential parameter of
# Sigma: at NUGGET = 0 the model asserts ~0.99 correlation between samples a
# few metres apart, and inverting such a matrix amplifies analytical noise
# instead of removing redundancy.
NUGGET = 0.20

# --- ellipsoid orientation (degrees) -------------------------------------
AZIMUTH_DEG = 110.0        # of the major axis, clockwise from North
DIP_DEG = 25.0             # positive downward
TILT_DEG = 20.0            # roll of semi/minor about the major axis

# --- ranges along the ellipsoid axes (metres) ----------------------------
RANGE_MAJOR = 120.0
RANGE_SEMI = 50.0
RANGE_MINOR = 25.0

# --- correlation function ------------------------------------------------
# "exp"      : rho(h) = exp(-h)                     (kink at the origin)
# "matern32" : rho(h) = (1 + sqrt3 h) exp(-sqrt3 h) (smooth at the origin)
KERNEL = "exp"

# --- numerical conditioning ---------------------------------------------
JITTER = 1e-5              # added to the diagonal; NOT a substitute for NUGGET
RIDGE = 1e-5               # Tikhonov term in the GLS normal equations

# --- design comparison ---------------------------------------------------
# With True the run also fits the isotropic control (all three ranges set to
# RANGE_MAJOR) and the alternative kernel, giving the 2x2 design table that
# justifies the choice above. The design is selected on the sub-CV metric and
# is held FIXED across folds — never chosen per fold.
COMPARE_DESIGNS = True
COMPARE_KERNELS = ["exp", "matern32"]

# --- prediction rule -----------------------------------------------------
# True  -> Eq. 11: mhat(x0) + kriging of the residual field through Sigma.
#          This is how spatial position reaches a prediction in RF-GLS.
# False -> mhat(x0) only (mean function; no spatial term at predict time).
# Both are always computed and reported; this flag picks the headline.
USE_RESIDUAL_KRIGING = True


# ==============================================================
# SECTION 3 — FOREST HYPERPARAMETERS
# ==============================================================

N_ESTIMATORS_GRID = [150]
MAX_DEPTH_GRID = [6, 10]
MIN_LEAF_GRID = [2, 4]
MAX_FEATURES_GRID = [0.5, 1.0]     # fraction of the 4 covariates tried per split
MAX_SAMPLES_GRID = [0.7]
BOOTSTRAP = True                   # contrast resampling with replacement


# ==============================================================
# SECTION 4 — CROSS-VALIDATION AND TUNING
# ==============================================================

# Hyperparameters are ranked by nested spatial sub-fold CV inside the training
# rows of each outer fold. Test rows are never touched during selection.
N_SUBFOLDS = 3
SUBFOLD_SEED = 777


# ==============================================================
# SECTION 5 — RUN CONTROL
# ==============================================================

SEED = 42
N_JOBS = -1                # parallelism over trees within a forest
FOLDS_TO_RUN = None        # None = every fold, or e.g. [0, 1]

# joblib backend for the tree loop. "processes" is ~10x faster here: the split
# search is a Python-level loop over features whose numpy calls are too small
# to amortise the GIL, so threads serialise it. Only use "threads" to debug.
PARALLEL_PREFER = "processes"

if SMOKE:
    # Plumbing check only: 4 configs of 20 trees instead of 32 of 150.
    N_ESTIMATORS_GRID = [20]
    MAX_DEPTH_GRID = [6]
    MIN_LEAF_GRID = [4]
    MAX_FEATURES_GRID = [0.5]
    COMPARE_KERNELS = ["exp", "matern32"]

# Reproducibility note: results are exactly reproducible for a FIXED N_JOBS,
# but not across values of it. Worker processes run BLAS single-threaded while
# n_jobs=1 does not, and the different reduction order flips the occasional
# near-tied split. The effect is ~0.006 R2 on a single fold and ~0.001 pooled.
# Report numbers together with the N_JOBS they were produced under, or pin
# BLAS threads (threadpoolctl) if backend-independent output is needed.


# ##############################################################
# #                       IMPLEMENTATION                       #
# ##############################################################

# ==============================================================
# PART A — anisotropic kernels and GLS algebra
# ==============================================================

def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    return v * 0.0 if n < eps else v / n


def _rodrigues_rotate(v, axis, theta_rad):
    axis = _normalize(axis)
    ct, st = np.cos(theta_rad), np.sin(theta_rad)
    return v * ct + np.cross(axis, v) * st + axis * np.dot(axis, v) * (1.0 - ct)


def rotation_matrix_geo(azimuth_deg: float, dip_deg: float,
                        tilt_deg: float = 0.0) -> np.ndarray:
    """Rows are the [major, semi, minor] unit axes. X=East, Y=North, Z=Up.
    Azimuth is clockwise from North, dip positive downward. At tilt=0 this
    reproduces the SRF pipeline's frame exactly."""
    az, dp = np.deg2rad(azimuth_deg), np.deg2rad(dip_deg)
    major = _normalize(np.array([np.sin(az) * np.cos(dp),
                                 np.cos(az) * np.cos(dp),
                                 -np.sin(dp)], dtype=np.float64))
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(ref, major)) > 0.99:
        ref = np.array([0.0, 1.0, 0.0])
    semi = _normalize(np.cross(ref, major))
    tl = np.deg2rad(tilt_deg)
    if abs(tl) > 0:
        semi = _normalize(_rodrigues_rotate(semi, major, tl))
    minor = _normalize(np.cross(major, semi))
    return np.vstack([major, semi, minor])


def _scaled(coords, R, ranges):
    """Rotate into the ellipsoid frame and divide by the range on each axis,
    so Euclidean distance in the result is the reduced lag h'."""
    return (coords @ R.T) / np.asarray(ranges, dtype=np.float64)[None, :]


def _rho(D: np.ndarray, kernel: str) -> np.ndarray:
    if kernel == "exp":
        return np.exp(-D)
    if kernel == "matern32":
        r = np.maximum(D, 1e-12)
        return (1.0 + np.sqrt(3.0) * r) * np.exp(-np.sqrt(3.0) * r)
    raise ValueError("kernel must be 'exp' or 'matern32'")


def build_sigma(coords: np.ndarray, Lx: float, Ly: float, Lz: float,
                kernel: str, azimuth_deg: float, dip_deg: float,
                tilt_deg: float, jitter: float, nugget: float) -> np.ndarray:
    """Cov(Y)/(sigma^2 + tau^2) for one set of locations."""
    R = rotation_matrix_geo(azimuth_deg, dip_deg, tilt_deg)
    S = _scaled(coords, R, (Lx, Ly, Lz))
    diff = S[:, None, :] - S[None, :, :]
    Sigma = _rho(np.sqrt((diff ** 2).sum(axis=2)), kernel)
    if nugget > 0.0:
        Sigma = (1.0 - nugget) * Sigma
        np.fill_diagonal(Sigma, 1.0)
    return Sigma + jitter * np.eye(Sigma.shape[0])


def cross_covariance(coords_new: np.ndarray, coords_tr: np.ndarray,
                     Lx: float, Ly: float, Lz: float, kernel: str,
                     azimuth_deg: float, dip_deg: float, tilt_deg: float,
                     nugget: float, **_ignored) -> np.ndarray:
    """v0 = Cov(omega(s_new), Y(s_train)), same normalisation as build_sigma.
    The nugget is measurement error and is not shared across locations, so it
    scales the off-diagonal block but never adds to it."""
    R = rotation_matrix_geo(azimuth_deg, dip_deg, tilt_deg)
    A = _scaled(coords_new, R, (Lx, Ly, Lz))
    B = _scaled(coords_tr, R, (Lx, Ly, Lz))
    diff = A[:, None, :] - B[None, :, :]
    return (1.0 - nugget) * _rho(np.sqrt((diff ** 2).sum(axis=2)), kernel)


def build_whitener(Sigma: np.ndarray):
    """Return (L, W) with Sigma = L L' and W = L^-1, so W'W = Sigma^-1."""
    n = Sigma.shape[0]
    for extra in (0.0, 1e-8, 1e-6, 1e-4, 1e-2):
        try:
            L = np.linalg.cholesky(Sigma + extra * np.eye(n))
            break
        except np.linalg.LinAlgError:
            continue
    else:
        raise np.linalg.LinAlgError("Cholesky failed for Sigma.")
    return L, solve_triangular(L, np.eye(n), lower=True, check_finite=False)


# ==============================================================
# PART B — RF-GLS tree (global DART criterion, Eq. 5 / 6)
# ==============================================================
#
# Eq. 5 scores a split on every row against the whole partition:
#
#   v = 1/n [ (Y - Z0 b(Z0))' Q (Y - Z0 b(Z0))
#           - (Y - Z  b(Z ))' Q (Y - Z  b(Z )) ],   b(Z) = (Z'QZ)^-1 Z'QY
#
# with Z the n x g indicator matrix over EVERY current node. Splitting node l
# replaces its column by two and leaves all other columns untouched.
#
# Restricting this to the node's own rows and using (Sigma_CC)^-1 in place of
# (Sigma^-1)_CC is not an approximation but a different estimator: it throws
# away every correlation between the node and the rest of the data, and it
# penalises a correctly specified anisotropic Sigma, because shrinking the
# semi/minor ranges drives the WITHIN-node correlations to zero, Sigma_sub ->
# I, and the tree silently degenerates to plain CART.
#
# Implementation: whiten once per forest (W = L^-1). Contrast resampling then
# makes the GLS loss an ordinary least-squares loss on selected rows of
# Ytil = W Y and Ztil = W Z. For a candidate node, project the other columns
# out once (residual maker M), then sweep every distinct cut point with a
# cumulative sum so each candidate costs one 2x2 solve. Exact, not binned.

@dataclass
class TreeParams:
    max_depth: int = 6
    min_leaf: int = 4
    mtry: Optional[int] = None
    ridge: float = 1e-5


class GLSTree:
    """One RF-GLS tree: global split criterion + joint GLS node values."""

    def __init__(self, params: TreeParams,
                 rng: Optional[np.random.Generator] = None):
        self.params = params
        self.rng = rng if rng is not None else np.random.default_rng(42)
        self.feature_arr = None
        self.threshold_arr = None
        self.left_arr = None
        self.right_arr = None
        self.value_arr = None
        self.is_leaf_arr = None

    def _best_split(self, X, node_rows, W_S, Ytil_S, Ztil_S, col_pos):
        g = Ztil_S.shape[1]
        c = Ztil_S[:, col_pos]
        if g > 1:
            B = np.delete(Ztil_S, col_pos, axis=1)
            G = B.T @ B + self.params.ridge * np.eye(g - 1)
            try:
                Ginv = np.linalg.inv(G)
            except np.linalg.LinAlgError:
                Ginv = np.linalg.pinv(G)
            MY = Ytil_S - B @ (Ginv @ (B.T @ Ytil_S))
            Mc = c - B @ (Ginv @ (B.T @ c))
        else:
            B, Ginv = None, None
            MY, Mc = Ytil_S, c

        r0 = float(c @ MY)
        k0 = float(c @ Mc)
        if k0 <= 1e-12:
            return None, None, -np.inf

        n_node = node_rows.size
        ml = self.params.min_leaf
        if n_node < 2 * ml:
            return None, None, -np.inf
        base = r0 * r0 / k0

        p = X.shape[1]
        mtry = max(1, min(self.params.mtry or int(np.ceil(np.sqrt(p))), p))
        feats = self.rng.choice(p, size=mtry, replace=False)

        best = (None, None, -np.inf)
        for feat in feats:
            xs = X[node_rows, feat]
            order = np.argsort(xs, kind="stable")
            rows_sorted = node_rows[order]
            x_sorted = xs[order]

            Ucum = np.cumsum(W_S[:, rows_sorted], axis=1)        # (m, n_node)
            MU = Ucum - B @ (Ginv @ (B.T @ Ucum)) if B is not None else Ucum

            a = np.einsum("ij,ij->j", Ucum, MU)                  # u'Mu
            r = Ucum.T @ MY                                      # u'MY
            cu = Ucum.T @ Mc                                     # u'Mc

            uu = a
            uv = cu - a
            vv = k0 - 2.0 * cu + a
            ru, rv = r, r0 - r
            det = uu * vv - uv * uv

            j = np.arange(n_node)
            ok = (j + 1 >= ml) & (n_node - (j + 1) >= ml) & (det > 1e-12)
            ok[-1] = False                                       # no empty child
            if n_node > 1:
                ok[:-1] &= x_sorted[:-1] < x_sorted[1:]          # no split in ties
            if not ok.any():
                continue

            val = np.full(n_node, -np.inf)
            d = det[ok]
            val[ok] = (ru[ok] ** 2 * vv[ok]
                       - 2.0 * ru[ok] * rv[ok] * uv[ok]
                       + rv[ok] ** 2 * uu[ok]) / d
            gain = val - base
            jb = int(np.argmax(gain))
            if gain[jb] > best[2]:
                thr = 0.5 * (x_sorted[jb] + x_sorted[jb + 1])
                best = (int(feat), float(thr), float(gain[jb]))
        return best

    def fit(self, X, y, W_S):
        """X: (n, p) ALL training rows — the partition spans all of them.
        W_S: (m, n) the selected rows of the whitener (contrast resampling)."""
        n = X.shape[0]
        Ytil_S = W_S @ y

        feat_l, thr_l, left_l, right_l, val_l, leaf_l = [], [], [], [], [], []

        def add_node():
            feat_l.append(-1); thr_l.append(0.0)
            left_l.append(-1); right_l.append(-1)
            val_l.append(0.0); leaf_l.append(True)
            return len(feat_l) - 1

        root = add_node()
        live = [(root, np.arange(n, dtype=np.int64), 0)]   # (node, rows, column)
        terminal = []                                      # stopped before the last level
        Ztil_S = W_S.sum(axis=1)[:, None]                  # W_S @ 1
        depth = 0

        while depth < self.params.max_depth and live:
            nxt = []
            for nid, rows, col in live:
                feat = thr = None
                if rows.size >= 2 * self.params.min_leaf:
                    feat, thr, gain = self._best_split(
                        X, rows, W_S, Ytil_S, Ztil_S, col)
                    if feat is not None and not (np.isfinite(gain) and gain > 0.0):
                        feat = None
                if feat is None:
                    # An unsplit node stays IN the partition and keeps its
                    # column, so it must still receive a representative from
                    # the joint solve below. Dropping it here is a silent bug:
                    # the node falls back to a local mean and the tree is no
                    # longer the Eq. 6 estimator.
                    terminal.append((nid, rows, col))
                    continue

                mask = X[rows, feat] < thr
                rl, rr = rows[mask], rows[~mask]
                if rl.size < self.params.min_leaf or rr.size < self.params.min_leaf:
                    terminal.append((nid, rows, col))
                    continue

                zl = W_S[:, rl].sum(axis=1)
                zr = Ztil_S[:, col] - zl
                Ztil_S[:, col] = zl                        # left keeps the column
                Ztil_S = np.column_stack([Ztil_S, zr])     # right appends one
                col_r = Ztil_S.shape[1] - 1

                lid, rid = add_node(), add_node()
                leaf_l[nid] = False
                feat_l[nid] = feat
                thr_l[nid] = thr
                left_l[nid] = lid
                right_l[nid] = rid
                nxt.append((lid, rl, col))
                nxt.append((rid, rr, col_r))
            live = nxt
            depth += 1

        # ---- Eq. 6: all node representatives, solved jointly ----
        A = Ztil_S.T @ Ztil_S + self.params.ridge * np.eye(Ztil_S.shape[1])
        b = Ztil_S.T @ Ytil_S
        try:
            beta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(A) @ b
        for nid, _rows, col in terminal + live:
            val_l[nid] = float(beta[col])

        self.feature_arr = np.asarray(feat_l, dtype=np.int64)
        self.threshold_arr = np.asarray(thr_l, dtype=np.float64)
        self.left_arr = np.asarray(left_l, dtype=np.int64)
        self.right_arr = np.asarray(right_l, dtype=np.int64)
        self.value_arr = np.asarray(val_l, dtype=np.float64)
        self.is_leaf_arr = np.asarray(leaf_l, dtype=bool)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        N = X.shape[0]
        if N == 0:
            return np.array([], dtype=np.float64)
        cur = np.zeros(N, dtype=np.int64)
        while True:
            active = ~self.is_leaf_arr[cur]
            if not active.any():
                break
            ai = np.where(active)[0]
            ca = cur[ai]
            go_left = X[ai, self.feature_arr[ca]] < self.threshold_arr[ca]
            cur[ai] = np.where(go_left, self.left_arr[ca], self.right_arr[ca])
        return self.value_arr[cur]


# ==============================================================
# PART C — the forest
# ==============================================================

@dataclass
class ForestParams:
    n_estimators: int = 150
    max_depth: int = 6
    min_leaf: int = 4
    ridge: float = 1e-5
    bootstrap: bool = True
    max_features: Optional[float] = None
    max_samples: Optional[float] = None
    random_state: int = 42
    n_jobs: int = 1
    verbose: int = 0


class GLSRandomForest3D:
    """RF-GLS. Sigma and its whitener are built once per forest; trees differ
    only in which contrast rows they see (Eq. 10)."""

    def __init__(self, forest_params: ForestParams, sigma_kwargs: Dict[str, Any]):
        self.params = forest_params
        self.sk = sigma_kwargs
        self.trees: List[GLSTree] = []
        self.oob_pred_: Optional[np.ndarray] = None
        self.chol_L_: Optional[np.ndarray] = None
        self.coords_tr_: Optional[np.ndarray] = None
        self.X_tr_: Optional[np.ndarray] = None
        self.y_tr_: Optional[np.ndarray] = None

    def _resolve_mtry(self, p: int) -> int:
        mf = self.params.max_features
        if mf is None:
            mtry = int(np.ceil(np.sqrt(p)))
        elif 0.0 < mf <= 1.0:
            mtry = int(np.ceil(mf * p))
        else:
            mtry = int(mf)
        return max(1, min(mtry, p))

    def _sample_indices(self, rng: np.random.Generator, n: int) -> np.ndarray:
        if self.params.bootstrap:
            m = n if self.params.max_samples is None else max(1, int(self.params.max_samples * n))
            return rng.integers(0, n, size=m)
        frac = 0.7 if self.params.max_samples is None else self.params.max_samples
        m = min(max(self.params.min_leaf * 2, int(frac * n)), n)
        return rng.choice(n, size=m, replace=False)

    def fit(self, X: np.ndarray, y: np.ndarray, coords: np.ndarray):
        N, p = X.shape
        tp = TreeParams(max_depth=self.params.max_depth,
                        min_leaf=self.params.min_leaf,
                        mtry=self._resolve_mtry(p),
                        ridge=self.params.ridge)

        rng = np.random.default_rng(self.params.random_state)
        seeds = rng.integers(0, 1_000_000_000, size=self.params.n_estimators)

        L, W_full = build_whitener(build_sigma(coords, **self.sk))
        self.chol_L_ = L
        self.coords_tr_ = np.asarray(coords, dtype=np.float64)
        self.X_tr_ = X
        self.y_tr_ = np.asarray(y, dtype=np.float64)

        def fit_one(seed: int):
            local_rng = np.random.default_rng(seed)
            idx = self._sample_indices(local_rng, N)
            tree = GLSTree(tp, rng=np.random.default_rng(seed))
            tree.fit(X, y, W_full[idx, :])
            in_bag = np.zeros(N, dtype=bool)
            in_bag[idx] = True
            oob_idx = np.where(~in_bag)[0]
            oob_pred = tree.predict(X[oob_idx]) if oob_idx.size else np.array([])
            return tree, oob_idx, oob_pred

        results = Parallel(n_jobs=self.params.n_jobs, prefer=PARALLEL_PREFER,
                           verbose=self.params.verbose)(
            delayed(fit_one)(int(s)) for s in seeds)

        self.trees = [r[0] for r in results]
        oob_sum = np.zeros(N)
        oob_cnt = np.zeros(N, dtype=np.int64)
        for _, oob_idx, oob_pred in results:
            if oob_idx.size:
                oob_sum[oob_idx] += oob_pred
                oob_cnt[oob_idx] += 1
        with np.errstate(divide="ignore", invalid="ignore"):
            self.oob_pred_ = np.where(oob_cnt > 0, oob_sum / oob_cnt, np.nan)
        return self

    # ---- prediction ----

    def predict_all(self, X: np.ndarray) -> np.ndarray:
        """(T, N) per-tree predictions of the mean function."""
        if not self.trees:
            raise RuntimeError("Model not fitted.")
        return np.vstack([tree.predict(X) for tree in self.trees])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """mhat(x0): the mean function only, no spatial term."""
        return self.predict_all(X).mean(axis=0)

    def train_residuals(self) -> np.ndarray:
        """y - mhat on the training rows, using OOB predictions where they
        exist. In-sample fitted values would be shrunk toward y and would make
        the kriging correction look better than it is."""
        m_hat = np.asarray(self.oob_pred_, dtype=np.float64).copy()
        bad = ~np.isfinite(m_hat)
        if bad.any():
            m_hat[bad] = self.predict(self.X_tr_[bad])
        return self.y_tr_ - m_hat

    def predict_kriged(self, X: np.ndarray, coords: np.ndarray,
                       mean_pred: Optional[np.ndarray] = None) -> np.ndarray:
        """Eq. 11:  yhat(x0, s0) = mhat(x0) + v0' Sigma^-1 (Y - mhat)."""
        m_new = self.predict(X) if mean_pred is None else mean_pred
        alpha = cho_solve((self.chol_L_, True), self.train_residuals(),
                          check_finite=False)
        v0 = cross_covariance(coords, self.coords_tr_, **self.sk)
        return m_new + v0 @ alpha

    def oob_r2(self, y: np.ndarray) -> Dict[str, float]:
        mask = np.isfinite(self.oob_pred_)
        if not mask.any():
            return {"OOB_R2": np.nan, "OOB_RMSE": np.nan,
                    "OOB_MAE": np.nan, "OOB_coverage": 0}
        yt = np.asarray(y, dtype=np.float64)[mask]
        yp = self.oob_pred_[mask]
        ss_res = float(((yt - yp) ** 2).sum())
        ss_tot = float(((yt - yt.mean()) ** 2).sum())
        return {"OOB_R2": 1.0 - ss_res / max(ss_tot, 1e-300),
                "OOB_RMSE": float(np.sqrt(ss_res / yt.size)),
                "OOB_MAE": float(np.abs(yt - yp).mean()),
                "OOB_coverage": int(mask.sum())}


# ==============================================================
# PART D — fold runner
# ==============================================================

def resolve_data_path() -> Path:
    """The table the folds were built from, taken from fold_config.json so the
    data and the fold assignment cannot silently disagree.

    fold_config.json records an absolute path. If the project has since been
    moved or cloned elsewhere, fall back to the same file name under
    PROJECT_ROOT rather than failing — the row count is checked against the
    roles array either way, so a genuine mismatch is still caught.
    """
    if DATA_PATH_OVERRIDE is not None:
        return Path(DATA_PATH_OVERRIDE)
    cfg_path = FOLDS_ROOT / "fold_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"{cfg_path} not found - run make_folds.py, or set DATA_PATH_OVERRIDE.")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    ds = cfg["datasets"]["classic"]
    src = Path(ds["source"])
    if not src.exists():
        local = PROJECT_ROOT / src.name
        if not local.exists():
            raise FileNotFoundError(
                f"The folds were built from {src}, which no longer exists, and "
                f"there is no {src.name} under {PROJECT_ROOT}.")
        src = local
    return src


DATA_PATH = resolve_data_path()
RANK_COL = "SUBCV_R2_K" if USE_RESIDUAL_KRIGING else "SUBCV_R2"
TEST_COL = "TEST_R2_K" if USE_RESIDUAL_KRIGING else "TEST_R2"


def _to_py(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, Path):
        return str(o)
    return o


def _fingerprint(arr) -> str:
    a = np.ascontiguousarray(np.asarray(arr))
    return hashlib.sha1(a.tobytes()).hexdigest()[:16]


def regression_metrics(y_true, y_pred):
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(((yt - yp) ** 2).sum())
    ss_tot = float(((yt - yt.mean()) ** 2).sum())
    return (1.0 - ss_res / max(ss_tot, 1e-300),
            float(np.sqrt(ss_res / yt.size)),
            float(np.abs(yt - yp).mean()))


def make_subfolds(block_ids, n_sub, seed):
    """Assign rows to spatial sub-folds by fold-builder BLOCK, greedy-balanced,
    so tuning is validated under the same spatial shift as the outer folds."""
    rng = np.random.default_rng(seed)
    occ, cnt = np.unique(block_ids, return_counts=True)
    if occ.size < n_sub:
        return rng.integers(0, n_sub, size=block_ids.size)
    order = rng.permutation(occ.size)
    tot = np.zeros(n_sub, dtype=np.int64)
    mp = {}
    for i in order:
        s = int(np.argmin(tot))
        mp[int(occ[i])] = s
        tot[s] += cnt[i]
    return np.array([mp[int(b)] for b in block_ids], dtype=np.int8)


def iter_configs():
    """The Sigma design and the tree hyperparameters. Ranges belong with their
    orientation, so geometries are candidate dicts, never a cartesian product
    of independent lists."""
    designs = [{"Mode": "ANISO", "A_major": RANGE_MAJOR, "A_semi": RANGE_SEMI,
                "A_minor": RANGE_MINOR, "Azimuth": AZIMUTH_DEG,
                "Dip": DIP_DEG, "Tilt": TILT_DEG}]
    kernels = [KERNEL]
    if COMPARE_DESIGNS:
        designs.append({"Mode": "ISO", "A_major": RANGE_MAJOR,
                        "A_semi": RANGE_MAJOR, "A_minor": RANGE_MAJOR,
                        "Azimuth": 0.0, "Dip": 0.0, "Tilt": 0.0})
        kernels = list(dict.fromkeys(COMPARE_KERNELS + [KERNEL]))

    trees = list(product(N_ESTIMATORS_GRID, MAX_DEPTH_GRID, MIN_LEAF_GRID,
                         MAX_FEATURES_GRID, MAX_SAMPLES_GRID))
    for dsg in designs:
        for kern in kernels:
            for (n_est, md, ml, mf, ms) in trees:
                yield {**dsg, "Kernel": kern, "nugget": float(NUGGET),
                       "n_estimators": int(n_est), "max_depth": int(md),
                       "min_leaf": int(ml), "max_features": float(mf),
                       "max_samples": float(ms),
                       "ridge": float(RIDGE), "jitter": float(JITTER)}


def build_model(cfg: Dict[str, Any], seed: int) -> GLSRandomForest3D:
    """cfg may carry extra keys (older selected_configs.csv rows); they are
    ignored. Missing covariance keys fall back to the Section 2 declarations."""
    sk = {"Lx": cfg["A_major"], "Ly": cfg["A_semi"], "Lz": cfg["A_minor"],
          "kernel": cfg["Kernel"], "azimuth_deg": cfg["Azimuth"],
          "dip_deg": cfg["Dip"], "tilt_deg": cfg["Tilt"],
          "jitter": cfg.get("jitter", JITTER),
          "nugget": cfg.get("nugget", NUGGET)}
    fp = ForestParams(n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"],
                      min_leaf=cfg["min_leaf"], ridge=cfg["ridge"],
                      bootstrap=BOOTSTRAP, max_features=cfg["max_features"],
                      max_samples=cfg["max_samples"], random_state=seed,
                      n_jobs=N_JOBS, verbose=0)
    return GLSRandomForest3D(fp, sk)


def fit_predict(cfg, Xtr, ytr, Ctr, Xte, Cte, seed):
    """Returns (model, mean-only prediction, Eq.-11 kriged prediction)."""
    model = build_model(cfg, seed)
    model.fit(Xtr, ytr, Ctr)
    mean_pred = model.predict(Xte)
    krig_pred = model.predict_kriged(Xte, Cte, mean_pred=mean_pred)
    return model, mean_pred, krig_pred


def subcv_score(cfg, Xtr, ytr, Ctr, sub_ids, seed):
    """Pooled spatial sub-fold CV for one config, both prediction rules."""
    n = len(ytr)
    p_mean = np.full(n, np.nan)
    p_krig = np.full(n, np.nan)
    for s in range(int(sub_ids.max()) + 1):
        te = sub_ids == s
        tr = ~te
        if te.sum() == 0 or tr.sum() < 4 * int(cfg["min_leaf"]):
            continue
        _, pm, pk = fit_predict(cfg, Xtr[tr], ytr[tr], Ctr[tr],
                                Xtr[te], Ctr[te], seed + s)
        p_mean[te], p_krig[te] = pm, pk
    seen = np.isfinite(p_mean)
    if seen.sum() < 4:
        raise RuntimeError("too few sub-fold predictions")
    r2, rmse, mae = regression_metrics(ytr[seen], p_mean[seen])
    r2k, rmsek, _ = regression_metrics(ytr[seen], p_krig[seen])
    return {"SUBCV_R2": r2, "SUBCV_RMSE": rmse, "SUBCV_MAE": mae,
            "SUBCV_R2_K": r2k, "SUBCV_RMSE_K": rmsek,
            "SUBCV_n": int(seen.sum())}


def run_folds():
    run_dir = RUNS_BASE / (f"GLSRF_{DATA_PATH.stem}_"
                           + datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print(f"GLSRF v{GLSRF_VERSION} | {run_dir.name}")
    if SMOKE:
        print("*** --smoke: cut-down grid and tiny forests. NOT paper numbers. ***")
    print(f"Data : {DATA_PATH.name} (resolved from fold_config.json)")
    print(f"Sigma: nugget={NUGGET} kernel={KERNEL} ranges="
          f"{RANGE_MAJOR:g}/{RANGE_SEMI:g}/{RANGE_MINOR:g} "
          f"az/dip/tilt={AZIMUTH_DEG:g}/{DIP_DEG:g}/{TILT_DEG:g}")
    print(f"Features: {FEATURE_COLS}  (coordinates are NOT features)")
    print(f"Prediction rule: {'Eq. 11 mean + residual kriging' if USE_RESIDUAL_KRIGING else 'mean function only'}")
    print("=" * 72)

    df = (pd.read_excel(DATA_PATH) if DATA_PATH.suffix.lower() in (".xlsx", ".xls")
          else pd.read_csv(DATA_PATH, low_memory=False))
    for c in COORD_COLS + FEATURE_COLS + TARGET_COLS:
        if c not in df.columns:
            raise KeyError(f"Missing column {c} in {DATA_PATH}")
        df[c] = pd.to_numeric(df[c], errors="coerce")

    roles = np.load(ROLES_PATH)
    if roles.shape[0] != len(df):
        raise ValueError(f"roles rows ({roles.shape[0]}) != data rows ({len(df)}).")
    n_folds = roles.shape[1]

    fdf = pd.read_csv(FOLDS_CSV)
    d = np.abs(df[COORD_COLS].values - fdf[["x", "y", "z"]].values)
    if np.nanmax(d) > 0.01:
        raise ValueError("Coordinates do not match the folds CSV row-by-row "
                         f"(max diff {np.nanmax(d):.3f} m). Row order changed?")
    block_ids_all = fdf["block_id"].values

    coords_all = df[COORD_COLS].values.astype(np.float64)
    configs = list(iter_configs())
    fold_list = list(range(n_folds)) if FOLDS_TO_RUN is None else list(FOLDS_TO_RUN)
    print(f"Data: {len(df)} rows | folds: {n_folds} | grid: {len(configs)} configs "
          f"per fold x target | ranking on {RANK_COL}\n")

    long_rows, metric_rows, selected_rows = [], [], []
    t_start = time.time()

    for target in TARGET_COLS:
        y_all = df[target].values.astype(np.float64)

        for f in fold_list:
            fold_no = f + 1
            tr = (roles[:, f] == 0) & ~np.isnan(y_all)
            te = roles[:, f] == 1
            n_tr, n_te = int(tr.sum()), int(te.sum())
            print("-" * 72)
            print(f"TARGET {target} | FOLD {fold_no}/{n_folds} | train={n_tr} "
                  f"test={n_te} (buffer={int((roles[:, f] == 2).sum())})")

            med = df.loc[tr, FEATURE_COLS].median(numeric_only=True)
            X_all = df[FEATURE_COLS].fillna(med).values.astype(np.float64)
            Xtr, ytr, Ctr = X_all[tr], y_all[tr], coords_all[tr]
            Xte, Cte = X_all[te], coords_all[te]

            sub_ids = make_subfolds(block_ids_all[tr], N_SUBFOLDS,
                                    SUBFOLD_SEED + 1000 * fold_no)
            print(f"  spatial sub-folds (tuning): "
                  f"sizes={[int((sub_ids == s).sum()) for s in range(N_SUBFOLDS)]}")

            grid_rows, best = [], None
            t0 = time.time()
            for cfg_id, cfg in enumerate(configs):
                seed = SEED + f * 10_000 + cfg_id * 10
                try:
                    scores = subcv_score(cfg, Xtr, ytr, Ctr, sub_ids, seed)
                    row = {"config_id": cfg_id, **cfg, **scores}
                    if best is None or (np.nan_to_num(scores[RANK_COL], nan=-9e9)
                                        > np.nan_to_num(best[1][RANK_COL], nan=-9e9)):
                        best = (cfg, scores, seed)
                except Exception as e:                       # noqa: BLE001
                    row = {"config_id": cfg_id, **cfg, "error": str(e)[:200]}
                grid_rows.append(row)
                if (cfg_id + 1) % 10 == 0:
                    print(f"  [{target} f{fold_no}] {cfg_id + 1}/{len(configs)} "
                          f"configs ({time.time() - t0:.0f} s)")

            df_grid = (pd.DataFrame(grid_rows)
                       .sort_values(RANK_COL, ascending=False, na_position="last")
                       .reset_index(drop=True))
            df_grid.insert(0, "RankID", np.arange(1, len(df_grid) + 1))
            df_grid.to_csv(run_dir / f"grid_{target}_fold{fold_no}.csv", index=False)

            best_cfg, best_scores, best_seed = best
            print(f"  best: {best_cfg['Mode']}/{best_cfg['Kernel']} "
                  f"depth={best_cfg['max_depth']} min_leaf={best_cfg['min_leaf']} "
                  f"mf={best_cfg['max_features']} | {RANK_COL}="
                  f"{best_scores[RANK_COL]:.4f} (mean-only {best_scores['SUBCV_R2']:.4f})")

            # ---- refit the selected config, predict the held-out block ----
            model = build_model(best_cfg, best_seed)
            model.fit(Xtr, ytr, Ctr)
            oob_refit = model.oob_r2(ytr)
            tree_preds = model.predict_all(Xte)
            y_mean = tree_preds.mean(axis=0)
            y_unc = tree_preds.std(axis=0)
            y_krig = model.predict_kriged(Xte, Cte, mean_pred=y_mean)
            y_head = y_krig if USE_RESIDUAL_KRIGING else y_mean

            y_true_te = y_all[te]
            for i, r_id in enumerate(np.where(te)[0]):
                long_rows.append({
                    "row": int(r_id), "x": coords_all[r_id, 0],
                    "y": coords_all[r_id, 1], "z": coords_all[r_id, 2],
                    "fold": fold_no, "target": target, "method": METHOD_NAME,
                    "y_true": y_true_te[i], "y_pred": y_head[i],
                    "y_pred_mean": y_mean[i], "y_pred_krig": y_krig[i],
                    "y_unc": float(y_unc[i])})

            ok = ~np.isnan(y_true_te)
            if ok.sum() >= 2:
                r2, rmse, mae = regression_metrics(y_true_te[ok], y_mean[ok])
                r2k, rmsek, maek = regression_metrics(y_true_te[ok], y_krig[ok])
                print(f"  TEST fold {fold_no}: mean R2={r2:.4f} | "
                      f"kriged R2={r2k:.4f} RMSE={rmsek:.4f}")
            else:
                r2 = rmse = mae = r2k = rmsek = maek = np.nan

            metric_rows.append({
                "target": target, "fold": fold_no, "method": METHOD_NAME,
                "n_train": n_tr, "n_test": int(ok.sum()),
                "Mode": best_cfg["Mode"], "Kernel": best_cfg["Kernel"],
                "nugget": best_cfg["nugget"],
                **{k: _to_py(v) for k, v in oob_refit.items()},
                "TEST_R2": r2, "TEST_RMSE": rmse, "TEST_MAE": mae,
                "TEST_R2_K": r2k, "TEST_RMSE_K": rmsek, "TEST_MAE_K": maek,
                **{f"BEST_{k}": v for k, v in best_scores.items()}})
            selected_rows.append({"target": target, "fold": fold_no,
                                  **{k: _to_py(v) for k, v in best_cfg.items()},
                                  "seed": best_seed})

    elapsed = time.time() - t_start

    df_long = pd.DataFrame(long_rows)
    df_long.to_csv(run_dir / f"predictions_long_{METHOD_NAME}.csv", index=False)
    df_metrics = pd.DataFrame(metric_rows)
    df_metrics.to_csv(run_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(run_dir / "selected_configs.csv", index=False)

    with open(run_dir / "run_config.json", "w", encoding="utf-8") as fjs:
        json.dump({
            "created_at": datetime.now().isoformat(),
            "glsrf_version": GLSRF_VERSION, "method": METHOD_NAME,
            "data_path": str(DATA_PATH), "roles_path": str(ROLES_PATH),
            "n_rows": int(len(df)),
            "coord_fingerprint_sha1": _fingerprint(coords_all),
            "roles_fingerprint_sha1": _fingerprint(roles),
            "coord_cols": COORD_COLS, "feature_cols": FEATURE_COLS,
            "target_cols": TARGET_COLS,
            "coords_as_features": False,
            # Everything that changes what the model IS:
            "nugget": NUGGET, "kernel": KERNEL,
            "azimuth_deg": AZIMUTH_DEG, "dip_deg": DIP_DEG, "tilt_deg": TILT_DEG,
            "range_major": RANGE_MAJOR, "range_semi": RANGE_SEMI,
            "range_minor": RANGE_MINOR,
            "ridge": RIDGE, "jitter": JITTER,
            "compare_designs": COMPARE_DESIGNS,
            "compare_kernels": COMPARE_KERNELS if COMPARE_DESIGNS else [KERNEL],
            "residual_kriging": USE_RESIDUAL_KRIGING,
            "bootstrap": BOOTSTRAP,
            "n_configs_per_fold": len(configs),
            "rank_col": RANK_COL,
            "n_subfolds": N_SUBFOLDS, "subfold_seed": int(SUBFOLD_SEED),
            "seed": int(SEED), "smoke": bool(SMOKE),
            "elapsed_sec": elapsed,
        }, fjs, indent=2)

    # ---- pooled test metrics: every sample predicted exactly once ----
    pooled_rows = []
    for target in TARGET_COLS:
        sl = df_long[df_long["target"] == target].dropna(subset=["y_true"])
        if len(sl) <= 2:
            continue
        yt = sl["y_true"].values.astype(np.float64)
        row = {"target": target, "method": METHOD_NAME, "n": len(sl)}
        for tag, col in (("", "y_pred_mean"), ("_K", "y_pred_krig")):
            yp = sl[col].values.astype(np.float64)
            ss_res = float(((yt - yp) ** 2).sum())
            ss_tot = float(((yt - yt.mean()) ** 2).sum())
            cov = float(((yt - yt.mean()) * (yp - yp.mean())).mean())
            row[f"POOLED_R2{tag}"] = 1.0 - ss_res / max(ss_tot, 1e-300)
            row[f"POOLED_RMSE{tag}"] = float(np.sqrt(ss_res / yt.size))
            row[f"POOLED_MAE{tag}"] = float(np.abs(yt - yp).mean())
            row[f"POOLED_CCC{tag}"] = float(
                2.0 * cov / (yt.var() + yp.var() + (yt.mean() - yp.mean()) ** 2))
        pooled_rows.append(row)
    if pooled_rows:
        pd.DataFrame(pooled_rows).to_csv(run_dir / "pooled_summary.csv", index=False)

    print("\n" + "=" * 72)
    print("SUMMARY  (POOLED = headline; _K = with Eq. 11 residual kriging)")
    for target in TARGET_COLS:
        sub = df_metrics[df_metrics["target"] == target]
        pr = [r for r in pooled_rows if r["target"] == target]
        if len(sub) == 0 or not pr:
            continue
        print(f"  {target:9s} | mean sub-CV {sub['BEST_' + RANK_COL].mean():.4f} "
              f"| mean OOB {sub['OOB_R2'].mean():.4f} "
              f"| POOLED R2 mean-only {pr[0]['POOLED_R2']:.4f} "
              f"kriged {pr[0]['POOLED_R2_K']:.4f} "
              f"(RMSE {pr[0]['POOLED_RMSE_K']:.3f}, CCC {pr[0]['POOLED_CCC_K']:.4f})")
    print(f"\nRun folder : {run_dir}")
    print(f"Elapsed    : {elapsed / 60:.1f} min")
    print("=" * 72)
    return run_dir


# ==============================================================
# PART E — full block-model prediction (deployment)
# ==============================================================

def predict_full_block(run_dir: Path, target: str,
                       full_grid_path: Path = FULL_GRID_PATH,
                       batch: int = FULL_GRID_BATCH) -> Path:
    """Refit the config with the best MEAN sub-CV score across folds on every
    labelled row, then apply it to the full estimated grid.

    Selection uses the across-fold mean, not a single fold's winner: a config
    that happens to suit one fold is not the deployment model.
    """
    run_dir = Path(run_dir)
    grids = sorted(run_dir.glob(f"grid_{target}_fold*.csv"))
    if not grids:
        raise FileNotFoundError(f"No grid tables for {target} in {run_dir}")
    key = ["Mode", "Kernel", "n_estimators", "max_depth", "min_leaf",
           "max_features", "max_samples"]
    allg = pd.concat([pd.read_csv(g) for g in grids], ignore_index=True)
    ranked = (allg.dropna(subset=[RANK_COL]).groupby(key, as_index=False)[RANK_COL]
              .mean().sort_values(RANK_COL, ascending=False))
    if ranked.empty:
        raise RuntimeError("No scored configs to select from.")
    top = ranked.iloc[0]
    cfg = next(c for c in iter_configs()
               if all(np.isclose(c[k], top[k]) if isinstance(top[k], (int, float, np.floating))
                      else c[k] == top[k] for k in key))
    print(f"[deploy {target}] {top['Mode']}/{top['Kernel']} depth={top['max_depth']} "
          f"min_leaf={top['min_leaf']} | mean {RANK_COL}={top[RANK_COL]:.4f}")

    df = (pd.read_excel(DATA_PATH) if DATA_PATH.suffix.lower() in (".xlsx", ".xls")
          else pd.read_csv(DATA_PATH, low_memory=False))
    for c in COORD_COLS + FEATURE_COLS + [target]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    tr = ~df[target].isna()
    med = df.loc[tr, FEATURE_COLS].median(numeric_only=True)
    Xtr = df.loc[tr, FEATURE_COLS].fillna(med).values.astype(np.float64)
    ytr = df.loc[tr, target].values.astype(np.float64)
    Ctr = df.loc[tr, COORD_COLS].values.astype(np.float64)

    model = build_model(cfg, SEED)
    model.fit(Xtr, ytr, Ctr)
    print(f"  fitted on {len(ytr)} rows | OOB R2={model.oob_r2(ytr)['OOB_R2']:.4f}")

    full = pd.read_csv(full_grid_path, low_memory=False)
    lower = {c.lower(): c for c in full.columns}
    ren = {lower[c.lower()]: c for c in COORD_COLS + FEATURE_COLS
           if c.lower() in lower and lower[c.lower()] != c}
    if ren:
        full = full.rename(columns=ren)
        print(f"  renamed grid columns: {ren}")
    missing = [c for c in COORD_COLS + FEATURE_COLS if c not in full.columns]
    if missing:
        raise KeyError(f"Full grid is missing columns: {missing}")
    for c in COORD_COLS + FEATURE_COLS:
        full[c] = pd.to_numeric(full[c], errors="coerce")
    full = full.dropna(subset=COORD_COLS).reset_index(drop=True)

    n = len(full)
    out_mean = np.empty(n)
    out_krig = np.empty(n)
    out_unc = np.empty(n)
    print(f"  predicting {n} grid cells (batch {batch})...")
    for s0 in range(0, n, batch):
        s1 = min(s0 + batch, n)
        Xb = full.loc[s0:s1 - 1, FEATURE_COLS].fillna(med).values.astype(np.float64)
        Cb = full.loc[s0:s1 - 1, COORD_COLS].values.astype(np.float64)
        tp = model.predict_all(Xb)
        out_mean[s0:s1] = tp.mean(axis=0)
        out_unc[s0:s1] = tp.std(axis=0)
        out_krig[s0:s1] = model.predict_kriged(Xb, Cb, mean_pred=out_mean[s0:s1])
        print(f"    {s1}/{n}")

    out_dir = run_dir / "full_prediction"
    out_dir.mkdir(exist_ok=True)
    res = full[COORD_COLS].copy()
    res[f"{target}_pred"] = out_krig if USE_RESIDUAL_KRIGING else out_mean
    res[f"{target}_pred_mean"] = out_mean
    res[f"{target}_pred_krig"] = out_krig
    res[f"{target}_spread"] = out_unc
    path = out_dir / f"PRED_{target}.csv"
    res.to_csv(path, index=False)
    print(f"  wrote {path}")
    return path


if __name__ == "__main__":
    run_folds()
