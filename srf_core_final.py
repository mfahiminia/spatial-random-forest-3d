# -*- coding: utf-8 -*-
"""
srf_core_final — stable, import-only module with the SRF classes.

Based on: Talebi et al. (2021) "A Truly Spatial Random Forests Algorithm for
Geoscience Data Analysis and Modelling", Math Geosci, extended from 2D pixels
to 3D voxel kernels.

This module has NO side effects (nothing runs on import). Training lives in
srf_train.py. joblib models pickled against this module stay loadable as long
as this file keeps its name and class names.

FEATURE CONTRACT (locked):
- X row = one vectorised 3D pattern: k^3 voxels, voxel-major, props contiguous
  per voxel: [vox0:(FE,FEO,Magsus,S,MASK), vox1:(...), ...].
  Voxel index order = i (X offset) slowest, then j (Y), then kk (Z):
  vox = i*k^2 + j*k + kk with offsets (i-half, j-half, kk-half).
- Mask channel is LAST. Aniso features are computed on the covariate props only.
- Aniso feats per prop are exactly 8, in this order:
  0 wmean, 1 wvar, 2 major_mean, 3 major_var,
  4 semi_mean, 5 semi_var, 6 minor_mean, 7 minor_var

PAPER MAPPING:
- best_split_sse            -> Eq (1), split minimising n_L*Q_L + n_R*Q_R
- best_split_gini           -> Eq (2), Gini impurity split (classification)
- SpatialRandomForest OOB   -> Eq (4) regression MSE_OOB / Eq (5) error rate
- normalized_entropy        -> Eq (3), local uncertainty for classification
- permutation_importance    -> Sect 2.3 predictor importance
- reshape_base_importance_to_zones -> "zone of influence" per variable
- Z-rotation augmentation   -> Sect 2.1 / Fig 1(f), 90-degree rotations about Z.
  DELIBERATE DEVIATION: the paper concatenates the 4 rotated copies as new
  observations and bootstraps over 4N rows. Here the bootstrap draws ORIGINAL
  pattern ids and picks a random rotation per draw, and OOB status is tracked
  at the original-pattern level ("group-honest OOB"). Naive 4N bootstrapping
  lets a rotated copy of an in-bag pattern count as OOB, which leaks and makes
  OOB optimistic — the paper's own Discussion warns about over-optimism from
  nonspatial resampling, so we keep OOB honest here.
"""

import numpy as np

# Bump this whenever the public API of this module grows. Consumer scripts
# (srf_train.py, srf_predict.py) force-reload stale in-session copies (e.g. a
# Jupyter kernel that imported an older version) when this number is too low.
CORE_API_VERSION = 8


# ============================================================
# 1) Rotations (feature index maps)
# ============================================================

def generate_z_rotations_4():
    """Identity + 90/180/270-degree rotations about the vertical (Z) axis.

    These are the only rotations used. The 24 proper rotations of the cube are
    not applicable here: they map Z onto X/Y, which is physically valid only
    for isotropic voxels, and CELL_SIZE is (5, 5, 2).

    Safe for anisotropic voxels as long as dx == dy. Identity is FIRST."""
    return [
        ((0, 1, 2), (1, 1, 1)),
        ((1, 0, 2), (1, -1, 1)),
        ((0, 1, 2), (-1, -1, 1)),
        ((1, 0, 2), (-1, 1, 1)),
    ]


def build_rotation_index_maps(k, n_props, rotations):
    """For each rotation, an index map over the k^3*n_props flat feature vector
    such that X[:, map] is the pattern as seen under that rotation."""
    if k % 2 != 1:
        raise ValueError("k must be odd (3,5,7,...)")

    half = k // 2
    maps = []

    def rot_coord(v, perm, flips):
        vv = v[list(perm)]
        return np.array([flips[0] * vv[0], flips[1] * vv[1], flips[2] * vv[2]], dtype=int)

    coord_to_vox = {}
    idx = 0
    for i in range(k):
        for j in range(k):
            for kk in range(k):
                coord_to_vox[(i - half, j - half, kk - half)] = idx
                idx += 1

    for (perm, flips) in rotations:
        rot_map_vox = np.empty(k ** 3, dtype=np.int64)
        out_idx = 0
        for iR in range(k):
            for jR in range(k):
                for kR in range(k):
                    vR = np.array([iR - half, jR - half, kR - half], dtype=int)

                    inv_perm = np.argsort(perm)
                    inv_flips = (flips[inv_perm[0]], flips[inv_perm[1]], flips[inv_perm[2]])
                    vO = rot_coord(vR, tuple(inv_perm.tolist()), tuple(inv_flips))

                    rot_map_vox[out_idx] = coord_to_vox[tuple(vO.tolist())]
                    out_idx += 1

        full_map = np.empty((k ** 3) * n_props, dtype=np.int64)
        for vR in range(k ** 3):
            vO = rot_map_vox[vR]
            baseR = vR * n_props
            baseO = vO * n_props
            full_map[baseR:baseR + n_props] = np.arange(baseO, baseO + n_props, dtype=np.int64)

        maps.append(full_map)

    return maps


def voxel_maps_from_feature_maps(feature_maps, n_props):
    """Recover per-voxel index maps (length k^3) from full feature maps."""
    return [m[::n_props] // n_props for m in feature_maps]


# ============================================================
# 2) Anisotropy weights / masks / features
# ============================================================

def build_rotation_matrix_from_azimuth_dip(azimuth_deg, dip_deg):
    az = np.deg2rad(azimuth_deg)
    dip = np.deg2rad(dip_deg)

    v1 = np.array([
        np.sin(az) * np.cos(dip),
        np.cos(az) * np.cos(dip),
        -np.sin(dip),
    ], dtype=np.float64)
    v1 /= max(np.linalg.norm(v1), 1e-12)

    ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(np.dot(ref, v1)) > 0.99:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    v2 = np.cross(ref, v1)
    v2 /= max(np.linalg.norm(v2), 1e-12)

    v3 = np.cross(v1, v2)
    v3 /= max(np.linalg.norm(v3), 1e-12)

    return np.vstack([v1, v2, v3])


def build_aniso_weights_and_masks(k, cell_size, a_major, a_semi, a_minor,
                                  azimuth_deg, dip_deg, sharpness):
    dx, dy, dz = cell_size
    half = k // 2

    ii, jj, kk = np.meshgrid(
        np.arange(-half, half + 1),
        np.arange(-half, half + 1),
        np.arange(-half, half + 1),
        indexing="ij",
    )
    hx = ii.astype(np.float64) * dx
    hy = jj.astype(np.float64) * dy
    hz = kk.astype(np.float64) * dz
    h = np.stack([hx, hy, hz], axis=-1)

    R = build_rotation_matrix_from_azimuth_dip(azimuth_deg, dip_deg)
    S = np.diag([1.0 / a_major, 1.0 / a_semi, 1.0 / a_minor]).astype(np.float64)

    h_flat = h.reshape(-1, 3).T

    u = (R @ h_flat).T
    up = (S @ (R @ h_flat)).T
    r = np.sqrt(np.sum(up * up, axis=1)).astype(np.float32)

    w = np.exp(-sharpness * r).astype(np.float32)
    w = np.maximum(w, 1e-12).astype(np.float32)

    au = np.abs(u[:, 0]); av = np.abs(u[:, 1]); aw = np.abs(u[:, 2])
    major_mask = ((au >= av) & (au >= aw)).astype(np.float32)
    semi_mask = ((av > au) & (av >= aw)).astype(np.float32)
    minor_mask = ((aw > au) & (aw > av)).astype(np.float32)
    s = major_mask + semi_mask + minor_mask
    major_mask = np.where(s == 0, 1.0, major_mask).astype(np.float32)

    return w, {"major": major_mask, "semi": semi_mask, "minor": minor_mask}


def rotate_voxel_flats(flat, rot_maps_vox):
    return [flat[vox_map].astype(np.float32, copy=False) for vox_map in rot_maps_vox]


ANISO_FEAT_NAMES_8 = [
    "wmean", "wvar",
    "major_mean", "major_var",
    "semi_mean", "semi_var",
    "minor_mean", "minor_var",
]


def build_aniso_features(X_rot, w_flat, m_major, m_semi, m_minor,
                         k, n_props_total, n_props_aniso):
    """Weighted local stats along variogram axes for the covariate props
    (mask excluded). Returns (B, n_props_aniso*8) float32."""
    k3 = k ** 3
    B = X_rot.shape[0]

    Xv = X_rot.reshape(B, k3, n_props_total).astype(np.float32, copy=False)
    Xv = Xv[:, :, :n_props_aniso]  # EXCLUDE MASK channel

    w = w_flat.reshape(1, k3)
    w_major = w * m_major.reshape(1, k3)
    w_semi = w * m_semi.reshape(1, k3)
    w_minor = w * m_minor.reshape(1, k3)

    sw = max(float(np.sum(w)), 1e-12)
    swm = max(float(np.sum(w_major)), 1e-12)
    sws = max(float(np.sum(w_semi)), 1e-12)
    swn = max(float(np.sum(w_minor)), 1e-12)

    feats = np.empty((B, n_props_aniso * 8), dtype=np.float32)

    for p in range(n_props_aniso):
        x = Xv[:, :, p]

        wmean = (x * w).sum(axis=1) / sw
        xc = x - wmean.reshape(-1, 1)
        wvar = (xc * xc * w).sum(axis=1) / sw

        maj_mean = (x * w_major).sum(axis=1) / swm
        xc_m = x - maj_mean.reshape(-1, 1)
        maj_var = (xc_m * xc_m * w_major).sum(axis=1) / swm

        semi_mean = (x * w_semi).sum(axis=1) / sws
        xc_s = x - semi_mean.reshape(-1, 1)
        semi_var = (xc_s * xc_s * w_semi).sum(axis=1) / sws

        min_mean = (x * w_minor).sum(axis=1) / swn
        xc_n = x - min_mean.reshape(-1, 1)
        min_var = (xc_n * xc_n * w_minor).sum(axis=1) / swn

        base = p * 8
        feats[:, base + 0] = wmean
        feats[:, base + 1] = wvar
        feats[:, base + 2] = maj_mean
        feats[:, base + 3] = maj_var
        feats[:, base + 4] = semi_mean
        feats[:, base + 5] = semi_var
        feats[:, base + 6] = min_mean
        feats[:, base + 7] = min_var

    return feats


def build_feature_names(k, props_total, props_cov, use_aniso_feats):
    """Names for the extended design matrix: voxel-major base features
    ('FE@(-2,-1,0)') then aniso features ('FE::major_var')."""
    half = k // 2
    names = []
    for i in range(k):
        for j in range(k):
            for kk in range(k):
                for p in props_total:
                    names.append(f"{p}@({i - half},{j - half},{kk - half})")
    if use_aniso_feats:
        for p in props_cov:
            for fn in ANISO_FEAT_NAMES_8:
                names.append(f"{p}::{fn}")
    return names


# ============================================================
# 3) Split criteria
# ============================================================

def best_split_sse_fast(X, y, indices, feature_subset, min_leaf=50, split_quantiles=None):
    """Vectorised regression split search (same criterion as best_split_sse,
    Eq 1): sorts ALL candidate features in one argsort call and evaluates all
    change-points with array ops instead of a per-feature Python loop.
    ~5-10x faster on wide pattern matrices.

    Note: quantile subsampling keeps the split_quantiles semantics but applies
    one evenly-spaced row mask to all features (the original picked quantiles
    per feature — practically equivalent)."""
    idx = indices
    n = idx.size
    if n < 2 * min_leaf:
        return None

    Xs = X[np.ix_(idx, feature_subset)].astype(np.float64, copy=False)   # (n, m)
    y_sub = y[idx].astype(np.float64, copy=False)
    y_sum = y_sub.sum()
    y2_sum = float((y_sub * y_sub).sum())
    sse_parent = y2_sum - (y_sum ** 2) / n

    order = np.argsort(Xs, axis=0, kind="stable")
    x_sorted = np.take_along_axis(Xs, order, axis=0)
    y_sorted = y_sub[order]                                              # (n, m)

    csum_y = np.cumsum(y_sorted, axis=0)
    csum_y2 = np.cumsum(y_sorted * y_sorted, axis=0)

    nL = np.arange(1, n, dtype=np.float64)[:, None]                      # (n-1, 1)
    syL = csum_y[:-1]
    sy2L = csum_y2[:-1]
    sseL = sy2L - (syL * syL) / nL
    nR = float(n) - nL
    syR = y_sum - syL
    sseR = (y2_sum - sy2L) - (syR * syR) / nR
    total = sseL + sseR                                                  # (n-1, m)

    valid = (x_sorted[1:] != x_sorted[:-1]) \
        & (nL >= min_leaf) & (nL <= n - min_leaf)

    if (split_quantiles is not None) and (split_quantiles > 0) and (n - 1 > split_quantiles):
        keep = np.zeros(n - 1, dtype=bool)
        qs = np.linspace(0.0, 1.0, split_quantiles + 2)[1:-1]
        keep[np.unique((qs * (n - 2)).astype(int))] = True
        valid &= keep[:, None]

    if not valid.any():
        return None
    total = np.where(valid, total, np.inf)

    m = total.shape[1]
    for _ in range(5):   # retry a few times if a tie violates min_leaf counts
        flat = int(np.argmin(total))
        r, c = flat // m, flat % m
        if not np.isfinite(total[r, c]):
            return None
        thr = (x_sorted[r, c] + x_sorted[r + 1, c]) / 2.0
        nl = int((Xs[:, c] <= thr).sum())
        if (nl >= min_leaf) and (n - nl >= min_leaf):
            return {"feature": int(feature_subset[c]),
                    "threshold": float(thr),
                    "gain": float(sse_parent - total[r, c])}
        total[r, c] = np.inf
    return None


def best_split_gini(X, y_codes, n_classes, indices, min_leaf=50,
                    feature_subset=None, split_quantiles=None):
    """Classification split, paper Eq (2): minimise n_L*G_L + n_R*G_R.
    Uses the identity n*G = n - sum(counts^2)/n for cumulative evaluation."""
    idx = indices
    n = idx.size
    if n < 2 * min_leaf:
        return None

    y_sub = y_codes[idx]
    counts_parent = np.bincount(y_sub, minlength=n_classes).astype(np.float64)
    q_parent = float(n) - float((counts_parent ** 2).sum()) / float(n)

    if feature_subset is None:
        feature_subset = np.arange(X.shape[1], dtype=np.int64)

    best = None

    for f in feature_subset:
        x = X[idx, f]
        order = np.argsort(x, kind="mergesort")
        x_sorted = x[order]
        y_sorted = y_sub[order]

        onehot = np.zeros((n, n_classes), dtype=np.float64)
        onehot[np.arange(n), y_sorted] = 1.0
        csum = np.cumsum(onehot, axis=0)

        change = np.where(np.diff(x_sorted) != 0)[0] + 1
        if change.size == 0:
            continue

        valid = change[(change >= min_leaf) & (change <= n - min_leaf)]
        if valid.size == 0:
            continue

        if (split_quantiles is not None) and (split_quantiles > 0) and (valid.size > split_quantiles):
            qs = np.linspace(0.0, 1.0, split_quantiles + 2, endpoint=True)[1:-1]
            sel = np.unique((qs * (valid.size - 1)).astype(int))
            valid = valid[sel]

        cL = csum[valid - 1]                       # (m, K)
        nL = valid.astype(np.float64)
        termL = nL - (cL * cL).sum(axis=1) / nL    # n_L * G_L

        cR = counts_parent[None, :] - cL
        nR = float(n) - nL
        termR = nR - (cR * cR).sum(axis=1) / nR    # n_R * G_R

        q_total = termL + termR
        i_best = int(np.argmin(q_total))
        split_pos = int(valid[i_best])
        thr = (x_sorted[split_pos - 1] + x_sorted[split_pos]) / 2.0
        gain = float(q_parent - q_total[i_best])

        mask_left = x <= thr
        if (mask_left.sum() >= min_leaf) and ((~mask_left).sum() >= min_leaf):
            if (best is None) or (gain > best["gain"]):
                best = {"feature": int(f), "threshold": float(thr), "gain": gain}

    return best


def normalized_entropy(proba):
    """Paper Eq (3): H(u) in [0,1] from predicted class fractions (B, K)."""
    K = proba.shape[1]
    if K < 2:
        return np.zeros(proba.shape[0], dtype=np.float32)
    p = np.clip(proba, 1e-12, 1.0)
    H = -(p * np.log(p)).sum(axis=1) / np.log(K)
    return H.astype(np.float32)


# ============================================================
# 4) Unified spatial decision tree (regression + classification)
# ============================================================

class SpatialDecisionTree(object):
    """CART on vectorised spatial patterns. Flat-array storage; vectorised
    predict (no per-row Python recursion)."""

    def __init__(self, max_depth=6, min_leaf=50, max_features="sqrt",
                 min_gain=1e-6, split_quantiles=None, random_state=None,
                 task="regression", n_classes=None):
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.max_features = max_features
        self.min_gain = min_gain
        self.split_quantiles = split_quantiles
        self.rng = np.random.default_rng(random_state)
        self.task = task
        self.n_classes = n_classes

        self.feature_arr = None
        self.threshold_arr = None
        self.left_arr = None
        self.right_arr = None
        self.value_arr = None      # (n_nodes,) reg | (n_nodes, K) clf proportions
        self.is_leaf_arr = None

    def _resolve_max_features(self, D):
        mf = self.max_features
        if (mf is None) or (mf == "all"):
            return D
        if isinstance(mf, str):
            if mf == "sqrt":
                return max(1, int(np.sqrt(D)))
            if mf == "third":       # paper default for regression: P/3
                return max(1, int(np.ceil(D / 3.0)))
            raise ValueError(f"Unknown max_features string: {mf}")
        if isinstance(mf, float):
            if 0.0 < mf <= 1.0:
                return max(1, int(np.ceil(mf * D)))
            raise ValueError("max_features float must be in (0, 1].")
        if isinstance(mf, int):
            return max(1, min(mf, D))
        return D

    def fit(self, X, y):
        N, D = X.shape
        mtry = self._resolve_max_features(D)
        is_clf = (self.task == "classification")
        K = int(self.n_classes) if is_clf else 0

        feat_l, thr_l, left_l, right_l, leaf_l, val_l = [], [], [], [], [], []

        def add_node(indices):
            # Never silently: an empty node means a split sent every row one way,
            # and its leaf value would be a NaN that poisons the whole forest's
            # predictions (and the config's grid score) without raising.
            if indices.size == 0:
                raise RuntimeError(
                    "Empty node: a split produced a child with no samples.")
            nid = len(feat_l)
            feat_l.append(-1)
            thr_l.append(0.0)
            left_l.append(-1)
            right_l.append(-1)
            leaf_l.append(True)
            if is_clf:
                cnt = np.bincount(y[indices], minlength=K).astype(np.float64)
                val_l.append(cnt / max(cnt.sum(), 1.0))
            else:
                val_l.append(float(y[indices].mean()))
            return nid

        def build(indices, depth):
            nid = add_node(indices)

            if (depth >= self.max_depth) or (indices.size < 2 * self.min_leaf):
                return nid

            feat_subset = self.rng.choice(D, size=mtry, replace=False)
            if is_clf:
                split = best_split_gini(
                    X, y, K, indices,
                    min_leaf=self.min_leaf,
                    feature_subset=feat_subset,
                    split_quantiles=self.split_quantiles,
                )
            else:
                split = best_split_sse_fast(
                    X, y, indices, feat_subset,
                    min_leaf=self.min_leaf,
                    split_quantiles=self.split_quantiles,
                )
            if (split is None) or (split["gain"] < self.min_gain):
                return nid

            f = split["feature"]
            thr = split["threshold"]

            # np.float64(thr), not the bare Python float: under NumPy 2 (NEP 50)
            # a Python scalar is "weak" and gets cast DOWN to the array's float32,
            # which collapses thr onto an adjacent float32 value. The split search
            # picks thresholds in float64 and _terminal_nodes traverses in float64
            # (threshold_arr is a float64 array), so a weak comparison here made
            # fit and predict disagree — and could send every row of a node one
            # way, producing an empty child whose mean is NaN.
            x_node = X[indices, f]
            mask_left = x_node <= np.float64(thr)
            left_idx = indices[mask_left]
            right_idx = indices[~mask_left]

            feat_l[nid] = f
            thr_l[nid] = thr
            leaf_l[nid] = False
            left_l[nid] = build(left_idx, depth + 1)
            right_l[nid] = build(right_idx, depth + 1)
            return nid

        build(np.arange(N, dtype=np.int64), 0)

        self.feature_arr = np.asarray(feat_l, dtype=np.int64)
        self.threshold_arr = np.asarray(thr_l, dtype=np.float64)
        self.left_arr = np.asarray(left_l, dtype=np.int64)
        self.right_arr = np.asarray(right_l, dtype=np.int64)
        self.is_leaf_arr = np.asarray(leaf_l, dtype=bool)
        if is_clf:
            self.value_arr = np.asarray(val_l, dtype=np.float64)   # (n_nodes, K)
        else:
            self.value_arr = np.asarray(val_l, dtype=np.float64)   # (n_nodes,)
        return self

    def _terminal_nodes(self, X):
        N = X.shape[0]
        cur = np.zeros(N, dtype=np.int64)
        while True:
            active = ~self.is_leaf_arr[cur]
            if not active.any():
                break
            ai = np.where(active)[0]
            ca = cur[ai]
            f = self.feature_arr[ca]
            t = self.threshold_arr[ca]
            go_left = X[ai, f] <= t
            cur[ai] = np.where(go_left, self.left_arr[ca], self.right_arr[ca])
        return cur

    def predict(self, X):
        if X.shape[0] == 0:
            return np.array([], dtype=np.float32)
        nodes = self._terminal_nodes(X)
        if self.task == "classification":
            return np.argmax(self.value_arr[nodes], axis=1).astype(np.int64)
        return self.value_arr[nodes].astype(np.float32)

    def predict_proba(self, X):
        if self.task != "classification":
            raise RuntimeError("predict_proba is only available for classification.")
        if X.shape[0] == 0:
            return np.zeros((0, int(self.n_classes)), dtype=np.float64)
        nodes = self._terminal_nodes(X)
        return self.value_arr[nodes]


# ============================================================
# 5) Unified spatial random forest
# ============================================================

class SpatialRandomForest(object):
    """SRF for regression and classification with:
    - optional Z-rotation augmentation (group-honest OOB, see module docstring)
    - optional anisotropy features
    - OOB error (Eq 4 / Eq 5)
    - permutation importance (Sect 2.3) and zone-of-influence reshaping.
    Geometry (k, n_props_total, n_props_aniso) is fixed at construction."""

    def __init__(self, n_estimators=200, max_depth=8, min_leaf=20,
                 max_features="sqrt", max_samples=None, min_gain=1e-6,
                 split_quantiles=None, bootstrap=True, random_state=42,
                 task="regression", n_classes=None, class_balance=None,
                 k=5, n_props_total=5, n_props_aniso=4,
                 n_static_features=0,
                 rot_feature_maps=None, use_augmentation=False,
                 use_aniso_feats=False,
                 aniso_w_flats=None, aniso_masks_flats=None,
                 aniso_var_idx=None, aniso_var_mu=None, aniso_var_sigma=None,
                 aniso_var_log1p=True, aniso_var_zscore=True):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.max_features = max_features
        self.max_samples = max_samples
        self.min_gain = min_gain
        self.split_quantiles = split_quantiles
        self.bootstrap = bootstrap
        # "balanced": each tree draws an equal number of rows PER CLASS. The
        # splitter is untouched; only the resample changes. Needed because the
        # high-grade class is ~9% of samples, so an unweighted Gini forest can
        # minimise impurity while never isolating it.
        self.class_balance = class_balance
        self.random_state = random_state

        self.task = task
        self.n_classes = n_classes
        self.classes_ = None          # original labels (classification)

        self.k = int(k)
        self.n_props_total = int(n_props_total)
        self.n_props_aniso = int(n_props_aniso)
        self.n_static_features = int(n_static_features)
        if self.n_static_features < 0:
            raise ValueError("n_static_features cannot be negative.")

        self.rot_feature_maps = rot_feature_maps
        self.use_augmentation = bool(use_augmentation)

        self.use_aniso_feats = bool(use_aniso_feats)
        self.aniso_w_flats = aniso_w_flats
        self.aniso_masks_flats = aniso_masks_flats
        self.aniso_var_idx = aniso_var_idx
        self.aniso_var_mu = aniso_var_mu
        self.aniso_var_sigma = aniso_var_sigma
        self.aniso_var_log1p = bool(aniso_var_log1p)
        self.aniso_var_zscore = bool(aniso_var_zscore)

        self.trees_ = []
        self.oob_pred_ = None         # reg: values | clf: label codes
        self.oob_proba_ = None        # clf only
        self._oob_ids_ = []           # per-tree ORIGINAL-pattern OOB ids

        self.meta_ = None

    # ---- sampling / transform -------------------------------------------

    def _resolve_max_samples(self, N):
        ms = self.max_samples
        if ms is None:
            return N
        if isinstance(ms, float):
            if 0.0 < ms <= 1.0:
                return max(1, int(np.ceil(ms * N)))
            raise ValueError("max_samples float must be in (0, 1].")
        if isinstance(ms, int):
            return max(1, min(ms, N))
        return N

    def _transform_X(self, X, rot_id):
        """Rotate only the local pattern, preserve static features, then append
        the anisotropy summaries for the selected rotation.

        Static features include the directional multi-scale block.  Rotating
        them as voxel columns would be both dimensionally invalid and
        geologically wrong: their names are tied to fixed major/semi/minor
        directions.
        """
        base_width = (self.k ** 3) * self.n_props_total
        n_static = int(getattr(self, "n_static_features", 0))
        expected_width = base_width + n_static
        if X.ndim != 2 or X.shape[1] != expected_width:
            raise ValueError(
                f"Input feature width is {X.shape[1] if X.ndim == 2 else '?'}; "
                f"expected local {base_width} + static {n_static} = "
                f"{expected_width}.")
        X_base = X[:, :base_width]
        X_static = X[:, base_width:] if n_static else None

        if self.rot_feature_maps is None:
            X_rot = X_base
        else:
            X_rot = X_base[:, self.rot_feature_maps[rot_id]]
        X_rot = X_rot.astype(np.float32, copy=False)

        if not self.use_aniso_feats:
            if X_static is None:
                return X_rot
            return np.concatenate(
                [X_rot, X_static.astype(np.float32, copy=False)], axis=1)

        feats = build_aniso_features(
            X_rot,
            self.aniso_w_flats[rot_id],
            self.aniso_masks_flats["major"][rot_id],
            self.aniso_masks_flats["semi"][rot_id],
            self.aniso_masks_flats["minor"][rot_id],
            self.k, self.n_props_total, self.n_props_aniso,
        )
        if (self.aniso_var_idx is not None) and (self.aniso_var_mu is not None) \
                and (self.aniso_var_sigma is not None):
            vcols = self.aniso_var_idx
            if self.aniso_var_log1p:
                feats[:, vcols] = np.log1p(np.maximum(feats[:, vcols], 0.0)).astype(np.float32)
            if self.aniso_var_zscore:
                feats[:, vcols] = (feats[:, vcols] - self.aniso_var_mu) / self.aniso_var_sigma

        static_width = 0 if X_static is None else X_static.shape[1]
        out = np.empty(
            (X_rot.shape[0], X_rot.shape[1] + static_width + feats.shape[1]),
            dtype=np.float32)
        cursor = X_rot.shape[1]
        out[:, :cursor] = X_rot
        if X_static is not None:
            out[:, cursor:cursor + static_width] = X_static
            cursor += static_width
        out[:, cursor:] = feats
        return out

    def build_design_matrix(self, X):
        """Public: extended design matrix in the identity rotation (predict view)."""
        return self._transform_X(X, rot_id=0)

    def build_design_matrices(self, X):
        """All rotation views this model will use during fit (list of arrays).
        Precompute ONCE per dataset and pass to fit(ext_rots=...) when fitting
        many models on the same rows (grid search) — the transform is
        config-independent, so this removes it from the per-config cost."""
        n_rots = 1
        if self.use_augmentation and (self.rot_feature_maps is not None):
            n_rots = len(self.rot_feature_maps)
        return [self._transform_X(X, rot_id=r) for r in range(n_rots)]

    # ---- fit --------------------------------------------------------------

    def fit(self, X, y, ext_rots=None):
        """X may be None when ext_rots (precomputed design matrices, one per
        rotation view, row-aligned with y) is provided."""
        rng = np.random.default_rng(self.random_state)
        self.trees_ = []
        self._oob_ids_ = []

        is_clf = (self.task == "classification")
        if is_clf:
            self.classes_, y_codes = np.unique(y, return_inverse=True)
            self.n_classes = int(self.classes_.size)
            y_fit = y_codes.astype(np.int64)
        else:
            y_fit = np.asarray(y, dtype=np.float64)

        if ext_rots is not None:
            X_ext_rots = list(ext_rots)
            n_rots = len(X_ext_rots)
        else:
            n_rots = 1
            if self.use_augmentation and (self.rot_feature_maps is not None):
                n_rots = len(self.rot_feature_maps)
            X_ext_rots = [self._transform_X(X, rot_id=r) for r in range(n_rots)]
        X_ext0 = X_ext_rots[0]
        N = X_ext0.shape[0]

        n_per_tree = self._resolve_max_samples(N)

        if is_clf and self.class_balance == "balanced":
            self._class_rows = [np.where(y_fit == c)[0]
                                for c in range(self.n_classes)]
            self._class_rows = [r for r in self._class_rows if r.size]

        if is_clf:
            oob_proba_sum = np.zeros((N, self.n_classes), dtype=np.float64)
        else:
            oob_sum = np.zeros(N, dtype=np.float64)
        oob_cnt = np.zeros(N, dtype=np.int64)

        for _ in range(self.n_estimators):
            # bootstrap over ORIGINAL pattern ids (group-honest under augmentation)
            if is_clf and self.class_balance == "balanced":
                # equal draw per class, with replacement (balanced RF)
                per = max(1, n_per_tree // len(self._class_rows))
                ids = np.concatenate([
                    rng.choice(r, size=per, replace=True)
                    for r in self._class_rows]).astype(np.int64)
                rng.shuffle(ids)
            elif self.bootstrap:
                ids = rng.integers(0, N, size=n_per_tree, dtype=np.int64)
            else:
                ids = rng.choice(N, size=n_per_tree, replace=False)

            # one random rotation per draw (mixes rotations inside each tree)
            if n_rots > 1:
                rots = rng.integers(0, n_rots, size=ids.size)
                X_tree = np.empty((ids.size, X_ext0.shape[1]), dtype=np.float32)
                for r in range(n_rots):
                    sel = rots == r
                    if sel.any():
                        X_tree[sel] = X_ext_rots[r][ids[sel]]
            else:
                X_tree = X_ext0[ids]

            tree = SpatialDecisionTree(
                max_depth=self.max_depth,
                min_leaf=self.min_leaf,
                max_features=self.max_features,
                min_gain=self.min_gain,
                split_quantiles=self.split_quantiles,
                random_state=int(rng.integers(0, 1_000_000)),
                task=self.task,
                n_classes=self.n_classes,
            ).fit(X_tree, y_fit[ids])

            self.trees_.append(tree)

            if self.bootstrap:
                in_bag = np.zeros(N, dtype=bool)
                in_bag[ids] = True
                oob_idx = np.where(~in_bag)[0]
                self._oob_ids_.append(oob_idx.astype(np.int32))
                if oob_idx.size > 0:
                    if is_clf:
                        oob_proba_sum[oob_idx] += tree.predict_proba(X_ext0[oob_idx])
                    else:
                        oob_sum[oob_idx] += tree.predict(X_ext0[oob_idx])
                    oob_cnt[oob_idx] += 1
            else:
                self._oob_ids_.append(np.array([], dtype=np.int32))

        if is_clf:
            with np.errstate(divide="ignore", invalid="ignore"):
                proba = oob_proba_sum / np.maximum(oob_cnt, 1)[:, None]
            proba[oob_cnt == 0] = np.nan
            self.oob_proba_ = proba
            pred = np.full(N, -1, dtype=np.int64)
            cov = oob_cnt > 0
            pred[cov] = np.argmax(proba[cov], axis=1)
            self.oob_pred_ = pred
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                self.oob_pred_ = np.where(oob_cnt > 0, oob_sum / oob_cnt, np.nan).astype(np.float32)

        return self

    # ---- predict -----------------------------------------------------------

    def _proba_from_ext(self, X_ext):
        proba = np.zeros((X_ext.shape[0], self.n_classes), dtype=np.float64)
        for tree in self.trees_:
            proba += tree.predict_proba(X_ext)
        return proba / float(len(self.trees_))

    def predict(self, X, tta=False, ext0=None):
        """tta=True averages predictions over all rotations (test-time
        augmentation). ext0 = precomputed identity-rotation design matrix
        (skips the transform; X may be None then; incompatible with tta)."""
        if len(self.trees_) == 0:
            raise RuntimeError("Model has not been fitted yet.")

        if ext0 is not None and not tta:
            if self.task == "classification":
                return self.classes_[np.argmax(self._proba_from_ext(ext0), axis=1)]
            y_sum = np.zeros(ext0.shape[0], dtype=np.float64)
            for tree in self.trees_:
                y_sum += tree.predict(ext0).astype(np.float64, copy=False)
            return (y_sum / float(len(self.trees_))).astype(np.float32)

        n_rots = 1
        if tta and self.use_augmentation and (self.rot_feature_maps is not None):
            n_rots = len(self.rot_feature_maps)

        if self.task == "classification":
            proba = np.zeros((X.shape[0], self.n_classes), dtype=np.float64)
            for r in range(n_rots):
                proba += self._proba_from_ext(self._transform_X(X, rot_id=r))
            proba /= n_rots
            return self.classes_[np.argmax(proba, axis=1)]

        y_sum = np.zeros(X.shape[0], dtype=np.float64)
        for r in range(n_rots):
            X_ext = self._transform_X(X, rot_id=r)
            for tree in self.trees_:
                y_sum += tree.predict(X_ext).astype(np.float64, copy=False)
        return (y_sum / float(len(self.trees_) * n_rots)).astype(np.float32)

    def predict_proba(self, X, tta=False):
        if self.task != "classification":
            raise RuntimeError("predict_proba is only available for classification.")
        n_rots = 1
        if tta and self.use_augmentation and (self.rot_feature_maps is not None):
            n_rots = len(self.rot_feature_maps)
        proba = np.zeros((X.shape[0], self.n_classes), dtype=np.float64)
        for r in range(n_rots):
            proba += self._proba_from_ext(self._transform_X(X, rot_id=r))
        return proba / n_rots

    def predict_entropy(self, X, tta=False):
        """Paper Eq (3): normalized local entropy map for classification."""
        return normalized_entropy(self.predict_proba(X, tta=tta))

    # ---- OOB metrics --------------------------------------------------------

    def oob_scores(self, y):
        """Returns dict of OOB metrics on the training targets (Eq 4 / Eq 5)."""
        if self.task == "classification":
            _, y_codes = np.unique(y, return_inverse=True)
            mask = self.oob_pred_ >= 0
            if not mask.any():
                return {"OOB_ERR": np.nan, "OOB_ACC": np.nan, "OOB_coverage": 0}
            err = float(np.mean(self.oob_pred_[mask] != y_codes[mask]))
            return {"OOB_ERR": err, "OOB_ACC": 1.0 - err, "OOB_coverage": int(mask.sum())}

        mask = ~np.isnan(self.oob_pred_)
        if not mask.any():
            return {"OOB_R2": np.nan, "OOB_RMSE": np.nan, "OOB_MAE": np.nan, "OOB_coverage": 0}
        yt = np.asarray(y, dtype=np.float64)[mask]
        yp = self.oob_pred_[mask].astype(np.float64)
        ss_res = float(((yt - yp) ** 2).sum())
        ss_tot = float(((yt - yt.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / max(ss_tot, 1e-300)
        rmse = float(np.sqrt(ss_res / yt.size))
        mae = float(np.abs(yt - yp).mean())
        return {"OOB_R2": r2, "OOB_RMSE": rmse, "OOB_MAE": mae, "OOB_coverage": int(mask.sum())}

    # ---- permutation importance (paper Sect 2.3) ----------------------------

    def permutation_importance(self, X, y, max_oob_per_tree=256, max_trees=None,
                               n_repeats=1, random_state=0, verbose_every=25):
        """Breiman/Talebi OOB permutation importance on the model's actual
        predictors (base voxels + aniso features).

        For each tree: take its OOB patterns (identity rotation view), compute
        the baseline error, then for each predictor column permute it within
        the OOB set and record the error increase. Importances are averaged
        over trees. Regression error = MSE; classification error = error rate.

        Returns dict with 'importance_mean' (D_ext,), 'importance_std',
        'baseline_error', 'n_trees_used'.
        """
        if len(self.trees_) == 0 or len(self._oob_ids_) == 0:
            raise RuntimeError("Fit with bootstrap=True before importance.")

        rng = np.random.default_rng(random_state)
        X_ext = self._transform_X(X, rot_id=0)
        D_ext = X_ext.shape[1]

        is_clf = (self.task == "classification")
        if is_clf:
            y_codes = np.searchsorted(self.classes_, y)  # classes_ is sorted (np.unique)
        else:
            y_arr = np.asarray(y, dtype=np.float64)

        tree_ids = np.arange(len(self.trees_))
        if (max_trees is not None) and (len(tree_ids) > max_trees):
            tree_ids = rng.choice(tree_ids, size=max_trees, replace=False)

        delta_sum = np.zeros(D_ext, dtype=np.float64)
        delta_sq = np.zeros(D_ext, dtype=np.float64)
        base_sum = 0.0
        used = 0

        for ti, t_id in enumerate(tree_ids):
            tree = self.trees_[t_id]
            oob = self._oob_ids_[t_id]
            if oob.size < 2:
                continue
            if (max_oob_per_tree is not None) and (oob.size > max_oob_per_tree):
                oob = rng.choice(oob, size=max_oob_per_tree, replace=False)

            Xs = X_ext[oob].copy()
            if is_clf:
                ys = y_codes[oob]
                base = float(np.mean(tree.predict(Xs) != ys))
            else:
                ys = y_arr[oob]
                base = float(np.mean((tree.predict(Xs) - ys) ** 2))

            deltas = np.zeros(D_ext, dtype=np.float64)
            for c in range(D_ext):
                saved = Xs[:, c].copy()
                acc = 0.0
                for _ in range(n_repeats):
                    Xs[:, c] = saved[rng.permutation(saved.size)]
                    if is_clf:
                        e = float(np.mean(tree.predict(Xs) != ys))
                    else:
                        e = float(np.mean((tree.predict(Xs) - ys) ** 2))
                    acc += (e - base)
                Xs[:, c] = saved
                deltas[c] = acc / n_repeats

            delta_sum += deltas
            delta_sq += deltas ** 2
            base_sum += base
            used += 1

            if verbose_every and ((ti + 1) % verbose_every == 0):
                print(f"[importance] {ti + 1}/{len(tree_ids)} trees done")

        if used == 0:
            raise RuntimeError("No tree had enough OOB samples for importance.")

        mean = delta_sum / used
        var = np.maximum(delta_sq / used - mean ** 2, 0.0)
        return {
            "importance_mean": mean,
            "importance_std": np.sqrt(var),
            "baseline_error": base_sum / used,
            "n_trees_used": used,
        }


def build_forest_from_config(hparams, run_cfg, rot_maps, aniso_pack, random_state):
    """Construct a SpatialRandomForest from plain dicts, so the trainer and the
    predict/deploy script build EXACTLY the same model.

    hparams : n_estimators, max_depth, min_leaf, max_features, max_samples,
              split_quantiles, bootstrap
    run_cfg : task, kernel_size, n_props_total, n_props_aniso,
              n_static_features,
              use_augmentation, use_aniso_feats, aniso_var_log1p, aniso_var_zscore
    aniso_pack : dict with w_flats, masks_flats, var_idx, var_mu, var_sigma
                 (or None when aniso features are disabled)
    """
    ap = aniso_pack or {}
    return SpatialRandomForest(
        n_estimators=int(hparams["n_estimators"]),
        max_depth=(None if hparams["max_depth"] is None
                   else int(hparams["max_depth"])),
        min_leaf=int(hparams["min_leaf"]),
        max_features=hparams["max_features"],
        max_samples=hparams["max_samples"],
        min_gain=1e-6,
        split_quantiles=int(hparams["split_quantiles"]),
        bootstrap=bool(hparams["bootstrap"]),
        class_balance=hparams.get("class_balance"),
        random_state=int(random_state),
        task=run_cfg["task"],
        k=int(run_cfg["kernel_size"]),
        n_props_total=int(run_cfg["n_props_total"]),
        n_props_aniso=int(run_cfg["n_props_aniso"]),
        n_static_features=int(run_cfg.get("n_static_features", 0)),
        rot_feature_maps=rot_maps,
        use_augmentation=bool(run_cfg["use_augmentation"]),
        use_aniso_feats=bool(run_cfg["use_aniso_feats"]),
        aniso_w_flats=ap.get("w_flats"),
        aniso_masks_flats=ap.get("masks_flats"),
        aniso_var_idx=ap.get("var_idx"),
        aniso_var_mu=ap.get("var_mu"),
        aniso_var_sigma=ap.get("var_sigma"),
        aniso_var_log1p=bool(run_cfg.get("aniso_var_log1p", True)),
        aniso_var_zscore=bool(run_cfg.get("aniso_var_zscore", True)),
    )


def reshape_base_importance_to_zones(importance_vec, k, n_props_total):
    """Zone of influence (paper Sect 2.3): reshape the base-feature block of an
    importance vector to one (k,k,k) map per property.

    Axis order of each map is [X offset, Y offset, Z offset] matching the
    extraction contract (vox = i*k^2 + j*k + kk).
    Returns list of (k,k,k) arrays indexed by property position."""
    k3 = k ** 3
    base = np.asarray(importance_vec[: k3 * n_props_total], dtype=np.float64)
    per_prop = base.reshape(k3, n_props_total)
    return [per_prop[:, p].reshape(k, k, k) for p in range(n_props_total)]

