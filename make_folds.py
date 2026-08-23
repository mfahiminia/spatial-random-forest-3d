# -*- coding: utf-8 -*-
"""
Spatial block K-fold builder — ONE fold definition shared by ALL methods.

Why identical folds matter (guideline Sect 1.2 / 1.4):
- Folds are defined SPATIALLY (block boundaries + block->fold map), then every
  dataset (SRF voxel centers, classical point table, ...) is stamped with them.
  A location belongs to the same fold no matter which method looks at it.
- If methods used different folds, per-fold paired statistics (Wilcoxon etc.)
  would be meaningless, and one method could train inside another method's
  test region — an unfair information advantage.

Buffer groups (information parity):
- Datasets compared AGAINST EACH OTHER share one buffer group. Within a group,
  the exclusion radius is the WORST CASE over the group (here: the k=5 SRF
  kernel, +/-(25,25,10) m Chebyshev), and a training point is excluded if it
  is within that radius of ANY test point of the fold in ANY dataset of the
  group. So the classical methods cannot train closer to test locations than
  SRF was allowed to — parity is guaranteed by construction.
- k3 lives in its own group (k=3 radius). Buffer groups only take effect when
  BUFFER_MODE != "none"; the reported design uses "none", so no exclusion ring
  is applied to any dataset and the groups are inert. To reinstate buffering
  while still comparing k3-SRF against the point-support methods, move "k3"
  into the "main" group and re-run.

Outputs (CV_folds folder, one subfolder per dataset):
  fold_config.json, README.txt
  {name}/folds_{name}.csv    row, x, y, z, block_id, fold_id, role_f1..fK
  {name}/roles_{name}.npy    (N, K) int8: 0=train, 1=test, 2=buffer, -1=no coords
  {name}/fold_layout_{name}.png, {name}/fold_roles_{name}.png

Re-run this script whenever build_voxels.py regenerates the vectors; all
methods must consume files from the SAME generation (see created_at in json).
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

try:
    _HERE = Path(__file__).resolve().parent
except NameError:                       # notebook cell
    _HERE = Path.cwd()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


# ============================================================
# CONFIG
# ============================================================

# Every input and output hangs off one root. Defaults to the repository folder;
# set the SRF_PROJECT_ROOT environment variable to run against data kept
# elsewhere. The layout under the root is fixed:
#   <root>/Vector/            voxel arrays from build_voxels.py
#   <root>/data_classic.xlsx  point-support sample table
#   <root>/CV_folds/          this script's output
PROJECT_ROOT = Path(os.environ.get("SRF_PROJECT_ROOT", _HERE))
VECTOR_DIR = PROJECT_ROOT / "Vector"

OUT_DIR = PROJECT_ROOT / "CV_folds"

# Each dataset: source file (npy = (N,3) centers | xlsx/csv = table with coord
# columns), kernel (defines its own footprint), buffer_group (datasets compared
# together share the group's worst-case buffer + union of test points).
DATASETS = {
    "k5": {
        "source": VECTOR_DIR / "centers_train_125.npy",
        "kernel": 5,
        "buffer_group": "main",
    },
    "classic": {
        "source": PROJECT_ROOT / "data_classic.xlsx",
        "coord_cols": ["XC", "YC", "ZC"],
        "kernel": 3,               # parity with the k3 SRF: same exclusion radius
        "buffer_group": "main",
    },
    "k3": {
        "source": VECTOR_DIR / "centers_train_27.npy",
        "kernel": 3,
        "buffer_group": "k3",      # separate experiment
    },
}
PRIMARY = "k3"          # balancing optimises this dataset's test counts

N_FOLDS = 5
# SEED is the fold set reported in the paper. Changing it changes the partition
# and therefore every reported number: fold-to-fold movement in this deposit is
# the same order as the design effects being measured, so results from different
# seeds are NOT comparable. Re-running this script with SEED unchanged
# reproduces the shipped CV_folds/ byte-for-byte.
SEED = 99536
N_SEED_TRIALS = 800

# Fold-assignment objective: candidate assignments are scored by test-COUNT
# balance AND by balance of the per-fold median test->train distance. Rationale:
# the design criterion is "CV asks the deployment-distance question"; without
# distance balancing one fold can draw all the isolated blocks (an earlier
# design gave one fold d50 = 32 m against 13-19 m elsewhere) and so test
# extrapolation far beyond anything deployment will ask for.
BALANCE_DISTANCE = True
DIST_BALANCE_WEIGHT = 1.0   # weight of distance imbalance vs count imbalance

CELL_SIZE = np.array([5.0, 5.0, 2.0])   # (dx, dy, dz) metres

# Buffer mode. The reported design is "none".
#   "none"   : no exclusion ring. Adjacent train/test kernels may share voxels
#              (mild leakage accepted); method-vs-method parity is preserved
#              because every method uses the same folds.
#   "exact"  : exclude within the exact kernel-overlap threshold (k-1)*cell.
#   "safety" : exact + SAFETY_CELLS extra cells (old conservative mode).
BUFFER_MODE = "none"
SAFETY_CELLS = 1

# Block divisions per axis.
#
# (3,1,6) is the reported design, chosen by the deployment-distance criterion:
# with K=5 and no buffer, the pooled CV test->train distance median is 16.0 m
# against a deployment median of 16.2 m. Denser layouts are far too easy
# (p50 = 5 m), coarser ones far too hard.
#
# (3,3,8) with block merging was run as an independent robustness check. It
# reproduces the same conclusion about which region of the deposit is hard, but
# is a WORSE design: the CV test->train p50 rises to 36.5 m against deployment's
# 16.2 m, because splitting Y and merging back into compact 3D chunks surrounds
# each held-out block with its own held-out siblings and over-states
# extrapolation. To reproduce it: N_DIV = (3, 3, 8), MIN_BLOCK_SAMPLES = 15,
# and write to a side directory so the reported folds are not overwritten.
#
# MIN_BLOCK_SAMPLES merges any block below N samples into its nearest neighbour;
# 0 disables merging. (3,1,6) needs no merging, so it is off for the reported run.
N_DIV = (3, 1, 6)
MIN_BLOCK_SAMPLES = 0

# Deployment reference for the design figure (block-model prediction nodes):
INFER_CENTERS = VECTOR_DIR / "centers_infer_125.npy"


# ============================================================
# Helpers
# ============================================================

def buffer_radii(kernel: int):
    """Chebyshev exclusion radius per axis, or None when buffering is off."""
    if BUFFER_MODE == "none":
        return None
    extra = SAFETY_CELLS if BUFFER_MODE == "safety" else 0
    return (kernel - 1 + extra) * CELL_SIZE


def load_coords(name: str, cfg: dict):
    """Returns (coords float64 (N,3) with NaN for bad rows, valid_mask (N,))."""
    src = Path(cfg["source"])
    if not src.exists():
        raise FileNotFoundError(f"[{name}] source not found: {src}")
    if src.suffix.lower() == ".npy":
        coords = np.load(src).astype(np.float64)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"[{name}] expected (N,3) centers, got {coords.shape}")
    else:
        if src.suffix.lower() in (".xlsx", ".xls"):
            df = pd.read_excel(src)
        else:
            df = pd.read_csv(src, low_memory=False)
        cols = cfg.get("coord_cols", ["XC", "YC", "ZC"])
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"[{name}] missing coord columns {missing} in {src}")
        coords = np.column_stack([pd.to_numeric(df[c], errors="coerce").values
                                  for c in cols]).astype(np.float64)
    valid = np.isfinite(coords).all(axis=1)
    return coords, valid


def auto_divisions(extent: np.ndarray, buf: np.ndarray) -> tuple:
    divs = []
    for e, b in zip(extent, buf):
        max_div = int(np.floor(e / max(3.0 * b, 1e-9)))
        divs.append(int(np.clip(max_div, 1, 6)))
    while np.prod(divs) < 3 * N_FOLDS:
        i = int(np.argmax(extent / np.array(divs)))
        divs[i] += 1
    return tuple(divs)


def greedy_balanced_assignment(block_ids, block_counts, n_folds, rng):
    order = rng.permutation(len(block_ids))
    fold_tot = np.zeros(n_folds, dtype=np.int64)
    mapping = {}
    for i in order:
        f = int(np.argmin(fold_tot))
        mapping[int(block_ids[i])] = f
        fold_tot[f] += block_counts[i]
    return mapping, fold_tot


def merge_small_blocks(raw_bid, pts, min_samples):
    """Absorb every block with fewer than min_samples points into its spatially
    nearest neighbouring block, iteratively (smallest group first, merged into
    the nearest OTHER group). Returns a {raw_block_id -> merged_block_id} map.

    The merged id is one of the raw ids in the group (a stable representative),
    so downstream code that keys folds by block id needs no other change.
    """
    occ = np.unique(raw_bid)
    size = {int(b): int((raw_bid == b).sum()) for b in occ}
    cent = {int(b): pts[raw_bid == b].mean(axis=0) for b in occ}
    parent = {int(b): int(b) for b in occ}

    def root(b):
        while parent[b] != b:
            parent[b] = parent[parent[b]]
            b = parent[b]
        return b

    def groups():
        g = {}
        for b in occ:
            g.setdefault(root(int(b)), []).append(int(b))
        return g

    while True:
        g = groups()
        if len(g) <= 1:
            break
        gsize = {r: sum(size[b] for b in members) for r, members in g.items()}
        small = [r for r, s in gsize.items() if s < min_samples]
        if not small:
            break
        target = min(small, key=lambda r: gsize[r])
        def gcent(r):
            members = g[r]
            w = np.array([size[b] for b in members], float)
            c = np.array([cent[b] for b in members])
            return (c * w[:, None]).sum(0) / w.sum()
        tc = gcent(target)
        others = [r for r in g if r != target]
        nearest = min(others, key=lambda r: np.linalg.norm(tc - gcent(r)))
        # attach the smaller group's root under the larger's, for stable roots
        if gsize[nearest] >= gsize[target]:
            parent[target] = nearest
        else:
            parent[nearest] = target
    return {int(b): root(int(b)) for b in occ}


# ============================================================
# MAIN
# ============================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data = {}
    for name, cfg in DATASETS.items():
        coords, valid = load_coords(name, cfg)
        data[name] = {"coords": coords, "valid": valid, **cfg}
        n_bad = int((~valid).sum())
        print(f"[{name}] rows={coords.shape[0]}"
              + (f" (missing coords: {n_bad} -> excluded)" if n_bad else ""))

    # ---- shared domain (union over all valid points) ----
    all_pts = np.vstack([d["coords"][d["valid"]] for d in data.values()])
    lo, hi = all_pts.min(axis=0), all_pts.max(axis=0)
    extent = hi - lo
    # geometry heuristic for auto-sizing only (independent of BUFFER_MODE)
    geom_buf = (DATASETS[PRIMARY]["kernel"] - 1) * CELL_SIZE

    n_div = N_DIV if N_DIV is not None else auto_divisions(extent, geom_buf)
    edges = [np.linspace(lo[a], hi[a], n_div[a] + 1) for a in range(3)]
    for a in range(3):
        # Snap interior edges to HALF-CELL offsets so no grid node can sit
        # exactly on an edge (float32 vs float64 coordinate rounding would
        # otherwise flip such nodes between folds across datasets).
        cell = CELL_SIZE[a]
        for e_i in range(1, n_div[a]):
            n_cells = np.floor((edges[a][e_i] - lo[a]) / cell)
            edges[a][e_i] = lo[a] + (n_cells + 0.5) * cell
        edges[a][0] = lo[a] - 0.5 * cell
        edges[a][-1] = hi[a] + 0.5 * cell

    print("=" * 70)
    print("SPATIAL BLOCK K-FOLD BUILDER (multi-dataset)")
    print(f"domain extent (m): X={extent[0]:.0f}  Y={extent[1]:.0f}  Z={extent[2]:.0f}")
    print(f"block divisions  : {n_div}  ->  up to {int(np.prod(n_div))} blocks")
    print(f"buffer mode      : {BUFFER_MODE}")
    print(f"folds            : {N_FOLDS} | seed trials: {N_SEED_TRIALS}")
    print("=" * 70)

    def block_id_of(pts):
        ix = np.clip(np.digitize(pts[:, 0], edges[0]) - 1, 0, n_div[0] - 1)
        iy = np.clip(np.digitize(pts[:, 1], edges[1]) - 1, 0, n_div[1] - 1)
        iz = np.clip(np.digitize(pts[:, 2], edges[2]) - 1, 0, n_div[2] - 1)
        return (ix * n_div[1] + iy) * n_div[2] + iz

    # ---- balanced block->fold map from the PRIMARY dataset ----
    prim = data[PRIMARY]
    prim_pts = prim["coords"][prim["valid"]]
    prim_bid_raw = block_id_of(prim_pts)

    # merge small blocks (on the primary), then route EVERY dataset through the
    # same raw->merged map so all datasets share one block definition.
    if MIN_BLOCK_SAMPLES and MIN_BLOCK_SAMPLES > 0:
        merge_map = merge_small_blocks(prim_bid_raw, prim_pts, MIN_BLOCK_SAMPLES)
        prim_bid = np.array([merge_map[int(b)] for b in prim_bid_raw])
        n_raw = len(np.unique(prim_bid_raw))
        n_mrg = len(np.unique(prim_bid))
        print(f"block merging: {n_raw} raw -> {n_mrg} merged "
              f"(min {MIN_BLOCK_SAMPLES} samples/block)")
    else:
        merge_map = None
        prim_bid = prim_bid_raw

    occupied, counts = np.unique(prim_bid, return_counts=True)
    print(f"non-empty blocks (primary): {occupied.size} "
          f"(sizes min={counts.min()}, median={int(np.median(counts))}, max={counts.max()})")

    # merged-block centroids, for datasets whose points land in a cell that is
    # empty in the primary (route them to the nearest merged block).
    merged_ids = occupied
    merged_centroids = np.vstack([prim_pts[prim_bid == m].mean(axis=0)
                                  for m in merged_ids])
    merged_tree = cKDTree(merged_centroids)

    def merged_block_of(pts):
        raw = block_id_of(pts)
        if merge_map is None:
            return raw
        out = np.empty(raw.shape[0], dtype=np.int64)
        for i, b in enumerate(raw):
            r = merge_map.get(int(b))
            if r is None:                       # cell empty in the primary
                _, j = merged_tree.query(pts[i])
                r = int(merged_ids[j])
            out[i] = r
        return out

    def per_fold_d50(mapping):
        """Median test->train NN distance per fold for a candidate assignment."""
        fold_of_pt = np.array([mapping[int(b)] for b in prim_bid])
        d50s = np.empty(N_FOLDS)
        for f in range(N_FOLDS):
            te = fold_of_pt == f
            if te.sum() < 3 or (~te).sum() < 3:
                return None
            dd, _ = cKDTree(prim_pts[~te]).query(prim_pts[te], k=1)
            d50s[f] = np.median(dd)
        return d50s

    best = None
    master = np.random.default_rng(SEED)
    for _ in range(N_SEED_TRIALS):
        trial_rng = np.random.default_rng(master.integers(0, 2**63 - 1))
        mapping, fold_tot = greedy_balanced_assignment(occupied, counts, N_FOLDS, trial_rng)
        cnt_imb = fold_tot.std() / max(fold_tot.mean(), 1.0)
        if BALANCE_DISTANCE:
            d50s = per_fold_d50(mapping)
            if d50s is None:
                continue
            dist_imb = d50s.std() / max(np.median(d50s), 1e-9)
            score = cnt_imb + DIST_BALANCE_WEIGHT * dist_imb
        else:
            d50s = None
            score = cnt_imb
        if (best is None) or (score < best[0]):
            best = (score, mapping, fold_tot.copy(), d50s)
    _, block_to_fold, fold_tot, fold_d50s = best
    # No raw-block backfill needed: merged_block_of() routes any point in an
    # empty-in-primary cell to the nearest merged block, which is always a key
    # of block_to_fold (the greedy assignment covered every occupied merged id).
    print(f"primary test counts per fold: {fold_tot.tolist()} (std={fold_tot.std():.1f})")
    if fold_d50s is not None:
        print("per-fold test d50 (m)       : "
              + str([round(float(v), 1) for v in fold_d50s])
              + f" (std={fold_d50s.std():.1f})")

    # ---- fold ids per dataset (same MERGED block map for everyone) ----
    for name, d in data.items():
        fold_id = np.full(d["coords"].shape[0], -1, dtype=np.int8)
        v = d["valid"]
        bid = merged_block_of(d["coords"][v])
        fold_id[v] = np.array([block_to_fold[int(b)] for b in bid], dtype=np.int8)
        d["block_id"] = np.full(d["coords"].shape[0], -1, dtype=np.int64)
        d["block_id"][v] = bid
        d["fold_id"] = fold_id

    # ---- buffer groups: worst-case radius + UNION of test points ----
    groups = {}
    for name, d in data.items():
        groups.setdefault(d["buffer_group"], []).append(name)

    config = {
        "created_at": datetime.now().isoformat(),
        "n_folds": int(N_FOLDS),
        "seed": int(SEED),
        "seed_trials": int(N_SEED_TRIALS),
        "cell_size": CELL_SIZE.tolist(),
        "buffer_mode": BUFFER_MODE,
        "safety_cells": int(SAFETY_CELLS),
        "block_divisions": list(n_div),
        "min_block_samples": int(MIN_BLOCK_SAMPLES),
        "block_edges": {ax: edges[a].tolist() for a, ax in enumerate("xyz")},
        "block_to_fold": {str(k): int(v) for k, v in block_to_fold.items()},
        "primary_dataset": PRIMARY,
        "buffer_groups": {g: members for g, members in groups.items()},
        "datasets": {},
    }

    if BUFFER_MODE == "none":
        print("\nbuffer: DISABLED (BUFFER_MODE='none') - roles are train/test only")
        for name, d in data.items():
            N = d["coords"].shape[0]
            roles = np.full((N, N_FOLDS), -1, dtype=np.int8)
            v = d["valid"]
            for f in range(N_FOLDS):
                te = d["fold_id"] == f
                roles[te, f] = 1
                roles[v & ~te, f] = 0
            d["roles"] = roles
            d["buffer_m"] = None
    else:
        for gname, members in groups.items():
            buf = np.max([buffer_radii(data[m]["kernel"]) for m in members], axis=0)
            print(f"\nbuffer group '{gname}': {members} | "
                  f"radius +/-({buf[0]:.0f},{buf[1]:.0f},{buf[2]:.0f}) m "
                  f"(worst case over group)")

            # union of test points per fold (across ALL group members)
            fold_test_trees = []
            for f in range(N_FOLDS):
                pts_list = [data[m]["coords"][data[m]["fold_id"] == f] for m in members]
                pts = np.vstack([p for p in pts_list if p.size]) if any(p.size for p in pts_list) else None
                fold_test_trees.append(cKDTree(pts / buf) if pts is not None else None)

            for m in members:
                d = data[m]
                N = d["coords"].shape[0]
                roles = np.full((N, N_FOLDS), -1, dtype=np.int8)
                v = d["valid"]
                scaled = d["coords"] / buf
                for f in range(N_FOLDS):
                    te = d["fold_id"] == f
                    tr_cand = v & ~te
                    roles[te, f] = 1
                    roles[tr_cand, f] = 0
                    if fold_test_trees[f] is not None and tr_cand.any():
                        cand_idx = np.where(tr_cand)[0]
                        hits = fold_test_trees[f].query_ball_point(scaled[cand_idx], r=1.0, p=np.inf)
                        is_buf = np.fromiter((len(h) > 0 for h in hits), dtype=bool, count=cand_idx.size)
                        roles[cand_idx[is_buf], f] = 2
                d["roles"] = roles
                d["buffer_m"] = buf

    # ---- outputs per dataset ----
    role_names = {-1: "excluded", 0: "train", 1: "test", 2: "buffer"}
    for name, d in data.items():
        ds_dir = OUT_DIR / name
        ds_dir.mkdir(exist_ok=True)
        pts = d["coords"]
        roles = d["roles"]
        N = pts.shape[0]

        df = pd.DataFrame({
            "row": np.arange(N),
            "x": pts[:, 0], "y": pts[:, 1], "z": pts[:, 2],
            "block_id": d["block_id"], "fold_id": d["fold_id"],
        })
        for f in range(N_FOLDS):
            df[f"role_f{f + 1}"] = [role_names[int(r)] for r in roles[:, f]]
        csv_path = ds_dir / f"folds_{name}.csv"
        npy_path = ds_dir / f"roles_{name}.npy"
        df.to_csv(csv_path, index=False)
        np.save(npy_path, roles)

        buf = d["buffer_m"]
        buf_str = ("none" if buf is None
                   else f"+/-({buf[0]:.0f},{buf[1]:.0f},{buf[2]:.0f}) m")
        stats = []
        print(f"\n[{name}] N={N} | kernel k={d['kernel']} | group '{d['buffer_group']}' "
              f"| buffer {buf_str}")
        for f in range(N_FOLDS):
            n_te = int((roles[:, f] == 1).sum())
            n_bu = int((roles[:, f] == 2).sum())
            n_tr = int((roles[:, f] == 0).sum())
            pct = round(100.0 * n_bu / max(n_tr + n_bu, 1), 1)
            stats.append({"fold": f + 1, "n_train": n_tr, "n_test": n_te,
                          "n_buffer": n_bu, "buffer_pct_of_nontest": pct})
            print(f"  fold {f + 1}: train={n_tr:4d}  test={n_te:4d}  buffer={n_bu:4d} "
                  f"({pct:.1f}% of non-test lost)")

        config["datasets"][name] = {
            "source": str(d["source"]),
            "kernel": int(d["kernel"]),
            "buffer_group": d["buffer_group"],
            "buffer_m": None if buf is None else buf.tolist(),
            "n_samples": int(N),
            "n_excluded_rows": int((~d["valid"]).sum()),
            "fold_stats": stats,
            "csv": str(csv_path),
            "roles_npy": str(npy_path),
        }

        # ---- figures ----
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.colors import ListedColormap

            fold_cmap = plt.get_cmap("tab10", N_FOLDS)
            fig, axes = plt.subplots(1, 2, figsize=(11, 5))
            vmask = d["valid"]
            for ax, (i, j, xl, yl) in zip(axes, [(0, 1, "X", "Y"), (0, 2, "X", "Z")]):
                ax.scatter(pts[vmask, i], pts[vmask, j], c=d["fold_id"][vmask],
                           cmap=fold_cmap, s=14, vmin=-0.5, vmax=N_FOLDS - 0.5)
                for e in edges[i]:
                    ax.axvline(e, color="k", lw=0.4, alpha=0.4)
                for e in edges[j]:
                    ax.axhline(e, color="k", lw=0.4, alpha=0.4)
                ax.set_xlabel(xl); ax.set_ylabel(yl)
                ax.set_title(f"{xl}-{yl} | colour = fold")
                ax.set_aspect("equal", adjustable="datalim")
            fig.suptitle(f"Spatial block folds — {name} (K={N_FOLDS})")
            fig.tight_layout()
            fig.savefig(ds_dir / f"fold_layout_{name}.png", dpi=150)
            plt.close(fig)

            role_cmap = ListedColormap(["#4daf4a", "#e41a1c", "#bbbbbb"])
            fig, axes = plt.subplots(1, N_FOLDS, figsize=(4.2 * N_FOLDS, 4.4))
            for f in range(N_FOLDS):
                ax = axes[f]
                m = roles[:, f] >= 0
                ax.scatter(pts[m, 0], pts[m, 2], c=roles[m, f], cmap=role_cmap,
                           s=14, vmin=-0.5, vmax=2.5)
                ax.set_title(f"fold {f + 1} (X-Z)\ngreen=train red=test grey=buffer",
                             fontsize=9)
                ax.set_xlabel("X"); ax.set_ylabel("Z")
            fig.suptitle(f"Roles per fold — {name}")
            fig.tight_layout()
            fig.savefig(ds_dir / f"fold_roles_{name}.png", dpi=150)
            plt.close(fig)
        except Exception as e:
            print(f"[WARN] figures skipped for {name}: {e}")

    # ---- design justification: CV test distances vs DEPLOYMENT distances ----
    # (the paper figure: the fold design asks the same question as deployment)
    try:
        prim = data[PRIMARY]
        pts_p = prim["coords"]
        roles_p = prim["roles"]
        cv_d = []
        for f in range(N_FOLDS):
            te = roles_p[:, f] == 1
            tr = roles_p[:, f] == 0
            if te.any() and tr.any():
                dd, _ = cKDTree(pts_p[tr]).query(pts_p[te], k=1)
                cv_d.append(dd)
        cv_d = np.concatenate(cv_d)

        dep_d = None
        if INFER_CENTERS.exists():
            infer_pts = np.load(INFER_CENTERS).astype(np.float64)
            dep_d, _ = cKDTree(pts_p[prim["valid"]]).query(infer_pts, k=1)

        qs = [10, 25, 50, 75, 90, 95]
        rationale = {
            "criterion": "CV pooled test->train NN distance should match the "
                         "deployment (block-model node -> nearest training sample) "
                         "distance distribution; per-fold d50 balanced across folds",
            "balance_distance": bool(BALANCE_DISTANCE),
            "dist_balance_weight": float(DIST_BALANCE_WEIGHT),
            "per_fold_test_d50_m": ([round(float(v), 2) for v in fold_d50s]
                                    if fold_d50s is not None else None),
            "cv_pooled_nn_distance_m": {f"p{q}": float(np.percentile(cv_d, q)) for q in qs},
        }
        if dep_d is not None:
            rationale["deployment_nn_distance_m"] = {
                f"p{q}": float(np.percentile(dep_d, q)) for q in qs}
            rationale["infer_centers"] = str(INFER_CENTERS)
        config["design_rationale"] = rationale

        print("\nDESIGN CHECK | pooled CV test->train NN distance (m): "
              + " ".join(f"p{q}={np.percentile(cv_d, q):.1f}" for q in [25, 50, 75, 90]))
        if dep_d is not None:
            print("             | deployment node->train NN distance  (m): "
                  + " ".join(f"p{q}={np.percentile(dep_d, q):.1f}" for q in [25, 50, 75, 90]))

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4.5))
        bins = np.linspace(0, max(np.percentile(cv_d, 99),
                                  np.percentile(dep_d, 99) if dep_d is not None else 0) + 5, 40)
        ax.hist(cv_d, bins=bins, density=True, alpha=0.55,
                label=f"CV test points (K={N_FOLDS}, median {np.median(cv_d):.1f} m)")
        if dep_d is not None:
            ax.hist(dep_d, bins=bins, density=True, alpha=0.55,
                    label=f"Deployment block nodes (median {np.median(dep_d):.1f} m)")
        ax.axvline(np.median(cv_d), color="C0", ls="--", lw=1)
        if dep_d is not None:
            ax.axvline(np.median(dep_d), color="C1", ls="--", lw=1)
        ax.set_xlabel("distance to nearest training sample (m)")
        ax.set_ylabel("density")
        ax.set_title("Fold design justification: CV question vs deployment question")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT_DIR / "design_distance_match.png", dpi=150)
        plt.close(fig)
        print(f"Saved design figure: {OUT_DIR / 'design_distance_match.png'}")
    except Exception as e:
        print(f"[WARN] design-rationale step skipped: {e}")

    with open(OUT_DIR / "fold_config.json", "w", encoding="utf-8") as fjs:
        json.dump(config, fjs, indent=2)

    readme = """SPATIAL BLOCK K-FOLD FILES — usage contract (all methods!)

roles_{name}.npy : int8 (N, K).  0=train  1=test  2=buffer  -1=excluded(no coords)
folds_{name}.csv : same + x,y,z per row (row order == source file row order)

Datasets:
  k5      -> rows of Vector\\X_train_125.npy (SRF k=5)
  classic -> rows of data_classic.xlsx (point-support methods: RF, RF+XYZ,
             bagged trees, GBM, XGBoost, GLS-RF)
  k3      -> rows of Vector\\X_train_27.npy (SRF k=3, the reported SRF variant;
             separate buffer group)

For fold f (0-based):
    roles = np.load('classic/roles_classic.npy')
    train_mask = roles[:, f] == 0
    test_mask  = roles[:, f] == 1
    # buffer (2) and excluded (-1) rows are used by NOBODY in this fold.

Rules:
- Identical spatial fold definition for every dataset (fair comparison).
- BUFFER_MODE is recorded in fold_config.json. In "none" mode roles contain
  only train/test: adjacent train/test kernels may share voxels (accepted,
  disclosed design decision); parity across methods is preserved because all
  methods use the same folds. Fold blocks were sized so the CV test-distance
  distribution matches the deployment distance distribution (see
  design_distance_match.png and design_rationale in fold_config.json).
- Train on train_mask, predict test_mask. No re-splitting, no exceptions.
- Per-fold preprocessing (imputation, normalisation, variogram fits) must use
  train_mask rows only.
- All methods must consume files from the SAME generation of this script
  (check created_at in fold_config.json).
"""
    (OUT_DIR / "README.txt").write_text(readme, encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"Folds written to: {OUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
