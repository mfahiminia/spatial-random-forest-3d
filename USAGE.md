# Usage

A step-by-step walkthrough: what to install, what to run, in what order, how
long each step takes, and what to check afterwards. `README.md` explains *why*
the pipeline is built this way; this file is the operating manual.

- [1. Install](#1-install)
- [2. Quick start](#2-quick-start)
- [3. Where the files live](#3-where-the-files-live)
- [4. The steps in detail](#4-the-steps-in-detail)
- [5. Running on your own data](#5-running-on-your-own-data)
- [6. Full runs vs `--smoke`](#6-full-runs-vs---smoke)
- [7. Reproducibility rules](#7-reproducibility-rules)
- [8. Troubleshooting](#8-troubleshooting)

---

## 1. Install

Python 3.10 or newer (developed and reported on 3.13.5). No compiled
extensions — the spatial trees are pure NumPy.

```bash
git clone https://github.com/USER/REPO.git
cd REPO
pip install -r requirements.txt
```

With conda:

```bash
conda create -n srf python=3.13
conda activate srf
pip install -r requirements.txt
```

`scikit-learn` and `xgboost` are needed only by `classic_final.py`; everything
else runs without them.

Check the install:

```bash
python -c "import numpy, pandas, scipy, joblib, matplotlib, openpyxl; print('ok')"
```

---

## 2. Quick start

Five commands take you from nothing to a cross-validated model and a block-model
map, on synthetic data:

```bash
python make_demo_data.py
python build_voxels.py
python make_folds.py
python srf_train.py
python srf_predict.py
```

Total about five minutes. Add `--small` to `make_demo_data.py` for a quarter-size
grid if you only want to see it work.

Then the other two method families:

```bash
python GLSRF.py --smoke
python glsrf_report.py
python GLSRF_figures.py
python classic_final.py --smoke
```

> The demo data is a simulation, not the deposit. Its numbers are not the
> paper's, and on some questions it points the other way — see the warning in
> `README.md`. Use it to check that the code runs.

---

## 3. Where the files live

Every script resolves its paths from one root, which defaults to the repository
folder. To keep data outside the repository, set one environment variable:

```bash
# Windows (cmd)
set SRF_PROJECT_ROOT=D:\path\to\project
# Windows (PowerShell)
$env:SRF_PROJECT_ROOT = "D:\path\to\project"
# macOS / Linux
export SRF_PROJECT_ROOT=/path/to/project
```

The layout under that root is fixed:

```
<root>/
  Vector/                  grade_full_D3.csv  +  the voxel arrays
  data_classic.xlsx        point-support sample table
  CV_folds/                the fold definition
  SRF_runs/                srf_train.py output
  SRF_Ablation/            srf_ablation_run.py output
  SRF_Sensitivity/         srf_sensitivity.py output
  GLSRF_runs/              GLSRF.py and glsrf_report.py output
  Classic_runs/            classic_final.py output
```

Nothing writes outside that root, and nothing outside it is read.

---

## 4. The steps in detail

### Step 0 — `make_demo_data.py`

Only for the synthetic dataset. Skip it if you have real data.

| | |
| --- | --- |
| Reads | nothing |
| Writes | `Vector/grade_full_D3.csv`, `data_classic.xlsx` |
| Time | ~1 min (`--small`: ~10 s) |

### Step 1 — `build_voxels.py`

Slides a k×k×k window over every grid node and vectorises it into one feature
row. Builds k=3 and k=5 in a single pass.

| | |
| --- | --- |
| Reads | `Vector/grade_full_D3.csv` |
| Writes | `Vector/X_train_{27,125}.npy`, `y_train_*`, `centers_*`, `X_infer_*`, `Grid_*` |
| Time | ~3 min for a 60×47×117 grid |

Check the printed `TRAIN` / `INFER` row counts. If `TRAIN` is 0, no window had
both responses present at its centre and at least 70 % covariate coverage.

`X_infer_125.npy` can reach a few hundred MB. It is needed only for deployment,
not for cross-validation.

### Step 2 — `make_folds.py`

Cuts the domain into blocks, assigns blocks to K=5 folds, and stamps that one
spatial definition onto **every** dataset.

| | |
| --- | --- |
| Reads | `Vector/centers_train_{27,125}.npy`, `data_classic.xlsx` |
| Writes | `CV_folds/{k3,k5,classic}/`, `fold_config.json`, `design_distance_match.png` |
| Time | ~1 min (800 candidate assignments are scored) |

Two things to read in the output:

* the per-fold test counts and per-fold `d50` — both should be reasonably even;
* the `DESIGN CHECK` lines. The CV test→train distance distribution should
  resemble the deployment distance distribution. If the CV median is far below
  the deployment median, cross-validation is measuring an easier problem than
  the one you will actually deploy into. Adjust `N_DIV` until they agree.

This script **overwrites `CV_folds/` in place.** Every result already computed
against the old folds becomes incomparable the moment you re-run it. Archive the
directory first if you have runs you intend to keep.

### Step 3 — `srf_train.py`

Spatial-CV training of the SRF. Inside each outer fold the training blocks are
split into three spatial sub-folds, every grid configuration is scored on
held-out sub-folds, and the winner is refit and scored once on the outer test
block.

| | |
| --- | --- |
| Reads | `Vector/X_train_27.npy`, `y_train_27.npy`, `centers_train_27.npy`, `CV_folds/k3/` |
| Writes | `SRF_runs/SRF_run_<timestamp>/` |
| Time | ~10 s (27 configurations × 3 sub-folds × 5 folds × 2 targets) |

The run folder:

```
00_metadata/       run_config.json  — everything needed to rebuild a model
                   prepared.pkl     — rotation maps + anisotropy normalisation
01_grid_search/    grid_results_{TARGET}_fold{N}.csv
02_models/         best_{TARGET}_fold{N}.joblib, meta_{TARGET}.json
03_test_evaluation/predictions_long_{METHOD}.csv, fold_metrics_*, test_metrics.csv
04_importance/     zone-of-influence maps (single-split modes only)
05_full_prediction/filled by srf_predict.py
summary.csv        the headline table
```

Useful switches at the top of the file: `PATTERN_VARIANT` (`local_aniso` is the
reported model), `KERNEL_VARIANT` (`k3` / `k5`), `TUNE_MODE` (`subfolds` / `oob`),
`SPLIT_MODE` (`folds` / `random` / `none`). The first three can also be set from
the environment: `SRF_PATTERN_VARIANT`, `SRF_KERNEL_VARIANT`, `SRF_QUICK`.

### Step 4 — `srf_predict.py`

Deployment. Ranks configurations by their mean sub-fold CV score **across**
folds, refits the winner on every labelled row, and predicts the block model in
batches.

| | |
| --- | --- |
| Reads | a run folder, plus `Vector/X_infer_27.npy` |
| Writes | `05_full_prediction/PRED_{TARGET}_Rank{K}.{csv,npy}`, `02_models/deploy_*.joblib` |
| Time | ~2 s for 40,000 block nodes |

Interactive by default (it asks which run, which target, which rank).
Non-interactive:

```python
from srf_predict import run_prediction
run_prediction(targets=["DTR"], rank_ids={"DTR": 1}, interactive=False)
```

The output carries `P_GT55` and `P_GT80` alongside the mean prediction. Those
exceedance probabilities, not the mean, are the intended product for high-grade
targeting — conditional-mean estimators smooth, and their hard predictions
under-reach the high-grade tail.

### Step 5 — `GLSRF.py`

RF-GLS on the point-support samples, on the same folds. Searches the covariance
design (anisotropic vs isotropic, exponential vs Matérn-3/2) alongside the tree
hyperparameters.

| | |
| --- | --- |
| Reads | `data_classic.xlsx` (resolved from `CV_folds/fold_config.json`), `CV_folds/classic/` |
| Writes | `GLSRF_runs/GLSRF_<data>_<timestamp>/` |
| Time | **5.8 h** for the full search; ~30 s with `--smoke` |

All geostatistical parameters live together in Section 2 of the file: nugget,
azimuth/dip/tilt, the three ranges, and the correlation function. Edit there,
nowhere else.

### Step 6 — `glsrf_report.py`

A `GLSRF.py` run reports whatever design won in each fold. The **reported
estimator** is one fixed specification — ANISO/exp held constant across folds,
mean-function prediction — and this script extracts it from a finished run.

| | |
| --- | --- |
| Reads | a `GLSRF_runs/GLSRF_*` folder with `grid_*` tables |
| Writes | `GLSRF_runs/GLSRF_reported_ANISOexp/` |
| Time | ~4 min (10 refits) |

It prints the design ranking first, so you can see that the fixed design is the
one the selection metric picks rather than taking it on trust.

### Step 7 — `GLSRF_figures.py`

The design-comparison figure, paired within fold.

```bash
python GLSRF_figures.py [run_folder]      # blank = most recent run
```

Writes `glsrf_design_choices.png` and `.pdf` into the run folder.

### Step 8 — `classic_final.py`

The five tree baselines: RF, RF+XYZ, bagged trees, GBM, XGBoost. Each is tuned
by nested spatial CV; the bootstrap ensembles are also tuned by out-of-bag error
so the two selection criteria can be compared on identical models.

| | |
| --- | --- |
| Reads | `data_classic.xlsx`, `CV_folds/classic/` |
| Writes | `Classic_runs/TREES_<data>_<timestamp>/` |
| Time | **~50 min** single-threaded; ~1 min with `--smoke` |

`paired_tests.csv` holds the Wilcoxon comparisons on per-sample squared errors;
`oob_vs_cv.csv` shows what each selection criterion chose and what it cost.

### Optional — SRF ablation and sensitivity

```bash
python srf_ablation_run.py           # 6 arms: kernel size × augmentation × anisotropy
python srf_ablation_run.py --smoke   # ~1 min plumbing check
python srf_ablation_run.py --force   # re-run arms already in the manifest

python srf_sensitivity.py            # wide hyperparameter sweep + seed noise floor
```

The ablation takes ~20 min per k3 arm and 60–90 min per k5 arm, and keeps a
manifest so an interrupted run resumes instead of repeating finished arms.

---

## 5. Running on your own data

You need two files. `DATA.md` documents both column by column; the short version:

1. **`Vector/grade_full_D3.csv`** — one row per node of a regular 3D grid, with
   `XC, YC, ZC`, four covariates and the two responses. Every grid node must
   appear, including empty ones: the grid dimensions are inferred from the
   unique coordinate values.
2. **`data_classic.xlsx`** — the point-support samples, with `XC, YC, ZC` and
   the same covariates and responses. Only needed for GLS-RF and the tree
   baselines; the SRF pipeline does not read it.

Then work through this checklist, in order:

* `build_voxels.py` → `COLUMN_MAP`, `CELL_SIZE`, `MIN_COVERAGE`
* `make_folds.py` → `CELL_SIZE`, `N_DIV`, `N_FOLDS`, `SEED`
* `srf_train.py` → `CELL_SIZE`, the continuity ellipsoid (`A_MAJOR`, `A_SEMI`,
  `A_MINOR`, `AZIMUTH_DEG`, `DIP_DEG`), `EXCEEDANCE_THRESHOLDS`
* `GLSRF.py` → Section 2 in full, especially `NUGGET`
* `classic_final.py` → `FEATURE_COLS`, `TARGET_COLS`

Two of these deserve real thought rather than a copied value:

**The continuity ellipsoid** comes from your own variography. The anisotropy
features and Σ are both built in that frame, so a wrong frame costs accuracy in
both methods.

**`N_DIV`** is not a free parameter. It is chosen so the CV test→train distance
distribution matches the deployment node→sample distance distribution.
`make_folds.py` prints both and draws `design_distance_match.png`. Match them.
The specific value `(3, 1, 6)` is an answer for this deposit, not a default.

---

## 6. Full runs vs `--smoke`

| Script | Full run | `--smoke` |
| --- | --- | --- |
| `srf_train.py` | ~10 s | `SRF_QUICK=1` env var |
| `GLSRF.py` | 5.8 h | ~30 s |
| `classic_final.py` | ~50 min | ~1 min |
| `srf_ablation_run.py` | 4–8 h | ~1 min |

`--smoke` cuts the grids and shrinks the ensembles. It is for checking that the
plumbing works, never for results — every smoke run records `"smoke": true` in
its `run_config.json` so it cannot later be mistaken for a real one.

To work on a single fold while developing, set `FOLDS_TO_RUN = [0]` in `GLSRF.py`
or `classic_final.py`.

---

## 7. Reproducibility rules

1. **Fold churn is the main hazard.** `make_folds.py` overwrites `CV_folds/` in
   place, and every downstream number is tied to that partition. Results from
   different fold sets are not comparable. Archive the directory before
   re-running it.
2. **Check the fingerprints before comparing anything.** Every run records
   `coord_fingerprint_sha1` and `roles_fingerprint_sha1` in its
   `run_config.json`. Two runs are comparable only if both match. A stale path
   is invisible otherwise — it yields a complete, plausible, wrong table.
3. **Seeds are part of the specification.** `make_folds.SEED`,
   `srf_train.SUBFOLD_SEED` / `REFIT_SEED`, `GLSRF.SEED`, `classic_final.SEED`.
4. **`N_JOBS` matters for GLS-RF.** It is exactly reproducible at a fixed
   `N_JOBS`, but not across values of it: worker processes run BLAS
   single-threaded while `n_jobs=1` does not, and the different reduction order
   flips the occasional near-tied split (~0.006 R² on one fold, ~0.001 pooled).
   Report GLS-RF numbers with the `N_JOBS` that produced them. The SRF pipeline
   is not affected.
5. **Supports differ.** SRF is scored on voxel patterns, the point-support
   methods on samples. Pooled R² from the two is not directly comparable; a
   cross-method comparison has to restrict every method to the common support.

---

## 8. Troubleshooting

**`FileNotFoundError: ...CV_folds/fold_config.json not found`**
`GLSRF.py` and `classic_final.py` resolve their data file *from* the folds, so
the folds must exist first. Run `make_folds.py`.

**`The folds were built from <path>, which no longer exists`**
`fold_config.json` stores an absolute path. Both scripts fall back to the same
file name under the project root; if that is missing too, either restore the
file or set `DATA_PATH_OVERRIDE` at the top of the script.

**`roles rows (N) != data rows (M)`** or
**`Roles rows (N) != X rows (M). The folds were built for a different extraction`**
The folds and the feature arrays are from different generations. Re-run
`make_folds.py` after any `build_voxels.py` run.

**`Coordinates do not match the folds CSV row-by-row (max diff ... m)`**
The row order of the data table changed after the folds were built. Row order is
the contract that ties the files together. Rebuild the folds.

**`Expected local 135 + static 0 = 135 features; got 625`**
`KERNEL_VARIANT` and the feature file disagree — 135 is k=3, 625 is k=5. Use
`srf_train.set_variant(pattern=..., kernel=...)` rather than assigning
`X_path` and friends by hand; it re-derives every dependent global at once.

**`srf_core_final is outdated (need API v8, got v7)`**
A Jupyter kernel is holding an older copy of the module in memory. Restart the
kernel.

**`X_infer not found: ...Vector/X_infer_27.npy`**
The inference array is not committed — it is large and fully regenerable. Run
`build_voxels.py`, then `srf_predict.py`.

**`max_features float must be in (0, 1]`**
A number above 1 is a *count* of candidate features and must stay an `int`;
a value in (0, 1] is a *fraction*. If you add a code path that reads
hyperparameters back from a CSV, preserve that distinction — `srf_predict.py`
does this in `_parse_max_features`.

**`RuntimeError: too few sub-fold predictions`** /
**`Outer fold N has fewer than two ...`**
A fold ended up too small to split into sub-folds, usually because the block
layout is too coarse for the sample count. Lower `N_DIV`, or raise
`MIN_BLOCK_SAMPLES` in `make_folds.py` so tiny blocks are merged.

**`MemoryError` in `build_voxels.py`**
The k=5 inference array is `n_nodes × 625` float32. Restrict the covariates to
the estimated domain (leave the rest missing), or drop `5` from `VOXEL_SIZES`.

**Everything prints `?` or mojibake in the Windows console**
Cosmetic only. `chcp 65001` switches the console to UTF-8.

**A run finished but the numbers look wrong**
Check `run_config.json` first: `"smoke"` should be `false`, and the two
fingerprints should match the run you are comparing against.
