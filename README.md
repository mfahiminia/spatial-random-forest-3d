# 3D Spatial Random Forest Workflows

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22067693.svg)](https://doi.org/10.5281/zenodo.22067693)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Code accompanying a manuscript on spatially explicit machine learning for
geometallurgical block modelling. It implements two spatially aware learners and
evaluates them, together with conventional tree ensembles, under a single
spatial block cross-validation design that every method shares unchanged.

| Method | Where spatial information enters |
| --- | --- |
| `SRF_3D` | the **representation** — each sample becomes a k x k x k voxel pattern |
| `GLS-RF_3D` | the **estimator** — split criterion and node values become generalised least squares under an oriented covariance |
| `RF + XYZ` | the **feature space** — coordinates appended as ordinary predictors |
| `RF`, bagged trees, `GBM`, `XGBoost` | nowhere — the non-spatial baselines |

SRF_3D extends Talebi et al. (2021), *A Truly Spatial Random Forests Algorithm
for Geoscience Data Analysis and Modelling*, Mathematical Geosciences, from 2D
pixel neighbourhoods to 3D voxel kernels. GLS-RF_3D follows Saha, Basu & Datta
(2023), *Random Forests for Spatially Dependent Data*, JASA, with an anisotropic
covariance.

The case-study data are not included. A synthetic dataset is provided so the
workflow can be tested from start to finish. See [`DATA.md`](DATA.md) for the
required input schema and [`USAGE.md`](USAGE.md) for detailed troubleshooting.

## Reported results

Pooled spatial cross-validation on the common 456-node support: every sample
predicted exactly once, by a model that never saw its spatial neighbourhood.

| Method | DTR R2 | DTR RMSE | Magnetic R2 | Magnetic RMSE |
| --- | --- | --- | --- | --- |
| GLS-RF_3D | **0.635** | **15.19** | **0.689** | **14.45** |
| RF + XYZ | 0.625 | 15.39 | 0.657 | 15.18 |
| Ordinary RF | 0.622 | 15.45 | 0.671 | 14.86 |
| SRF_3D | 0.601 | 15.87 | 0.666 | 14.99 |
| XGBoost | 0.586 | 16.17 | 0.660 | 15.13 |

Scored under out-of-bag error instead, as in the original SRF formulation,
SRF_3D reaches R2 0.815 and 0.814 — the highest in the study, and an optimism of
+0.215 and +0.148 over the honest spatial-CV figures above. That gap is the
reason the evaluation design matters more than the choice of learner.

## Installation

Python 3.10 or newer is required. The recorded dependency versions are in
`requirements.txt`.

```bash
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Quick start with synthetic data

Run these commands from the repository folder:

```bash
python make_demo_data.py --small
python build_voxels.py
python make_folds.py
python srf_train.py
python srf_predict.py
```

`srf_predict.py` is interactive and asks for the training run, target, and model
rank. The synthetic data are only for testing the workflow.

Run the other model families with reduced grids:

```bash
python GLSRF.py --smoke
python glsrf_report.py
python GLSRF_figures.py
python classic_final.py --smoke
python srf_ablation_run.py --smoke
```

Remove `--small` or `--smoke` for full runs. For a quick SRF diagnostic, set
`SRF_QUICK=1` before running `srf_train.py`.

## Project root

Scripts use the repository folder as the default input/output root. To keep data
and results elsewhere, set `SRF_PROJECT_ROOT` before running any script.

```powershell
# Windows PowerShell
$env:SRF_PROJECT_ROOT = "D:\path\to\project"
```

```cmd
:: Windows cmd
set SRF_PROJECT_ROOT=D:\path\to\project
```

```bash
# macOS / Linux
export SRF_PROJECT_ROOT=/path/to/project
```

The selected root uses this layout:

```text
<root>/
  Vector/
    grade_full_D3.csv
  data_classic.xlsx
  CV_folds/
  SRF_runs/
  SRF_Ablation/
  SRF_Sensitivity/
  GLSRF_runs/
  Classic_runs/
```

## Input files

The workflow accepts two main inputs:

- `Vector/grade_full_D3.csv`: regular 3D grid used by the SRF workflow.
- `data_classic.xlsx`: point-support samples used by GLS-RF and the tree
  baselines.

Column names, missing-value rules, and derived-file names are documented in
[`DATA.md`](DATA.md).

## Parameters

Parameters are changed either through environment variables or in the
configuration section near the top of each script.

### Synthetic data — `make_demo_data.py`

Edit these constants when changing the demonstration dataset:

- `GRID`, `GRID_SMALL`: grid dimensions.
- `CELL_SIZE`, `ORIGIN`: grid geometry.
- `A_MAJOR`, `A_SEMI`, `A_MINOR`, `AZIMUTH_DEG`, `DIP_DEG`: continuity frame.
- `N_COLLARS`, `HOLE_LEN_NODES`: sampling layout.
- `ENVELOPE_FRACTION`, `FRAC_MISSING`: coverage and missingness.
- `SEED`: random seed.

Use `--small` to select `GRID_SMALL`:

```bash
python make_demo_data.py --small
```

### Voxel construction — `build_voxels.py`

Edit:

- `COLUMN_MAP` and `resp_cols` for input column names.
- `VOXEL_SIZES` for the kernel sizes to build.
- `MIN_COVERAGE` for accepted-window coverage.
- `CELL_SIZE`, `IDW_POWER`, `IDW_EPS` for grid spacing and imputation.

Run:

```bash
python build_voxels.py
```

After rebuilding voxel arrays, run `make_folds.py` again before training.

### Spatial folds — `make_folds.py`

Edit the configuration section containing:

- `DATASETS`, `PRIMARY`.
- `N_FOLDS`, `SEED`, `N_SEED_TRIALS`.
- `CELL_SIZE`, `N_DIV`, `MIN_BLOCK_SAMPLES`.
- `BUFFER_MODE`, `SAFETY_CELLS`.

Run:

```bash
python make_folds.py
```

### SRF training — `srf_train.py`

The most common settings can be supplied without editing the file:

```powershell
$env:SRF_PATTERN_VARIANT = "local_aniso"
$env:SRF_KERNEL_VARIANT = "k3"
$env:SRF_QUICK = "1"
$env:SRF_N_JOBS = "1"
python srf_train.py
```

Accepted pattern variants are `local`, `local_aug`, `local_aniso`,
`local_plain`, and `multiscale`. Kernel variants are `k3` and `k5`.

For a full run in PowerShell, remove the quick-run variable:

```powershell
Remove-Item Env:SRF_QUICK -ErrorAction SilentlyContinue
python srf_train.py
```

Edit the file when changing:

- `TASK`, `SPLIT_MODE`, `TARGET_COLS`.
- `CELL_SIZE`, `A_MAJOR`, `A_SEMI`, `A_MINOR`, `AZIMUTH_DEG`, `DIP_DEG`.
- `TUNE_MODE`, `N_SUBFOLDS`, `SUBFOLD_SEED`, `REFIT_SEED`.
- `PARAM_GRID`.
- `EXCEEDANCE_THRESHOLDS`.
- permutation-importance settings.

### SRF block-model prediction — `srf_predict.py`

Interactive run:

```bash
python srf_predict.py
```

Non-interactive run:

```python
from srf_predict import run_prediction

run_prediction(
    targets=["DTR"],
    rank_ids={"DTR": 1},
    interactive=False,
)
```

Edit `REFIT_ON`, `N_ESTIMATORS_OVERRIDE`, `BATCH_SIZE`, `USE_TTA`, and
`EXCEEDANCE_THRESHOLDS` when changing deployment behaviour. Keep exceedance
thresholds consistent with `srf_train.py`.

### GLS-RF — `GLSRF.py`

Edit:

- `FEATURE_COLS`, `TARGET_COLS`, `DATA_PATH_OVERRIDE`.
- `NUGGET`, `AZIMUTH_DEG`, `DIP_DEG`, `TILT_DEG`.
- `RANGE_MAJOR`, `RANGE_SEMI`, `RANGE_MINOR`, `KERNEL`.
- `COMPARE_DESIGNS`, `COMPARE_KERNELS`, `USE_RESIDUAL_KRIGING`.
- `N_ESTIMATORS_GRID`, `MAX_DEPTH_GRID`, `MIN_LEAF_GRID`.
- `MAX_FEATURES_GRID`, `MAX_SAMPLES_GRID`.
- `N_SUBFOLDS`, `SEED`, `N_JOBS`, `FOLDS_TO_RUN`.

Run a reduced or full search:

```bash
python GLSRF.py --smoke
python GLSRF.py
```

Create the fixed report and comparison figure after a completed run:

```bash
python glsrf_report.py [run_folder]
python GLSRF_figures.py [run_folder]
```

If `[run_folder]` is omitted, the most recent compatible run is used.

### Tree baselines — `classic_final.py`

Edit `FEATURE_COLS`, `TARGET_COLS`, `METHODS`, the method parameter grids,
`SEED`, and `FOLDS_TO_RUN`. Parallel workers can be set with
`CLASSIC_N_JOBS`:

```powershell
$env:CLASSIC_N_JOBS = "1"
python classic_final.py --smoke
```

Run the full grids without `--smoke`:

```bash
python classic_final.py
```

### Ablation and sensitivity

Edit `ARMS` in `srf_ablation_run.py` to change the ablation set.

```bash
python srf_ablation_run.py --smoke
python srf_ablation_run.py
python srf_ablation_run.py --force
```

Edit `PATTERN`, `KERNEL`, `N_JOBS`, `N_EST_FIXED`, `SWEEP`, and
`N_EST_CURVE` in `srf_sensitivity.py` before running:

```bash
python srf_sensitivity.py
```

## Reproducing the manuscript

Each numbered result maps to one script. Run them in this order; every step
after the first two consumes the outputs of the earlier ones.

| Manuscript item | Script |
| --- | --- |
| Table 1 — configurations and search spaces | `srf_train.py`, `GLSRF.py` |
| Table 2 — covariance design ranking | `glsrf_report.py` |
| Table 3 — GLS-RF out-of-bag vs test | `glsrf_report.py` |
| Figure 3 — GLS-RF design choices | `GLSRF_figures.py` |
| Table 4 — SRF design selection | `srf_ablation_run.py` |
| Section 3.4 — noise floor, search-space screening | `srf_sensitivity.py` |
| Table 5 — five methods, common support | `compare_methods.py` |
| Table 6 — distributional fidelity | `distributional_fidelity.py` |
| Table 7 — domain continuity at P65 | `spatial_coherence.py`, then `coherence_figure.py` |
| Table 8, Figure 4 — salt-and-pepper by 27-block support | `class_coherence.py` |
| Table S1 — continuity sensitivity | `coherence_sensitivity.py` |
| Figures 5, 6 — deployed block-model sections | `deploy_all_methods.py` |
| Methodology schematic | `methodology_figure.py` |

Full order:

```bash
python build_voxels.py
python make_folds.py

python srf_train.py                  # SRF_3D spatial CV
python GLSRF.py                      # GLS-RF_3D design + hyperparameter search
python glsrf_report.py               # the reported GLS-RF estimator
python classic_final.py              # the tree baselines

python compare_methods.py            # Table 5
python distributional_fidelity.py    # Table 6

python save_best_configs.py          # freeze the selected configurations
python deploy_all_methods.py         # deployed block model, all methods
python spatial_coherence.py          # connectivity and spikes
python class_coherence.py            # Table 8, Figure 4
python coherence_sensitivity.py      # Table S1
python coherence_figure.py           # Table 7 and the coherence figure
```

`compare_methods.py` resolves the newest run of each method automatically and
refuses to compute anything unless the three runs share a fold fingerprint, the
coordinate join is one-to-one within 0.5 m, and the folds agree for every
sample. Pass three run folders as arguments to pin specific runs instead.

## Output folders

| Script | Main output |
| --- | --- |
| `build_voxels.py` | `Vector/` |
| `make_folds.py` | `CV_folds/` |
| `srf_train.py` | `SRF_runs/SRF_run_<timestamp>/` |
| `srf_predict.py` | `SRF_runs/.../05_full_prediction/` |
| `GLSRF.py` | `GLSRF_runs/GLSRF_<data>_<timestamp>/` |
| `glsrf_report.py` | `GLSRF_runs/GLSRF_reported_ANISOexp/` |
| `classic_final.py` | `Classic_runs/TREES_<data>_<timestamp>/` |
| `srf_ablation_run.py` | `SRF_Ablation/` |
| `srf_sensitivity.py` | `SRF_Sensitivity/` |

## Data availability

The drillhole and block-model data of the case study are proprietary to the mine
operator and are not redistributed here. They are available from the
corresponding author on reasonable request, subject to the operator's
permission. `make_demo_data.py` generates a synthetic dataset with the same
schema and spatial structure, so the complete workflow can be run without them.

## Citation and license

Citation metadata are in [`CITATION.cff`](CITATION.cff); GitHub's "Cite this
repository" button reads it. Archived releases carry a DOI:
[10.5281/zenodo.22067693](https://doi.org/10.5281/zenodo.22069198). Please cite
the accompanying manuscript as well,Saha et al., 2023 and Talebi et al. (2021) for the original
SRF and GLS-RF formulation.

The code is available under the MIT License; see [`LICENSE`](LICENSE).
