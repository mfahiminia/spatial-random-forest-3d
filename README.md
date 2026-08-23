# 3D Spatial Random Forest Workflows

Python scripts for preparing 3D voxel features, creating spatial folds, training
SRF and GLS-RF models, predicting a block model, and running tree baselines.

The case-study data are not included. A synthetic dataset is provided so the
workflow can be tested from start to finish. See [`DATA.md`](DATA.md) for the
required input schema and [`USAGE.md`](USAGE.md) for detailed troubleshooting.

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

## Citation and license

Citation metadata are stored in [`CITATION.cff`](CITATION.cff). Complete its
release URL, author information, and publication fields before creating the
public release.

The code is available under the MIT License; see [`LICENSE`](LICENSE).
