# Data

The deposit data behind the paper is not redistributed with this repository.
The repository is still fully runnable: `make_demo_data.py` fabricates a
synthetic stand-in with the same schema, and every other script then runs
unchanged. This file documents the schema precisely, so you can substitute the
real data, your own deposit, or a modified simulation.

```bash
python make_demo_data.py     # writes both files below
```

---

## Availability of the case-study data

The drillhole and block-model data are the property of the mine operator and are
not public. They are available from the corresponding author on reasonable
request, subject to the operator's permission.

Because the data are withheld, **this repository cannot reproduce the paper's
numbers on its own.** What it does provide is the complete method: every
modelling decision, hyperparameter, seed and evaluation rule is in the code, and
with the case-study files in place (`SRF_PROJECT_ROOT` pointing at them) the
pipeline regenerates the reported folds and the reported results exactly.

---

## File 1 — `Vector/grade_full_D3.csv`

The estimated grid table: one row per node of a regular 3D grid. This is the
input to `build_voxels.py`, and it defines the grid the voxel kernels slide over.

| Column | Type | Meaning |
| --- | --- | --- |
| `XC`, `YC`, `ZC` | float | Node centre coordinates, metres. Must lie on a regular grid. |
| `Fe` | float | Iron grade, % |
| `FeO` | float | Ferrous iron, % |
| `MagSus` | float | Magnetic susceptibility (SI × 10⁻³) |
| `S` | float | Sulphur, % |
| `DTR` | float | Davis Tube Recovery, % — response 1 |
| `Magnetic` | float | Magnetic concentrate grade, % — response 2 |
| `IJK` | string | *(optional)* grid indices as `i_j_k`; if absent, indices are derived from the unique coordinate values |

Rules:

* **Every node of the grid must appear as a row**, including nodes with no data.
  The grid dimensions are inferred from the unique `XC`/`YC`/`ZC` values (or from
  `IJK`), so omitting rows silently shrinks the grid.
* Missing values may be blank, `NaN`, or the literal `-`; all three are read as
  missing.
* Rows with a missing coordinate are dropped.
* Covariates are typically present only inside the estimated (mineralised)
  domain. That is expected — windows failing the coverage rule are skipped.
* The two responses are typically present only at sampled locations. A node
  becomes a *training* pattern only if **both** responses are present at its
  centre; all accepted windows become *inference* rows regardless.

Cell size is not read from the file. It is set in `build_voxels.py` and
`make_folds.py` as `CELL_SIZE = (5.0, 5.0, 2.0)` metres, and must match the
grid spacing in the CSV.

## File 2 — `data_classic.xlsx`

The point-support sample table. `make_folds.py` reads it so that the
point-support methods receive the identical spatial folds as the voxel methods.
Only `XC`, `YC`, `ZC` are required by the fold builder; the covariate and
response columns are there for the point-support models themselves.

| Column | Required by | Meaning |
| --- | --- | --- |
| `XC`, `YC`, `ZC` | `make_folds.py`, `GLSRF.py` | Sample coordinates, metres |
| `FE`, `FEO`, `Magsus`, `S` | `GLSRF.py` | Covariates |
| `DTR`, `Magnetic` | `GLSRF.py` | Responses |

**Note the spelling.** This file uses `FE`, `FEO`, `Magsus`; the grid CSV above
uses `Fe`, `FeO`, `MagSus`. That difference is inherited from the case-study
files and is preserved deliberately — `GLSRF.FEATURE_COLS` expects the former
and `build_voxels.COLUMN_MAP` the latter, and `make_demo_data.py` reproduces
both. Rename in those two constants if your files differ.

The coordinate column names are configurable in `make_folds.py`
(`DATASETS["classic"]["coord_cols"]`) and in `GLSRF.COORD_COLS`.

If you have no point-support table, delete the `"classic"` entry from `DATASETS`
in `make_folds.py`; the SRF pipeline itself does not use it.

---

## Derived files

Everything below is produced by the scripts and is not input data.

| Path | Produced by | Notes |
| --- | --- | --- |
| `Vector/X_train_{27,125}.npy` | `build_voxels.py` | (N, k³·5) voxel patterns |
| `Vector/y_train_{27,125}.npy` | `build_voxels.py` | (N, 2) → `[DTR, Magnetic]` |
| `Vector/centers_train_*.npy` | `build_voxels.py` | (N, 3) pattern-centre coordinates |
| `Vector/X_infer_{27,125}.npy` | `build_voxels.py` | all accepted windows, for deployment. Large; git-ignored |
| `Vector/Grid_*.npy` | `build_voxels.py` | the gridded covariates and responses |
| `CV_folds/{k3,k5,classic}/roles_*.npy` | `make_folds.py` | (N, K) int8: 0=train, 1=test, 2=buffer, −1=excluded |
| `CV_folds/{k3,k5,classic}/folds_*.csv` | `make_folds.py` | same, plus coordinates and `block_id` |
| `CV_folds/fold_config.json` | `make_folds.py` | the full fold definition, including `block_to_fold` |
| `GLSRF_runs/GLSRF_*/` | `GLSRF.py` | grid tables, fold metrics, prediction table |
| `GLSRF_runs/GLSRF_reported_ANISOexp/` | `glsrf_report.py` | the reported GLS-RF estimator |
| `Classic_runs/TREES_*/` | `classic_final.py` | the tree baselines: search tables, fold metrics, predictions |

Row order is the contract that ties these together: row *i* of `X_train_27.npy`,
`y_train_27.npy`, `centers_train_27.npy` and `CV_folds/k3/roles_k3.npy` is the
same sample. `srf_train.py` asserts the row counts agree, so a stale folds
directory fails loudly rather than silently mislabelling folds.

---

## Substituting your own deposit

1. Write your grid table to `Vector/grade_full_D3.csv` with the columns above.
   Rename covariates in `build_voxels.py` (`COLUMN_MAP`) if yours differ; the
   internal names `FE / FEO / Magsus / S` propagate to `srf_train.PROPS_COV`.
2. Set `CELL_SIZE` in `build_voxels.py` **and** `make_folds.py` to your grid
   spacing, and `srf_train.CELL_SIZE` to match.
3. Set the continuity ellipsoid in `srf_train.py` (`A_MAJOR`, `A_SEMI`,
   `A_MINOR`, `AZIMUTH_DEG`, `DIP_DEG`) from your own variography. The
   anisotropy features are built in that frame, so a wrong frame costs accuracy.
4. Re-check the block layout. `N_DIV` in `make_folds.py` is chosen so the CV
   test-to-train distance distribution matches the deployment
   node-to-sample distance distribution; the script prints both, and writes
   `CV_folds/design_distance_match.png`. Adjust `N_DIV` until they agree — that
   comparison, not the specific value `(3, 1, 6)`, is what transfers.
5. Set `EXCEEDANCE_THRESHOLDS` in `srf_train.py` and `srf_predict.py` to cutoffs
   that actually occur in your responses.
6. For GLS-RF, set Section 2 of `GLSRF.py` — nugget, orientation, ranges and
   correlation function — from the same variography. The nugget matters most:
   it is the share of variance that is *not* spatially structured, read off the
   experimental variogram, and at nugget 0 the model asserts near-perfect
   correlation between neighbouring samples and inverting Σ amplifies noise.
   Leave coordinates out of `FEATURE_COLS`; that exclusion is a modelling
   decision, not an oversight.
