# RF-STGN

Implementation of **Reliability-Calibrated Rough-Fuzzy Graph Learning for Forecast-Informed Electric-Vehicle Charging Scheduling**.

## Installation

Use Python 3.11+. Create and activate a virtual environment before installing:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

In PowerShell, replace the activation line with `.venv\Scripts\Activate.ps1`.

## Data and preprocessing

Obtain power/pilot and session records from [ACN-Data](https://ev.caltech.edu/dataset) through its JSON download interface or registered API. API time series require the `/ts` endpoint and an access token. Aggregate the acquired records into `slots.npz`:

| Key | Shape / meaning |
| --- | --- |
| `power` | `T,N`, mean metered EVSE power in kW per 15-minute slot |
| `time` | `T`, timezone-aware ISO timestamps, contiguous at 900 s |
| `connected` | `T,N`, connection indicators |
| `remaining_energy` | `T,N`, requested remaining session energy known at that slot |
| `weather` | `T,3`, contemporaneous temperature/humidity/one registered weather covariate |
| `cluster` | `N`, EVSE infrastructure cluster IDs |
| `price` | `T`, announced tariff per kWh |
| `base` | `T`, measured non-EV base load in kW |
| `split` | `T`, chronological nonempty `train`, `val`, `test` blocks |

Resolve missing current measurements using pilot/connection fields before aggregation; absent metering is not automatically zero demand. Convert current to power with the measured/declared voltage and phase convention, recording that convention with your data. ACN sessions alone do not supply weather, tariffs, or building base load; join those sources explicitly.

```bash
rf-stgn prepare --source slots.npz --output site.npz --capacity 225 --bootstrap 200 --seed 0
```

## Training and inference

```bash
rf-stgn train --data site.npz --output outputs/full --epochs 200 --seed 0
```

## Evaluation

```bash
rf-stgn evaluate --data site.npz --model outputs/full/model.pt --output outputs/full/evaluation
```

## Ablations

Supported controls: `no_reduction`, `fixed_correlation`, `point`, `lower`, `upper`, `mean_fusion`, `uniform_loss`. Reuse the same inputs, seed and budget. For retrained controls, evaluate the corresponding saved model.

```bash
rf-stgn train --data site.npz --output outputs/no_reduction --epochs 200 --seed 0 --variant no_reduction
```

## Outputs

Preparation writes `site.npz`; training writes `model.pt` and `history.json`. Evaluation writes `forecast.npz`, `metrics.json` and `forecast.png`.

## Scheduling replay

Create `sessions.json` as a list of records with `node` (EVSE index), `start`/`end` (prepared slot indices, end exclusive), `energy` (requested kWh), and `pmax` (kW). These values must be available at connection time; do not use future measured power as a demand forecast. The current slot uses measured base load; forecast horizon one is aligned to the next slot.

```bash
rf-stgn schedule --data site.npz --forecast outputs/full/evaluation/forecast.npz --sessions sessions.json --output outputs/full/scheduling --beta 0.5
rf-stgn schedule --data site.npz --forecast outputs/full/evaluation/forecast.npz --sessions sessions.json --output outputs/edf --strategy edf
```

Use `--beta 0` for no-reserve scheduling and `--strategy tou_edf` or `uncontrolled` for the other implemented controls. Forecast controls use `rf-stgn evaluate ... --baseline persistence` (also `seasonal`, `oracle`).

## Model and metric conventions

Preparation fits rough reduction and bounded graph evidence on training data. Forecasting combines context-weighted graph branches, a GRU and weighted Huber loss. Evaluation reports horizon MAE/RMSE/WAPE and reserve calibration; scheduling writes `schedule.json`, operational metrics and `schedule.png`. Decision weights default to `(1,1,1)`.

On Windows Anaconda, set process-local `MKL_THREADING_LAYER=SEQUENTIAL` when competing Intel OpenMP runtimes cause a load error.

## Validation selection and forecasting comparisons

Install optional comparison dependencies and fetch pinned author sources (kept locally under ignored `data/upstream`, with their original licenses):

```bash
python -m pip install -e ".[baselines]"
python -m rf_stgn.baselines fetch
python -m rf_stgn.baselines grid --data site.npz --output outputs/selected --values 0 0.5 1 --epochs 200 --seed 0
rf-stgn evaluate --data site.npz --model outputs/selected/model.pt --output outputs/selected/evaluation
rf-stgn train --data site.npz --output outputs/dcrnn --architecture dcrnn --epochs 200 --seed 0
rf-stgn evaluate --data site.npz --model outputs/dcrnn/model.pt --output outputs/dcrnn/evaluation
```

The grid trains all 27 `(load, price, occupancy)` combinations with identical seeds and budgets, selects the saved epoch/trial by **validation** 12-step RMSE, and records every score in `selection.json`. Test data are used only by the subsequent `evaluate`. `--max-trials 2 --epochs 1 --width 8` selects a small grid check. The constant one in the weighted loss remains for the all-zero setting.

`--architecture` accepts `rf_stgn`, `gru`, `dcrnn`, `graph_wavenet`, `astgrn_adapted`, `timesnet`, `itransformer`, and `xgboost`. All comparisons use the same 96-slot histories, 12 future targets, train-only scaling, validation checkpoint choice, forecast metrics, reserve calibration, and scheduling interface. Comparison methods retain all 28 predictors; no rough reducer or rough-fuzzy edge channels are added. Neural defaults are width 32 and the CLI epoch/batch budget; XGBoost uses 200 depth-6 histogram trees per output with one CPU thread (`--trees`, `--tree-depth` configure it).

- [DCRNN](https://github.com/liyaguang/DCRNN): a local PyTorch implementation of forward/backward random-walk diffusion (order 2), graph-GRU encoder and autoregressive decoder. Curriculum teacher forcing uses only training targets, with inverse-sigmoid decay 1000; evaluation is fully autoregressive. The static common graph is the row-normalized mean of training correlation, connection co-occurrence and infrastructure-cluster graphs plus self-loops.
- [Graph WaveNet author implementation](https://github.com/nnzhan/Graph-WaveNet), pinned at `6b162e80c59a1d494809252eca055cff93dc66b1`: actual `gwnet`, adaptive adjacency enabled, 2 blocks/6 layers (receptive field 127 covers 96). Its legacy four-dimensional gate/residual/skip `Conv1d` modules are instantiated as `Conv2d` with the same kernels and parameters for current PyTorch. The fetched source is unmodified.
- [TimesNet and iTransformer author library](https://github.com/thuml/Time-Series-Library), pinned at `4e938a1767106324dd753b2a44832bf870a0252e`: actual author `Model` classes with two encoder layers and a shared linear readout from node-feature channels to station power; TimesNet uses top-k 3 and three inception kernels. Hidden width is configurable.
- `astgrn_adapted` uses learned 8-dimensional node embeddings, query/key adaptive adjacency mixed with the static graph, and a diffusion-GRU encoder/decoder. Related work: [ASTGRN](https://ira.lib.polyu.edu.hk/handle/10397/116696).
- GRU uses the same flattened station-feature history and direct 12-step readout; [XGBoost](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBRegressor) fits each horizon/station target from training origins only.

## Checks

```bash
python smoke.py --baselines
```

Synthetic checks cover forecasting gradients, checkpoint replay, XGBoost fitting, validation-only selection, constrained scheduling and next-slot alignment.
