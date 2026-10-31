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

Create `slots.npz` from measured power/pilot and session records:

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

## Implementation and functional validation

The existing manuscript-based implementation is reused. This update requires timezone-aware timestamps, fixes the forecast/scheduling horizon offset, and adds a functional check. The local paper directory contains result-figure code but no original training/session data; this code is not evidence for the reported results.

| Paper module / metric | Entry | Command / output |
| --- | --- | --- |
| Rough reduction / bounded graph evidence | `prepare`, `reduct` | `rf-stgn prepare ...` |
| Context-weighted graph branches / GRU / weighted Huber | `Forecast`, `train` | `rf-stgn train ...` |
| Horizon MAE/RMSE/WAPE and reserve calibration | `evaluate` | `forecast.npz`, `metrics.json` |
| Fixed rolling scheduling and operational metrics | `plan`, `schedule` | `schedule.json`, `metrics.json`, `schedule.png` |
| Functional validation | `smoke.py` | `python smoke.py` |

CPU verification uses Python 3.13.5 / torch 2.7.1+cu118, seed 0 and threads 1. On the Windows Anaconda environment set `MKL_THREADING_LAYER=SEQUENTIAL` in the command process to avoid loading competing Intel OpenMP runtimes. The small synthetic check covers actual NPZ preparation, training-only scaling/reduction, one-epoch training, graph/GRU gradients and updates, checkpoint equality, persistence/oracle metrics, real constrained SLSQP scheduling and an explicit next-slot alignment check. No full ACN campaign or reported performance is reproduced. Decision weights default to `(1,1,1)`; the manuscript's validation grid and all external graph-network baselines are not automated here.
