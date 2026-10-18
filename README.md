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

Source inspection: Python 3.13.5, numpy 2.1.3, scipy 1.15.3, torch 2.7.1+cu118, matplotlib 3.10.0 (installed versions inspected). Checks covered Python syntax, configuration parsing and CLI arguments. No training, full evaluation or CI was run in this update.
