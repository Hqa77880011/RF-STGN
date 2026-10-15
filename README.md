# RF-STGN

Implementation of **Reliability-Calibrated Rough-Fuzzy Graph Learning for Forecast-Informed Electric-Vehicle Charging Scheduling**.

The model combines neighborhood-rough feature selection, bootstrap correlation bounds, arrival/cluster fuzzy relations, shared-weight dual graph convolutions, context fusion, GRU decoding, and decision-weighted Huber learning. Validation residuals supply a calibrated site-demand reserve to a rolling quadratic charging scheduler.

## Install and prepare ACN data

```bash
python -m venv .venv
python -m pip install -e .
```

Python 3.11+ is required. Retrieve Caltech/JPL sessions through [ACN-Data](https://ev.caltech.edu/dataset), preserving UTC timestamps and cluster/EVSE identities. Use API records rather than historical web exports affected by timezone conversion. The paper uses 54 Caltech nodes and 50 JPL nodes, 15-minute slots from January 2019 through February 2020, training through September, validation in October/November, and testing from December. Keep each site in a separate prepared file.

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
rf-stgn prepare --source slots.npz --output site.npz --capacity 225 --bootstrap 200
rf-stgn train --data site.npz --output outputs/full --epochs 200 --seed 0
```

Preparation constructs 28 predictors: twelve load lags, six rolling mean/std statistics at 4/16/96 slots, five calendar terms, three weather values, and two session states. Training-only statistics standardize features and targets. The rough reducer uses quartile site-load decisions, radius 0.2, dependency gain 0.005, and a seeded maximum of 1000 sampled node/origin pairs. A first feature is retained even when initial gains are zero so the neural input is defined. Increase `--reduction-samples` to reduce the subsampling approximation.

Correlation bounds use day-block resampling of the preceding 28 days and 5th/95th percentiles. They update daily using information strictly before that day; each input window uses its forecast-origin bounds across its 96 input slots. Arrival histograms and cluster relations are fitted from training only. This daily update convention reduces graph-construction cost without using future targets. The model retains eight neighbors on upper membership and applies the same support to both branches. Context mixture weights vary over the input window. The GRU width is 32, the output horizon twelve, and Adam uses 0.001 with 1e-5 weight decay. Early stopping uses validation RMSE with patience 20.

## Baselines and component controls

```bash
rf-stgn train --data site.npz --output outputs/point --variant point
rf-stgn train --data site.npz --output outputs/mean --variant mean_fusion
rf-stgn train --data site.npz --output outputs/uniform --variant uniform_loss
rf-stgn evaluate --data site.npz --model outputs/full/model.pt --output outputs/full_eval
rf-stgn evaluate --data site.npz --model outputs/full/model.pt --output outputs/persistence --baseline persistence
rf-stgn evaluate --data site.npz --model outputs/full/model.pt --output outputs/seasonal --baseline seasonal
```

Other training variants are `no_reduction`, `fixed_correlation`, `lower`, and `upper`. Fixed correlation uses a single positive-correlation graph fitted on training only. `point` averages relation bounds, `mean_fusion` removes learned branch selection, and `uniform_loss` removes future load/price/occupancy weighting. Persistence repeats current measured load; seasonal persistence uses the same slots one day earlier. `--baseline oracle` supplies future measured targets as a separately labeled retrospective scheduling reference. Every run uses the same partitions and forecast origins.

`forecast.npz` stores origins, twelve-step predictions/truth, and validation-only 90th-percentile site residuals. `metrics.json` reports MAE/RMSE/WAPE at horizons 1/4/12, calibrated interval coverage/width, retained features, parameter count, and measured forecast time. It also reports RMSE for site loads above the training 90th percentile, the highest training tariff, and occupancy at least 80%. `history.json` records validation error and epoch time. `forecast.png` plots the one-step site forecast. Use `--split val` to evaluate reserve settings before touching test results.

## Forecast-informed scheduling

Create `sessions.json`: a list of objects with `start`, `end` (exclusive integer slot), `node`, `energy` (requested battery-side kWh), and `pmax` (kW). These fields represent information available after connection; departures and requested energy are assumed known for currently active sessions.

```bash
rf-stgn schedule --data site.npz --forecast outputs/full_eval/forecast.npz --sessions sessions.json --output outputs/mpc --beta 0.5
rf-stgn schedule --data site.npz --forecast outputs/full_eval/forecast.npz --sessions sessions.json --output outputs/no_reserve --beta 0
rf-stgn schedule --data site.npz --forecast outputs/seasonal/forecast.npz --sessions sessions.json --output outputs/seasonal_mpc --beta 0
rf-stgn schedule --data site.npz --forecast outputs/full_eval/forecast.npz --sessions sessions.json --output outputs/edf --strategy edf
rf-stgn schedule --data site.npz --forecast outputs/full_eval/forecast.npz --sessions sessions.json --output outputs/uncontrolled --strategy uncontrolled
```

The convex QP minimizes squared peak, tariff cost, 0.01 load-ramp penalty, and 1000 times unserved-energy slack. It keeps port and transformer constraints hard, uses efficiency 0.92 and a twelve-slot horizon, and applies only the first command. The active-session continuation profile is subtracted from predicted demand before arrival reserve is added. Future non-EV load uses current-load persistence; the applied first step uses measured base load. SLSQP solves the small QP and raises an error if it cannot satisfy the declared hard constraints. An unavoidable base load above site capacity requires correcting the scheduling instance.

Select `beta` from 0/0.5/1 on validation scheduling cost, peak, and unserved energy, then freeze it for test replay. The implementation supplies the three settings directly; it does not select reserve strength on test labels. EDF is a capacity-limited earliest-departure control. `--strategy tou_edf` waits for the lowest announced remaining-session tariff unless charging now is required to preserve deadline feasibility, allocating available capacity in deadline order. Uncontrolled replay follows recorded EVSE power and can violate the transformer cap.

`schedule.json`, `metrics.json`, and `schedule.png` expose peak kW, tariff cost, unserved kWh at departures within the replay, transformer violation rate, and measured solve time. Sessions already active at replay start subtract their recorded pre-start delivery. Sessions departing after replay end are censored from unserved energy. Compare matching session sets and forecast horizons, and report multiple seeds/sites separately. No experimental scores are bundled.
