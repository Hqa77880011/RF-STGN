import argparse
import datetime as dt
import json
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import Bounds, LinearConstraint, minimize
from scipy.spatial.distance import cdist
from torch import nn


def reduct(x, decision, radius=.2, epsilon=.005):
    selected, dependency = [], 0.
    remaining = list(range(x.shape[1]))
    while remaining:
        scores = []
        for feature in remaining:
            trial = selected + [feature]
            neighborhood = cdist(x[:, trial], x[:, trial]) / np.sqrt(len(trial)) <= radius
            pure = np.all(~neighborhood | (decision[:, None] == decision[None]), axis=1)
            scores.append(float(pure.mean()))
        winner = int(np.argmax(scores))
        if selected and scores[winner] - dependency < epsilon:
            break
        dependency = scores[winner]
        selected.append(remaining.pop(winner))
    return selected


def prepare(args):
    raw = np.load(args.source)
    power = np.asarray(raw['power'], dtype=np.float32)
    times = [dt.datetime.fromisoformat(str(v).replace('Z', '+00:00')) for v in raw['time']]
    if any(stamp.utcoffset() is None for stamp in times):
        raise ValueError('time values must include an explicit timezone offset')
    times = [stamp.astimezone(ZoneInfo(args.timezone)) for stamp in times]
    n = power.shape[1]
    if power.ndim != 2 or not np.isfinite(power).all() or (power < 0).any():
        raise ValueError('power must be finite nonnegative slots,EVSE and resolved from real metering/pilot data')
    if any(abs((b.astimezone(dt.timezone.utc) - a.astimezone(dt.timezone.utc)).total_seconds() - 900) > .01 for a, b in zip(times[:-1], times[1:])):
        raise ValueError('Slots must be contiguous fifteen-minute intervals')
    connected = raw['connected'].astype(float)
    features, context = [], []
    for t, stamp in enumerate(times):
        lags = [power[max(0, t - lag)] for lag in range(12)]
        rolling = []
        for width in [4, 16, 96]:
            past = power[max(0, t - width + 1):t + 1]
            rolling += [past.mean(0), past.std(0)]
        hour = stamp.hour + stamp.minute / 60
        calendar = np.array([np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
                             np.sin(2 * np.pi * stamp.weekday() / 7), np.cos(2 * np.pi * stamp.weekday() / 7), float(stamp.weekday() < 5)])
        weather = np.asarray(raw['weather'][t])
        if weather.shape != (3,):
            raise ValueError('weather has three registered contemporaneous variables')
        features.append(np.column_stack(lags + rolling + [np.full(n, v) for v in calendar] +
                                        [np.full(n, v) for v in weather] + [connected[t], raw['remaining_energy'][t]]))
        context.append([calendar[0], calendar[1], calendar[-1], power[t].sum() / args.capacity])
    x = np.asarray(features, dtype=np.float32)
    split = np.asarray(raw['split'])
    train = split == 'train'
    if set(split) != {'train', 'val', 'test'} or np.any(np.diff(np.array([{'train': 0, 'val': 1, 'test': 2}[s] for s in split])) < 0):
        raise ValueError('Partitions must be nonempty chronological train,val,test blocks')
    mean, std = x[train].mean((0, 1)), x[train].std((0, 1)).clip(1e-6)
    x = (x - mean) / std
    target_mean, target_std = power[train].mean(), max(power[train].std(), 1e-6)
    origins = np.array([t for t in range(95, len(x) - 12) if np.all(split[t - 95:t + 13] == split[t])])
    rng = np.random.default_rng(args.seed)
    train_origins = origins[split[origins] == 'train']
    candidates = x[train_origins].reshape(-1, 28)
    next_site = np.repeat(power[train_origins + 1].sum(1), n)
    sample = rng.choice(len(candidates), min(args.reduction_samples, len(candidates)), replace=False)
    quartiles = np.quantile(next_site, [.25, .5, .75])
    selected = reduct(candidates[sample], np.digitize(next_site[sample], quartiles))
    hour_bin = np.array([s.hour * 4 + s.minute // 15 for s in times])
    arrivals = connected > np.roll(connected, 1, axis=0)
    arrivals[0] = False
    histogram = np.stack([np.bincount(hour_bin[train & arrivals[:, j]], minlength=96) for j in range(n)]).astype(float)
    histogram /= np.maximum(np.linalg.norm(histogram, axis=1, keepdims=True), 1e-8)
    similarity = histogram @ histogram.T
    cluster = raw['cluster'][:, None] == raw['cluster'][None]
    low, high, cached = [], [], {}
    for origin in origins:
        day = origin // 96
        if day not in cached:
            end = max(1, day * 96)
            past = power[max(0, end - 28 * 96):end]
            blocks = [past[start:start + 96] for start in range(0, len(past) - 95, 96)]
            correlations = []
            for _ in range(args.bootstrap):
                sample_power = np.concatenate([blocks[int(i)] for i in rng.integers(len(blocks), size=len(blocks))]) if blocks else past
                # With fewer than two historical slots there is no inter-EVSE correlation evidence.
                correlations.append(np.nan_to_num(np.corrcoef(sample_power.T)) if len(sample_power) >= 2 else np.eye(n))
            cached[day] = np.maximum(np.quantile(correlations, [.05, .95], axis=0), 0).astype(np.float32)
        low.append(cached[day][0])
        high.append(cached[day][1])
    np.savez_compressed(args.output, x=x, power=power, context=np.asarray(context, dtype=np.float32), split=split,
                        origins=origins, low=low, high=high, arrival=similarity.astype(np.float32), cluster=cluster.astype(np.float32),
                        fixed=np.maximum(0, np.nan_to_num(np.corrcoef(power[train].T))).astype(np.float32),
                        selected=selected, price=raw['price'], base=raw['base'], connected=connected,
                        mean=mean, std=std, target_mean=target_mean, target_std=target_std, capacity=args.capacity)


class Forecast(nn.Module):
    def __init__(self, features, variant='full', width=32):
        super().__init__()
        self.variant = variant
        self.relation = nn.Linear(4, 3)
        self.layers = nn.ModuleList([nn.Linear(features, width), nn.Linear(width, width)])
        self.fusion = nn.Linear(width * 2 + 4, width)
        self.temporal = nn.GRU(width, width, batch_first=True)
        self.head = nn.Linear(width, 12)

    def forward(self, x, context, low, high, arrival, cluster):
        batch, ticks, nodes, _ = x.shape
        alpha = self.relation(context).softmax(-1)
        graphs = []
        for correlation in [low, high]:
            graph = (alpha[..., 0, None, None] * correlation[:, None] +
                     alpha[..., 1, None, None] * arrival + alpha[..., 2, None, None] * cluster)
            graphs.append(graph)
        if self.variant == 'fixed_correlation':
            graphs = [low[:, None].expand(-1, ticks, -1, -1)] * 2
        if self.variant == 'point':
            graphs = [(graphs[0] + graphs[1]) / 2] * 2
        if self.variant == 'lower':
            graphs[1] = graphs[0]
        if self.variant == 'upper':
            graphs[0] = graphs[1]
        count = min(8, nodes)
        indices = graphs[1].topk(count, dim=-1).indices
        support = torch.zeros_like(graphs[1]).scatter_(-1, indices, 1.)
        branches = []
        for graph in graphs:
            graph = graph * support + torch.eye(nodes, device=x.device)
            degree = graph.sum(-1).clamp_min(1e-8).rsqrt()
            graph = graph * degree[..., :, None] * degree[..., None, :]
            h = x
            for layer in self.layers:
                h = torch.matmul(graph, layer(h)).relu()
            branches.append(h)
        gate = torch.sigmoid(self.fusion(torch.cat([*branches, context[:, :, None].expand(-1, -1, nodes, -1)], -1)))
        if self.variant == 'mean_fusion':
            gate = torch.full_like(gate, .5)
        fused = gate * branches[0] + (1 - gate) * branches[1]
        sequence = fused.transpose(1, 2).reshape(batch * nodes, ticks, -1)
        _, hidden = self.temporal(sequence)
        return self.head(hidden[0]).reshape(batch, nodes, 12).transpose(1, 2)


def tensors(data, rows, selected, variant='full'):
    origins = data['origins'][rows]
    x = np.stack([data['x'][t - 95:t + 1][:, :, selected] for t in origins])
    context = np.stack([data['context'][t - 95:t + 1] for t in origins])
    lower, upper = data['low'][rows], data['high'][rows]
    if variant == 'fixed_correlation':
        lower = np.repeat(data['fixed'][None], len(rows), axis=0)
        upper = lower
    return [torch.tensor(v, dtype=torch.float32) for v in [x, context, lower, upper, data['arrival'], data['cluster']]]


def predict(model, data, rows, selected, batch):
    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch):
            predictions.append(model(*tensors(data, rows[start:start + batch], selected, model.variant)).numpy())
    return np.concatenate(predictions) * float(data['target_std']) + float(data['target_mean'])


def train(args):
    torch.manual_seed(args.seed)
    d = np.load(args.data)
    selected = np.arange(d['x'].shape[-1]) if args.variant == 'no_reduction' else d['selected']
    model = Forecast(len(selected), args.variant)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=1e-5)
    origin = d['origins']
    train_rows = np.flatnonzero(d['split'][origin] == 'train')
    val_rows = np.flatnonzero(d['split'][origin] == 'val')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    best, patience, history = float('inf'), 0, []
    for epoch in range(args.epochs):
        begin = time.perf_counter()
        model.train()
        for start in range(0, len(train_rows), args.batch):
            rows = train_rows[start:start + args.batch]
            target = np.stack([d['power'][t + 1:t + 13] for t in origin[rows]])
            price = np.stack([d['price'][t + 1:t + 13] for t in origin[rows]])
            occupied = np.stack([d['connected'][t + 1:t + 13].mean(1) for t in origin[rows]])
            price_scale = max(float(np.ptp(d['price'][d['split'] == 'train'])), 1e-8)
            weight = 1 + target.sum(-1) / float(d['capacity']) + (price - d['price'][d['split'] == 'train'].min()) / price_scale + occupied
            if args.variant == 'uniform_loss':
                weight[:] = 1
            target_tensor = torch.tensor((target - d['target_mean']) / d['target_std'], dtype=torch.float32)
            prediction = model(*tensors(d, rows, selected, args.variant))
            error = nn.functional.huber_loss(prediction, target_tensor, reduction='none')
            w = torch.tensor(weight, dtype=torch.float32)
            loss = (error * w[:, :, None]).sum() / (target.shape[-1] * w.sum())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        prediction = predict(model, d, val_rows, selected, args.batch)
        truth = np.stack([d['power'][t + 1:t + 13] for t in origin[val_rows]])
        rmse = float(np.sqrt(np.square(prediction - truth).mean()))
        history.append({'epoch': epoch + 1, 'val_rmse': rmse, 'seconds': time.perf_counter() - begin})
        if rmse < best:
            best, patience = rmse, 0
            residual = np.abs(truth.sum(-1) - prediction.sum(-1))
            torch.save({'state': model.state_dict(), 'selected': selected, 'variant': args.variant,
                        'reserve': np.quantile(residual, .9, axis=0, method='higher')}, out / 'model.pt')
        else:
            patience += 1
        if patience >= 20:
            break
    (out / 'history.json').write_text(json.dumps(history, indent=2))


def load_model(path):
    checkpoint = torch.load(path, weights_only=False, map_location='cpu')
    model = Forecast(len(checkpoint['selected']), checkpoint['variant'])
    model.load_state_dict(checkpoint['state'])
    return model, checkpoint


def evaluate(args):
    d = np.load(args.data)
    model, checkpoint = load_model(args.model)
    rows = np.flatnonzero(d['split'][d['origins']] == args.split)
    begin = time.perf_counter()
    prediction = predict(model, d, rows, checkpoint['selected'], args.batch).clip(0)
    latency = (time.perf_counter() - begin) * 1000 / len(rows)
    truth = np.stack([d['power'][t + 1:t + 13] for t in d['origins'][rows]])
    if args.baseline == 'persistence':
        prediction = np.stack([np.repeat(d['power'][t:t + 1], 12, axis=0) for t in d['origins'][rows]])
    elif args.baseline == 'seasonal':
        prediction = np.stack([d['power'][t - 95:t - 83] for t in d['origins'][rows]])
    elif args.baseline == 'oracle':
        prediction = truth.copy()
    metrics = {}
    for h in [1, 4, 12]:
        error = prediction[:, h - 1] - truth[:, h - 1]
        metrics[str(h)] = {'mae': float(np.abs(error).mean()), 'rmse': float(np.sqrt(np.square(error).mean())),
                           'wape': float(100 * np.abs(error).sum() / max(1e-8, np.abs(truth[:, h - 1]).sum()))}
    site = prediction.sum(-1)
    low, high = np.maximum(0, site - checkpoint['reserve']), site + checkpoint['reserve']
    metrics.update(interval_coverage=float(((truth.sum(-1) >= low) & (truth.sum(-1) <= high)).mean()),
                   interval_width=float((high - low).mean()), parameters=sum(p.numel() for p in model.parameters()),
                   retained_features=len(checkpoint['selected']), forecast_ms_per_origin=latency)
    train_threshold = np.quantile(d['power'][d['split'] == 'train'].sum(1), .9)
    target_slots = d['origins'][rows, None] + np.arange(1, 13)
    masks = {'peak': truth.sum(-1) >= train_threshold,
             'high_price': d['price'][target_slots] >= d['price'][d['split'] == 'train'].max(),
             'high_occupancy': d['connected'][target_slots].mean(-1) >= .8}
    for name, selected in masks.items():
        metrics[name + '_rmse'] = float(np.sqrt(np.square(prediction - truth)[selected].mean())) if selected.any() else None
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    np.savez_compressed(out / 'forecast.npz', origin=d['origins'][rows], prediction=prediction, truth=truth, reserve=checkpoint['reserve'])
    plt.plot(truth[:, 0].sum(-1), label='Measured')
    plt.plot(prediction[:, 0].sum(-1), label='Forecast')
    plt.legend()
    plt.ylabel('Site power (kW)')
    plt.xlabel('Forecast origin')
    plt.tight_layout()
    plt.savefig(out / 'forecast.png', dpi=180)
    plt.close()


def plan(active, remaining, horizon, price, exogenous, capacity, now):
    n = len(active)
    if not n:
        return np.array([])
    size = n * horizon
    available = np.array([[now + h < s['end'] for h in range(horizon)] for s in active])
    max_power = np.array([s['pmax'] for s in active])
    required = np.maximum(0, remaining - .92 * .25 * max_power * np.maximum(np.array([s['end'] for s in active]) - now - horizon, 0))
    total = np.zeros((horizon, size + 1 + n))
    for h in range(horizon):
        total[h, np.arange(n) * horizon + h] = 1
        total[h, size] = -1
    energy = np.zeros((n, size + 1 + n))
    delivery = np.zeros_like(energy)
    for i in range(n):
        energy[i, i * horizon:(i + 1) * horizon] = .92 * .25
        delivery[i, i * horizon:(i + 1) * horizon] = .92 * .25
        energy[i, size + 1 + i] = 1
    upper = np.r_[(available * max_power[:, None]).ravel(), capacity, np.full(n, np.inf)]
    initial = np.r_[np.zeros(size), capacity, required]
    def objective(v):
        site = exogenous + v[:size].reshape(n, horizon).sum(0)
        return v[size] ** 2 + .25 * price @ site + .01 * np.square(np.diff(site)).sum() + 1000 * v[size + 1:].sum()
    solution = minimize(objective, initial, method='SLSQP', bounds=Bounds(np.zeros_like(upper), upper),
                        constraints=[LinearConstraint(total, -np.inf, -exogenous), LinearConstraint(energy, required, np.inf),
                                     LinearConstraint(delivery, -np.inf, remaining)],
                        options={'maxiter': 300, 'ftol': 1e-7})
    if not solution.success:
        raise RuntimeError('Scheduling QP failed: ' + solution.message)
    command = solution.x[:size].reshape(n, horizon)[:, 0]
    room = max(0., capacity - exogenous[0])
    if command.sum() > room:
        command *= room / command.sum()
    return command


def schedule(args):
    d, f = np.load(args.data), np.load(args.forecast)
    sessions = json.loads(Path(args.sessions).read_text())
    origins = f['origin']
    start, stop = int(origins.min()), int(origins.max()) + 1
    remaining = np.array([s['energy'] for s in sessions], dtype=float)
    for i, s in enumerate(sessions):
        if s['start'] < start:
            delivered = .92 * .25 * d['power'][max(0, s['start']):min(start, s['end']), s['node']].sum()
            remaining[i] = max(0., remaining[i] - delivered)
    lookup = {int(t): i for i, t in enumerate(origins)}
    trace, unserved = [], 0.
    for t in range(start, stop):
        for i, s in enumerate(sessions):
            if s['end'] == t:
                unserved += remaining[i]
        active_ids = [i for i, s in enumerate(sessions) if s['start'] <= t < s['end'] and remaining[i] > 1e-8]
        active = [sessions[i] for i in active_ids]
        horizon = min(12, len(d['price']) - t)
        if t not in lookup:
            continue
        index = lookup[t]
        continuation = np.zeros(horizon)
        for i, s in zip(active_ids, active):
            energy = remaining[i]
            for h in range(horizon):
                power = min(s['pmax'], energy / (.92 * .25)) if t + h < s['end'] else 0.
                continuation[h] += power
                energy -= .92 * .25 * power
        # Forecast horizon zero is t+1, whereas scheduling horizon zero is the current slot t.
        future = np.r_[0., f['prediction'][index, :horizon - 1].sum(-1)]
        arrival = np.maximum(0, future - continuation)
        reserve = arrival + args.beta * np.r_[0., f['reserve'][:horizon - 1]]
        exogenous = np.repeat(d['base'][t], horizon) + reserve
        exogenous[0] = d['base'][t]
        begin = time.perf_counter()
        if args.strategy == 'uncontrolled':
            command = np.array([min(float(d['power'][t, s['node']]), s['pmax'], remaining[i] / (.92 * .25)) for i, s in zip(active_ids, active)])
        elif args.strategy in ('edf', 'tou_edf'):
            command = np.zeros(len(active))
            room = max(0, float(d['capacity']) - float(d['base'][t]))
            for j in sorted(range(len(active)), key=lambda j: active[j]['end']):
                maximum = active[j]['pmax']
                if args.strategy == 'tou_edf':
                    future = d['price'][t:min(active[j]['end'], len(d['price']))]
                    urgent = max(0., remaining[active_ids[j]] / (.92 * .25) - maximum * (active[j]['end'] - t - 1))
                    maximum = maximum if d['price'][t] <= future.min() else min(maximum, urgent)
                command[j] = min(maximum, remaining[active_ids[j]] / (.92 * .25), room)
                room -= command[j]
        else:
            command = plan(active, remaining[active_ids], horizon, d['price'][t:t + horizon], exogenous, float(d['capacity']), t)
        for i, power in zip(active_ids, command):
            remaining[i] = max(0., remaining[i] - .92 * .25 * power)
        site = float(d['base'][t]) + command.sum()
        trace.append({'slot': t, 'site_kw': float(site), 'energy_cost': float(.25 * d['price'][t] * site),
                      'transformer_violation': bool(site > float(d['capacity']) + 1e-6),
                      'solve_ms': (time.perf_counter() - begin) * 1000})
    for i, s in enumerate(sessions):
        if start <= s['end'] <= stop:
            if s['end'] == stop:
                unserved += remaining[i]
    result = {'peak_kw': max(v['site_kw'] for v in trace), 'energy_cost': sum(v['energy_cost'] for v in trace),
              'unserved_kwh': float(unserved), 'transformer_violation_rate': float(np.mean([v['transformer_violation'] for v in trace])),
              'mean_solve_ms': float(np.mean([v['solve_ms'] for v in trace]))}
    departing_requested = sum(s['energy'] for s in sessions if start <= s['end'] <= stop)
    result['unserved_percent'] = 100 * unserved / departing_requested if departing_requested > 0 else None
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'metrics.json').write_text(json.dumps(result, indent=2))
    (out / 'schedule.json').write_text(json.dumps(trace, indent=2))
    plt.plot([v['slot'] for v in trace], [v['site_kw'] for v in trace])
    plt.axhline(float(d['capacity']), linestyle='--', color='gray')
    plt.xlabel('Slot')
    plt.ylabel('Site load (kW)')
    plt.tight_layout()
    plt.savefig(out / 'schedule.png', dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--timezone', default='America/Los_Angeles')
    p.add_argument('--capacity', type=float, default=225.)
    p.add_argument('--bootstrap', type=int, default=200)
    p.add_argument('--reduction-samples', type=int, default=1000)
    p.add_argument('--seed', type=int, default=0)
    p = sub.add_parser('train')
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--variant', choices=['full', 'no_reduction', 'fixed_correlation', 'point', 'lower', 'upper', 'mean_fusion', 'uniform_loss'], default='full')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--seed', type=int, default=0)
    p = sub.add_parser('evaluate')
    for key in ['data', 'model', 'output']:
        p.add_argument('--' + key, required=True)
    p.add_argument('--baseline', choices=['model', 'persistence', 'seasonal', 'oracle'], default='model')
    p.add_argument('--split', choices=['val', 'test'], default='test')
    p.add_argument('--batch', type=int, default=16)
    p = sub.add_parser('schedule')
    for key in ['data', 'forecast', 'sessions', 'output']:
        p.add_argument('--' + key, required=True)
    p.add_argument('--beta', type=float, default=.5)
    p.add_argument('--strategy', choices=['mpc', 'uncontrolled', 'edf', 'tou_edf'], default='mpc')
    args = parser.parse_args()
    {'prepare': prepare, 'train': train, 'evaluate': evaluate, 'schedule': schedule}[args.command](args)
