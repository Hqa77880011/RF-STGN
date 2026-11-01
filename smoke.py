"""Synthetic checks for forecasting and scheduling."""
import argparse
import datetime as dt
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from rf_stgn import cli as c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baselines', action='store_true')
    parser.add_argument('--upstream', default='data/upstream')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        ticks, nodes = 330, 3
        power = rng.uniform(.1, 2., (ticks, nodes)).astype(np.float32)
        split = np.repeat(['train', 'val', 'test'], 110)
        stamps = [(dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(minutes=15 * t)).isoformat() for t in range(ticks)]
        np.savez(root / 'raw.npz', power=power, time=stamps, connected=power > .5,
                 remaining_energy=np.ones_like(power), weather=rng.normal(size=(ticks, 3)),
                 cluster=np.array([0, 0, 1]), price=np.linspace(.1, .4, ticks),
                 base=np.ones(ticks), split=split)
        c.prepare(SimpleNamespace(source=str(root / 'raw.npz'), output=str(root / 'site.npz'),
                                  timezone='UTC', capacity=225., seed=0, reduction_samples=9, bootstrap=2))
        c.train(SimpleNamespace(data=str(root / 'site.npz'), output=str(root / 'fit'),
                                variant='full', seed=0, epochs=1, batch=3))
        d = np.load(root / 'site.npz')
        selected = d['selected']
        model = c.Forecast(len(selected))
        features = c.tensors(d, np.array([0, 1]), selected)
        opt = torch.optim.Adam(model.parameters(), lr=.001)
        old = model.relation.weight.detach().clone()
        loss = model(*features).square().mean()
        loss.backward()
        assert torch.isfinite(loss) and model.relation.weight.grad.abs().sum() > 0
        assert model.temporal.weight_ih_l0.grad.abs().sum() > 0
        opt.step()
        assert not torch.equal(old, model.relation.weight)
        first, state = c.load_model(root / 'fit/model.pt')
        second, _ = c.load_model(root / 'fit/model.pt')
        first.eval(); second.eval()
        torch.testing.assert_close(first(*features), second(*features), rtol=0, atol=0)
        if args.baselines:
            from rf_stgn import baselines as b
            all_features = np.arange(d['x'].shape[-1])
            common = c.tensors(d, np.array([0, 1]), all_features)
            graph = (d['fixed'] + d['arrival'] + d['cluster']) / 3 + np.eye(nodes)
            graph /= graph.sum(-1, keepdims=True)
            for architecture in ['gru', 'dcrnn', 'graph_wavenet', 'astgrn_adapted', 'timesnet', 'itransformer']:
                torch.manual_seed(0)
                control = b.build(len(all_features), 'full', architecture, graph, nodes, 8, args.upstream)
                opt = torch.optim.Adam(control.parameters(), lr=.001)
                before = {name: value.detach().clone() for name, value in control.named_parameters()}
                forecast = control(*common)
                objective = forecast.square().mean()
                objective.backward()
                assert torch.isfinite(objective)
                assert any(value.grad is not None and value.grad.abs().sum() > 0 for name, value in control.named_parameters() if 'encoder' in name or 'embedding' in name)
                if architecture == 'astgrn_adapted':
                    assert control.embedding.grad.abs().sum() > 0
                if architecture == 'graph_wavenet':
                    assert control.encoder.nodevec1.grad.abs().sum() > 0
                opt.step()
                assert any(not torch.equal(before[name], value) for name, value in control.named_parameters())
                c.train(SimpleNamespace(data=str(root / 'site.npz'), output=str(root / architecture), architecture=architecture,
                                        variant='full', seed=0, epochs=1, batch=3, width=8, upstream=args.upstream))
                a, state = c.load_model(root / architecture / 'model.pt')
                other, _ = c.load_model(root / architecture / 'model.pt')
                a.eval(); other.eval()
                torch.testing.assert_close(a(*common), other(*common), atol=0, rtol=0)
                assert len(state['selected']) == 28
                c.evaluate(SimpleNamespace(data=str(root / 'site.npz'), model=str(root / architecture / 'model.pt'),
                                           output=str(root / (architecture + '_eval')), split='test', baseline='model', batch=3))
            c.train(SimpleNamespace(data=str(root / 'site.npz'), output=str(root / 'xgboost'), architecture='xgboost',
                                    variant='full', seed=0, epochs=1, batch=3, trees=2, tree_depth=2))
            c.evaluate(SimpleNamespace(data=str(root / 'site.npz'), model=str(root / 'xgboost/model.pt'),
                                       output=str(root / 'xgboost_eval'), split='test', baseline='model', batch=3))
            b.grid(SimpleNamespace(data=str(root / 'site.npz'), output=str(root / 'grid'), values=[0., 1.], max_trials=2,
                                   epochs=1, batch=3, seed=0, width=8, upstream=args.upstream))
            selection = json.loads((root / 'grid/selection.json').read_text())
            assert selection['selected']['validation_rmse'] == min(row['validation_rmse'] for row in selection['trials'])
            chosen = torch.load(root / 'grid/model.pt', weights_only=False)
            assert chosen['loss_weights'] == selection['selected']['weights']
            print('PASS: published-source temporal/GNN baselines, adaptive graph gradients, XGBoost and validation-only weight grid')
        d.close()
        for baseline in ['model', 'persistence', 'oracle']:
            c.evaluate(SimpleNamespace(data=str(root / 'site.npz'), model=str(root / 'fit/model.pt'),
                                       output=str(root / baseline), split='test', baseline=baseline, batch=3))
        metrics = json.loads((root / 'oracle/metrics.json').read_text())
        assert metrics['1']['mae'] == 0 and metrics['12']['rmse'] == 0
        with np.load(root / 'model/forecast.npz') as f:
            origins = f['origin'].copy()
        start = int(origins.min())
        sessions = [{'node': 0, 'start': start, 'end': start + 3, 'energy': .3, 'pmax': 1.}]
        (root / 'sessions.json').write_text(json.dumps(sessions))
        c.schedule(SimpleNamespace(data=str(root / 'site.npz'), forecast=str(root / 'model/forecast.npz'),
                                   sessions=str(root / 'sessions.json'), output=str(root / 'schedule'), beta=.5, strategy='mpc'))
        result = json.loads((root / 'schedule/metrics.json').read_text())
        assert result['transformer_violation_rate'] == 0 and result['unserved_kwh'] < 1e-6
        # Scheduling horizon 1 must receive forecast index 0 (t+1), not index 1.
        np.savez(root / 'known.npz', origin=np.array([start]),
                 prediction=np.full((1, 12, nodes), 7.), reserve=np.arange(1, 13, dtype=float))
        captured = []
        def capture(active, remaining, horizon, price, exogenous, capacity, now):
            captured.append(exogenous.copy())
            return np.zeros(len(active))
        with patch.object(c, 'plan', capture):
            c.schedule(SimpleNamespace(data=str(root / 'site.npz'), forecast=str(root / 'known.npz'),
                                       sessions=str(root / 'sessions.json'), output=str(root / 'alignment'), beta=.5, strategy='mpc'))
        # At t+1, the session has .3-.23 kWh left, hence ( .3-.23 )/.23 kW continuation.
        expected = 1 + 21 - (.3 - .92 * .25) / (.92 * .25) + .5
        np.testing.assert_allclose(captured[0][1], expected, atol=1e-12)
    print('PASS: timestamped NPZ preparation, train-only scaling/reduction, graph/GRU gradient update, reload equality, metrics/baselines and real QP scheduling/horizon alignment')


if __name__ == '__main__':
    main()
