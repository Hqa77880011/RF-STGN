"""Published forecasting families on the common 96-slot/12-step station protocol."""
import argparse
import importlib.util
import itertools
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

UPSTREAM = {
    'tsl': ('https://github.com/thuml/Time-Series-Library.git', '4e938a1767106324dd753b2a44832bf870a0252e'),
    'gwnet': ('https://github.com/nnzhan/Graph-WaveNet.git', '6b162e80c59a1d494809252eca055cff93dc66b1'),
}
ARCHITECTURES = ['rf_stgn', 'gru', 'dcrnn', 'graph_wavenet', 'astgrn_adapted', 'timesnet', 'itransformer', 'xgboost']


def fetch(root):
    """Fetch pinned author sources into ignored local data; preserve their licenses."""
    for name, (url, revision) in UPSTREAM.items():
        target = Path(root) / ('TSL' if name == 'tsl' else 'Graph-WaveNet')
        if not (target / '.git').exists():
            subprocess.run(['git', 'clone', '--filter=blob:none', '--no-checkout', url, str(target)], check=True)
        subprocess.run(['git', '-C', str(target), 'fetch', '--depth', '1', 'origin', revision], check=True)
        if name == 'tsl':
            subprocess.run(['git', '-C', str(target), 'sparse-checkout', 'set', 'models', 'layers', 'utils'], check=True)
        subprocess.run(['git', '-C', str(target), 'checkout', '--detach', revision], check=True)


def module(path, name, root):
    if not path.exists():
        raise FileNotFoundError('Fetch the pinned author source first: ' + str(path))
    sys.path.insert(0, str(root.resolve()))
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


class DiffusionCell(nn.Module):
    """Forward/backward random-walk diffusion gates for one graph-GRU step."""
    def __init__(self, inputs, width, order=2):
        super().__init__()
        self.width, self.order = width, order
        self.gates = nn.Linear((inputs + width) * (1 + 2 * order), width * 2)
        self.candidate = nn.Linear((inputs + width) * (1 + 2 * order), width)

    def diffuse(self, value, graph):
        terms = [value]
        for support in [graph, graph.T / graph.T.sum(-1, keepdim=True).clamp_min(1e-8)]:
            current = value
            for _ in range(self.order):
                current = torch.matmul(support, current)
                terms.append(current)
        return torch.cat(terms, -1)

    def forward(self, value, hidden, graph):
        reset, update = self.gates(self.diffuse(torch.cat([value, hidden], -1), graph)).sigmoid().chunk(2, -1)
        candidate = self.candidate(self.diffuse(torch.cat([value, reset * hidden], -1), graph)).tanh()
        return update * hidden + (1 - update) * candidate


class GraphRecurrent(nn.Module):
    def __init__(self, features, graph, width, adaptive=False):
        super().__init__()
        self.register_buffer('graph', torch.as_tensor(graph, dtype=torch.float32))
        self.adaptive = adaptive
        self.encoder = DiffusionCell(features, width)
        self.decoder = DiffusionCell(1, width)
        self.head = nn.Linear(width, 1)
        self.register_buffer('batches_seen', torch.zeros((), dtype=torch.long))
        if adaptive:
            self.embedding = nn.Parameter(torch.randn(len(graph), 8) * .1)
            self.project = nn.Linear(8, width)
            self.query, self.key = nn.Linear(width, width, bias=False), nn.Linear(width, width, bias=False)
            self.key.weight.data.copy_(self.query.weight.data)

    def forward(self, x, *unused, target=None):
        graph = self.graph
        if self.adaptive:
            embedded = self.project(self.embedding).tanh()
            graph = (self.query(embedded) @ self.key(embedded).T).relu().softmax(-1)
        hidden = x.new_zeros(x.shape[0], x.shape[2], self.head.in_features)
        for tick in range(x.shape[1]):
            hidden = self.encoder(x[:, tick], hidden, graph)
        previous = x.new_zeros(x.shape[0], x.shape[2], 1)
        forecasts = []
        ratio = 1000 / (1000 + np.exp(min(50., float(self.batches_seen) / 1000)))
        for horizon in range(12):
            hidden = self.decoder(previous, hidden, graph)
            previous = self.head(hidden)
            forecasts.append(previous[..., 0])
            if self.training and target is not None and torch.rand(()) < ratio:
                previous = target[:, horizon, :, None]
        if self.training:
            self.batches_seen += 1
        return torch.stack(forecasts, 1)


class TemporalBaseline(nn.Module):
    def __init__(self, features, nodes, width, architecture, upstream):
        super().__init__()
        self.architecture, self.nodes = architecture, nodes
        channels = features * nodes
        if architecture == 'gru':
            self.encoder = nn.GRU(channels, width, batch_first=True)
            self.head = nn.Linear(width, 12 * nodes)
        else:
            name = 'TimesNet' if architecture == 'timesnet' else 'iTransformer'
            root = Path(upstream) / 'TSL'
            source = module(root / 'models' / (name + '.py'), 'rf_author_' + name, root)
            config = SimpleNamespace(task_name='long_term_forecast', seq_len=96, label_len=48, pred_len=12,
                                     enc_in=channels, dec_in=channels, c_out=channels, d_model=width,
                                     n_heads=4, d_ff=width * 2, e_layers=2, d_layers=1, dropout=0.,
                                     embed='timeF', freq='t', factor=1, activation='gelu', output_attention=False,
                                     top_k=3, num_kernels=3, use_norm=True, channel_independence=0)
            self.encoder = source.Model(config)
            self.head = nn.Linear(channels, nodes)

    def forward(self, x, *unused, target=None):
        sequence = x.flatten(2)
        if self.architecture == 'gru':
            _, hidden = self.encoder(sequence)
            return self.head(hidden[-1]).reshape(len(x), 12, self.nodes)
        return self.head(self.encoder(sequence, None, None, None)[:, -12:])


class GraphWaveNet(nn.Module):
    def __init__(self, features, graph, width, upstream):
        super().__init__()
        root = Path(upstream) / 'Graph-WaveNet'
        source = module(root / 'model.py', 'rf_author_gwnet', root)
        self.encoder = source.gwnet(torch.device('cpu'), len(graph), dropout=0.,
                                    supports=[torch.tensor(graph.T.copy(), dtype=torch.float32)],
                                    gcn_bool=True, addaptadj=True, in_dim=features, out_dim=12,
                                    residual_channels=width, dilation_channels=width,
                                    skip_channels=width * 2, end_channels=width * 4, blocks=2, layers=6)
        # The original source uses Conv1d with two-dimensional kernels in these
        # paths. Replace only those modules locally for current PyTorch's 4-D input.
        for name in ['gate_convs', 'residual_convs', 'skip_convs']:
            layers = getattr(self.encoder, name)
            for index, layer in enumerate(layers):
                if isinstance(layer, nn.Conv1d) and layer.weight.ndim == 4:
                    replacement = nn.Conv2d(layer.in_channels, layer.out_channels, layer.kernel_size,
                                            stride=layer.stride, padding=layer.padding, dilation=layer.dilation,
                                            groups=layer.groups, bias=layer.bias is not None)
                    replacement.load_state_dict(layer.state_dict())
                    layers[index] = replacement

    def forward(self, x, *unused, target=None):
        return self.encoder(x.permute(0, 3, 2, 1))[..., -1]


class TreeForecast:
    def __init__(self, model):
        self.model, self.variant = model, 'full'

    def eval(self):
        return self

    def parameters(self):
        return iter(())

    def __call__(self, x, *unused):
        output = self.model.predict(x.numpy().reshape(len(x), -1))
        return torch.tensor(output.reshape(len(x), 12, x.shape[2]), dtype=torch.float32)


def build(features, variant, architecture, graph, nodes, width=32, upstream='data/upstream'):
    if architecture == 'rf_stgn':
        from .cli import Forecast
        model = Forecast(features, variant, width)
    elif architecture in ['dcrnn', 'astgrn_adapted']:
        model = GraphRecurrent(features, graph, width, architecture == 'astgrn_adapted')
    elif architecture == 'graph_wavenet':
        model = GraphWaveNet(features, graph, width, upstream)
    else:
        model = TemporalBaseline(features, nodes, width, architecture, upstream)
    model.variant = variant
    return model



def fit_tree(args, data):
    """Fit horizon-wise XGBoost only on registered training origins."""
    from sklearn.multioutput import MultiOutputRegressor
    from xgboost import XGBRegressor
    from .cli import tensors
    selected = np.arange(data['x'].shape[-1])
    origin = data['origins']
    rows = [np.flatnonzero(data['split'][origin] == role) for role in ['train', 'val']]
    inputs = [tensors(data, ids, selected)[0].numpy().reshape(len(ids), -1) for ids in rows]
    targets = [np.stack([data['power'][t + 1:t + 13] for t in origin[ids]]) for ids in rows]
    model = MultiOutputRegressor(XGBRegressor(n_estimators=getattr(args, 'trees', 200),
                                              max_depth=getattr(args, 'tree_depth', 6), learning_rate=.05,
                                              n_jobs=1, random_state=args.seed, tree_method='hist'), n_jobs=1)
    normalized = (targets[0] - data['target_mean']) / data['target_std']
    model.fit(inputs[0], normalized.reshape(len(normalized), -1))
    prediction = model.predict(inputs[1]).reshape(targets[1].shape) * data['target_std'] + data['target_mean']
    residual = np.abs(targets[1].sum(-1) - prediction.sum(-1))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(dict(tree=model, selected=selected, architecture='xgboost', variant='full',
                    reserve=np.quantile(residual, .9, axis=0, method='higher')), out / 'model.pt')
    (out / 'history.json').write_text(json.dumps([dict(epoch=1, val_rmse=float(np.sqrt(np.mean((prediction - targets[1]) ** 2))))], indent=2))


def grid(args):
    from .cli import train
    root = Path(args.output)
    trials = []
    for index, values in enumerate(itertools.product(args.values, repeat=3)):
        if args.max_trials is not None and index >= args.max_trials:
            break
        output = root / f'trial{index}'
        train(SimpleNamespace(data=args.data, output=str(output), architecture='rf_stgn', variant='full',
                              width=args.width, upstream=args.upstream, epochs=args.epochs, batch=args.batch,
                              seed=args.seed, load_weight=values[0], price_weight=values[1], occupancy_weight=values[2]))
        history = json.loads((output / 'history.json').read_text())
        trials.append(dict(trial=index, weights=values, validation_rmse=min(v['val_rmse'] for v in history)))
    best = min(trials, key=lambda row: row['validation_rmse'])
    root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(root / f"trial{best['trial']}" / 'model.pt', root / 'model.pt')
    (root / 'selection.json').write_text(json.dumps(dict(selected=best, trials=trials), indent=2))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('fetch')
    p.add_argument('--upstream', default='data/upstream')
    p = sub.add_parser('grid')
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--values', type=float, nargs='+', default=[0., .5, 1.])
    p.add_argument('--max-trials', type=int)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--width', type=int, default=32)
    p.add_argument('--upstream', default='data/upstream')
    args = parser.parse_args()
    fetch(args.upstream) if args.command == 'fetch' else grid(args)


if __name__ == '__main__':
    main()
