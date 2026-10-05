"""F9 sender-only correspondence-conditioned coding; the inherited receiver is unchanged."""
import math

import torch
from torch import nn
import torch.nn.functional as F

from geocomm.stereo_feature_link import StereoFeatureLink


ARMS = ('U', 'G', 'P', 'S')
SHUFFLE = tuple(row * 8 + (column + 4) % 8 for row in range(4) for column in range(8))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def group_ids(height, width, device=None):
    """Public nonoverlapping pixel-center assignment, also used on the code grid."""
    require(type(height) is int and type(width) is int and height >= 4 and width >= 8,
            'every public 4x8 group must contain at least one cell')
    rows = ((2 * torch.arange(height, device=device) + 1) * 4) // (2 * height)
    columns = ((2 * torch.arange(width, device=device) + 1) * 8) // (2 * width)
    return rows[:, None] * 8 + columns[None, :]


def _slices(length, count):
    cells = [[] for _ in range(count)]
    for i in range(length):
        cells[((2 * i + 1) * count) // (2 * length)].append(i)
    require(all(cells), 'empty public group')
    return [(indices[0], indices[-1] + 1) for indices in cells]


def group_mean(feature):
    require(feature.ndim == 4 and feature.shape[0] == 1, 'group mean requires batch1 BCHW')
    h, w = feature.shape[-2:]
    group_ids(h, w, feature.device)
    return torch.stack([feature[:, :, r0:r1, c0:c1].mean((-2, -1))
                        for r0, r1 in _slices(h, 4) for c0, c1 in _slices(w, 8)], dim=1)


def _check_stereo(left, right):
    require(left.ndim == 4 and left.shape == right.shape and left.shape[:2] == (1, 32),
            'matched batch1 32-channel stereo required')
    require(left.dtype == right.dtype == torch.float32 and left.device == right.device,
            'matched FP32 stereo required')
    h, w = left.shape[-2:]
    require(h % 4 == w % 4 == 0 and h >= 16 and w >= 32, 'nonoverlapping 4x4 stereo cells')
    require(torch.isfinite(left).all() and torch.isfinite(right).all(), 'finite stereo features')


def _aggregate(probability, sign):
    """Sum dense candidate probability, then average actual reference cells per group."""
    _, disparities, h, w = probability.shape
    ids = group_ids(h, w, probability.device)
    counts = torch.bincount(ids.flatten(), minlength=32).to(probability.dtype)
    columns = torch.arange(w, device=probability.device)[None, None, :]
    shifts = torch.arange(disparities, device=probability.device)[:, None, None]
    target_x = (columns + sign * shifts).expand(-1, h, -1).clamp(0, w - 1)
    rows = torch.arange(h, device=probability.device)[None, :, None].expand_as(target_x)
    target = ids[rows, target_x]
    address = (ids[None] * 32 + target).reshape(1, -1)
    matrix = probability.new_zeros((1, 32 * 32)).scatter_add(1, address, probability.reshape(1, -1))
    return matrix.reshape(1, 32, 32) / counts[None, :, None]


def correspondence(left, right):
    """Differentiable LR/RL matrices for pooled shifts 0..48, including flat features."""
    _check_stereo(left, right)
    left = F.avg_pool2d(left, 4, 4)
    right = F.avg_pool2d(right, 4, 4)
    left = left / torch.linalg.vector_norm(left, dim=1, keepdim=True).clamp_min(1e-6)
    right = right / torch.linalg.vector_norm(right, dim=1, keepdim=True).clamp_min(1e-6)
    width = left.shape[-1]
    lr, rl = [], []
    for shift in range(49):
        if shift == 0:
            score = (left * right).sum(1)
            lr.append(score); rl.append(score)
        elif shift < width:
            score = (left[:, :, :, shift:] * right[:, :, :, :-shift]).sum(1)
            lr.append(F.pad(score, (shift, 0), value=-torch.inf))
            rl.append(F.pad(score, (0, shift), value=-torch.inf))
        else:
            invalid = left.new_full((1, left.shape[-2], width), -torch.inf)
            lr.append(invalid); rl.append(invalid)
    lr = _aggregate(torch.stack(lr, 1).div(.1).softmax(1), -1)
    rl = _aggregate(torch.stack(rl, 1).div(.1).softmax(1), 1)
    for matrix in (lr, rl):
        require(torch.isfinite(matrix).all() and (matrix >= 0).all(), 'finite nonnegative correspondence')
        require(torch.allclose(matrix.sum(-1), torch.ones_like(matrix[:, :, 0]), rtol=1e-5, atol=5e-5),
                'FP32 row-stochastic correspondence')
    return lr, rl


def shuffle_columns(matrix):
    require(matrix.shape == (1, 32, 32), 'public correspondence matrix shape')
    return matrix[:, :, list(SHUFFLE)]


def new_gain_head():
    # torch.manual_seed also seeds CUDA. Use only the CPU default generator here.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(1930)
        head = nn.Sequential(nn.Linear(33, 16), nn.GELU(), nn.Linear(16, 1))
        nn.init.zeros_(head[2].weight)
        nn.init.zeros_(head[2].bias)
    require(len(tuple(head.parameters())) == 4 and sum(p.numel() for p in head.parameters()) == 561,
            'locked head parameter count')
    return head


def _matrix_record(matrix):
    vertical = torch.arange(32, device=matrix.device) // 8
    cross = vertical[:, None] != vertical[None, :]
    value = matrix.detach()
    return {'row_sum_max_error': float((value.sum(-1) - 1).abs().max()),
            'cross_vertical_mass': float(value[0][cross].abs().sum())}


def _energy_record(symbols, layouts):
    power = symbols.detach().double().square().sum(-1)
    counts = [c * h * w // 2 for _, c, h, w in layouts]
    streams = power.split(counts, 1)
    records = []
    for stream, (_, channels, h, w) in zip(streams[:2], layouts[:2]):
        require(channels == 2, 'one complex pair per stereo grid cell')
        grid = stream.reshape(1, h, w)
        records.append([float(grid[:, r0:r1, c0:c1].sum())
                        for r0, r1 in _slices(h, 4) for c0, c1 in _slices(w, 8)])
    return {'stereo_group_energy': records[0] + records[1],
            'appearance_energy': float(streams[2].sum()),
            'actual_energy_float64': float(power.sum()),
            'group_energy_layout': 'left32_then_right32_pixel_center_groups'}


class CorrespondenceLink(StereoFeatureLink):
    """Installed after strict parent loading. decode() is exactly the inherited method."""

    def forward(self, left_stereo, right_stereo, appearance, batch):
        _check_stereo(left_stereo, right_stereo)
        require(self.arm in ARMS and self.channel.kind in ('identity', 'awgn'), 'locked F9 arm/channel')
        require(math.isfinite(float(self.snr_db)), 'public finite nominal SNR')
        require(appearance.dtype == left_stereo.dtype and appearance.device == left_stereo.device
                and torch.isfinite(appearance).all(), 'finite matching appearance')
        public_shapes = tuple(tuple(feature.shape) for feature in (left_stereo, right_stereo, appearance))
        expected = self.layout(public_shapes)
        group_ids(expected[0][2], expected[0][3], left_stereo.device)
        codes = (self.stereo_encoder(left_stereo), self.stereo_encoder(right_stereo),
                 self.appearance_encoder(appearance))
        require([tuple(code.shape) for code in codes] == expected, 'encoded fixed layout differs')
        matrices = None
        if self.arm == 'U':
            gains = left_stereo.new_ones((1, 2, 32))
            changed = codes
        else:
            q = []
            for feature in (left_stereo, right_stereo):
                means = group_mean(feature)
                snr = means.new_full((1, 32, 1), float(self.snr_db) / 20.)
                q.append(self.gain_head(torch.cat((means, snr), -1)).squeeze(-1))
            if self.arm in ('P', 'S'):
                lr, rl = correspondence(left_stereo, right_stereo)
                if self.arm == 'S':
                    lr, rl = shuffle_columns(lr), shuffle_columns(rl)
                matrices = [_matrix_record(lr), _matrix_record(rl)]
                q = [(q[0] + torch.bmm(lr, q[1].unsqueeze(-1)).squeeze(-1)) / 2.,
                     (q[1] + torch.bmm(rl, q[0].unsqueeze(-1)).squeeze(-1)) / 2.]
            gains = torch.exp(math.log(2.) * torch.tanh(torch.stack(q, 1)))
            require(torch.isfinite(gains).all() and (gains >= .5).all() and (gains <= 2.).all(), 'bounded gains')
            changed = tuple(code * gains[:, view, group_ids(code.shape[2], code.shape[3], code.device)].unsqueeze(1)
                            for view, code in enumerate(codes[:2])) + (codes[2],)
        symbols = torch.cat([self.pack(code) for code in changed], dim=1)
        energy = symbols.square().sum(-1).mean(1, keepdim=True)
        require(torch.isfinite(energy).all() and (energy > 0).all(), 'valid actual transmitted code energy')
        symbols = symbols / energy.sqrt().unsqueeze(-1)
        energy_record = _energy_record(symbols, expected)
        received, account = self.channel(symbols, self.snr_db)
        # No source-dependent arguments are passed across this boundary.
        outputs = self.decode(received, public_shapes)
        pixels = batch['left_img'].shape[-2] * batch['left_img'].shape[-1]
        account.update(allocation='uniform' if self.arm == 'U' else 'F9_' + self.arm,
                       cbr_complex_per_input_real_scalar=account['total_complex_uses'] / (6 * pixels),
                       boundary='stereo_features_before_receiver_cost',
                       stereo_complex_uses=expected[0][1] * expected[0][2] * expected[0][3],
                       appearance_complex_uses=expected[2][1] * expected[2][2] * expected[2][3] // 2)
        self.last_accounting = account
        batch['communication_accounting'] = account
        self.last_f9 = dict(arm=self.arm, nominal_snr=float(self.snr_db), matrix_records=matrices,
                           amplitude_gains=gains.detach().cpu().flatten().tolist(),
                           data_complex_uses=symbols.shape[1], head_parameters=561,
                           receiver_side_information=False, **energy_record)
        return outputs


def install(link, arm):
    """Keep every loaded parent tensor; add four isolated-initialized tensors in place."""
    require(type(link) is StereoFeatureLink and arm in ARMS, 'fresh original link and U/G/P/S arm required')
    require(not hasattr(link, 'gain_head'), 'head already registered')
    parent = {name: value.detach().clone() for name, value in link.state_dict().items()}
    first = next(link.parameters())
    head = new_gain_head().to(device=first.device, dtype=first.dtype)
    head.requires_grad_(arm != 'U')
    head.train(link.training)
    link.__class__ = CorrespondenceLink
    link.add_module('gain_head', head)
    link.arm = arm
    link.last_f9 = None
    current = link.state_dict()
    require(set(current) - set(parent) == {'gain_head.0.weight', 'gain_head.0.bias', 'gain_head.2.weight', 'gain_head.2.bias'},
            'only four head states may be added')
    require(all(torch.equal(value, current[name]) for name, value in parent.items()), 'all parent link states unchanged')
    return link
