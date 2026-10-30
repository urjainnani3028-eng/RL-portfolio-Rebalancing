"""The five bandit arms. Every model maps a window (batch, L, F) to one number: the predicted
standardised next-bar log return. For a fair comparison they share the project plan's budget:
2 hidden layers, 64 units, dropout 0.1.

    rnn       Elman RNN (tanh)                                   - torch.nn.RNN
    lstm      Long Short-Term Memory (Hochreiter & Schmidhuber 1997) - torch.nn.LSTM
    gru       Gated Recurrent Unit (Cho et al. 2014)                - torch.nn.GRU
    tcn       Temporal Convolutional Network, Bai, Kolter & Koltun (2018),
              open source: github.com/locuslab/TCN
    patchtst  PatchTST, Nie et al. (ICLR 2023), "A Time Series is Worth 64 Words",
              open source: github.com/yuqinie98/PatchTST

TCN and PatchTST are compact re-implementations that follow the reference code (causal dilated
residual blocks with weight norm; channel-independent patching + Transformer encoder + flatten
head) so the project needs no extra dependency beyond PyTorch.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm

MODEL_INFO = {
    "rnn": {"label": "RNN", "family": "Recurrent", "source": "torch.nn.RNN (Elman, 1990)"},
    "lstm": {"label": "LSTM", "family": "Recurrent (gated)", "source": "Hochreiter & Schmidhuber (1997)"},
    "gru": {"label": "GRU", "family": "Recurrent (gated)", "source": "Cho et al. (2014)"},
    "tcn": {"label": "TCN", "family": "Convolutional (open source)",
            "source": "Bai, Kolter & Koltun (2018), github.com/locuslab/TCN"},
    "patchtst": {"label": "PatchTST", "family": "Transformer (open source)",
                 "source": "Nie et al. (ICLR 2023), github.com/yuqinie98/PatchTST"},
}
MODEL_IDS = list(MODEL_INFO)


class _Recurrent(nn.Module):
    def __init__(self, cell: type[nn.Module], n_features: int, hidden: int, layers: int, dropout: float, **kw):
        super().__init__()
        self.rnn = cell(n_features, hidden, num_layers=layers, batch_first=True,
                        dropout=dropout if layers > 1 else 0.0, **kw)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(self.drop(out[:, -1])).squeeze(-1)


class RNNNet(_Recurrent):
    def __init__(self, n_features, hidden=64, layers=2, dropout=0.1, **_):
        super().__init__(nn.RNN, n_features, hidden, layers, dropout, nonlinearity="tanh")


class LSTMNet(_Recurrent):
    def __init__(self, n_features, hidden=64, layers=2, dropout=0.1, **_):
        super().__init__(nn.LSTM, n_features, hidden, layers, dropout)


class GRUNet(_Recurrent):
    def __init__(self, n_features, hidden=64, layers=2, dropout=0.1, **_):
        super().__init__(nn.GRU, n_features, hidden, layers, dropout)


# ---------------------------------------------------------------------------
# TCN (locuslab/TCN): stacks of TemporalBlocks = 2 x [causal dilated conv -> ReLU -> dropout]
# with a residual 1x1 connection. Kernel 5 and dilations 1, 2 give a receptive field of 25 bars,
# which covers the 20-bar window.
# ---------------------------------------------------------------------------
class _Chomp(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x):
        return x[:, :, :-self.n] if self.n else x


class TemporalBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel, dilation, dropout):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.net = nn.Sequential(
            weight_norm(nn.Conv1d(c_in, c_out, kernel, padding=pad, dilation=dilation)), _Chomp(pad), nn.ReLU(),
            nn.Dropout(dropout),
            weight_norm(nn.Conv1d(c_out, c_out, kernel, padding=pad, dilation=dilation)), _Chomp(pad), nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.down = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else None
        self.relu = nn.ReLU()

    def forward(self, x):
        res = x if self.down is None else self.down(x)
        return self.relu(self.net(x) + res)


class TCNNet(nn.Module):
    def __init__(self, n_features, hidden=64, layers=2, dropout=0.1, kernel=5, **_):
        super().__init__()
        blocks, c = [], n_features
        for i in range(layers):
            blocks.append(TemporalBlock(c, hidden, kernel, 2 ** i, dropout))
            c = hidden
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):                        # x: B x L x F
        h = self.tcn(x.transpose(1, 2))          # B x C x L
        return self.head(h[:, :, -1]).squeeze(-1)


# ---------------------------------------------------------------------------
# PatchTST (yuqinie98/PatchTST): each input channel is cut into overlapping patches, patches are
# linearly embedded with a learnable position code, a shared Transformer encoder runs on every
# channel independently, and a flatten head maps all patch tokens to the forecast.
# ---------------------------------------------------------------------------
class PatchTSTNet(nn.Module):
    def __init__(self, n_features, hidden=64, layers=2, dropout=0.1, lookback=20, patch_len=4, stride=2,
                 n_heads=4, **_):
        super().__init__()
        self.patch_len, self.stride = patch_len, stride
        n_patches = (lookback - patch_len) // stride + 1
        self.embed = nn.Linear(patch_len, hidden)
        self.pos = nn.Parameter(torch.zeros(1, n_patches, hidden))
        nn.init.uniform_(self.pos, -0.02, 0.02)
        layer = nn.TransformerEncoderLayer(hidden, n_heads, dim_feedforward=2 * hidden, dropout=dropout,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(n_features * n_patches * hidden, 1)

    def forward(self, x):                        # x: B x L x F
        B, L, F = x.shape
        p = x.transpose(1, 2).unfold(-1, self.patch_len, self.stride)    # B x F x P x patch_len
        P = p.shape[2]
        tok = self.embed(p).reshape(B * F, P, -1) + self.pos             # channel independence
        h = self.encoder(self.drop(tok)).reshape(B, -1)
        return self.head(self.drop(h)).squeeze(-1)


_CLASSES = {"rnn": RNNNet, "lstm": LSTMNet, "gru": GRUNet, "tcn": TCNNet, "patchtst": PatchTSTNet}


def build_model(model_id: str, n_features: int, lookback: int, hidden: int = 64, layers: int = 2,
                dropout: float = 0.1) -> nn.Module:
    if model_id not in _CLASSES:
        raise ValueError(f"Unknown model '{model_id}'. Choose from {MODEL_IDS}")
    return _CLASSES[model_id](n_features, hidden=hidden, layers=layers, dropout=dropout, lookback=lookback)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
