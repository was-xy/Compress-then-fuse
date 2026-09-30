"""Lightweight Temporal Attention Encoder (L-TAE).

Reference: Garnot and Landrieu, "Lightweight Temporal Self-Attention for
Classifying Satellite Image Time Series", 2020.  A single set of learned
queries is shared across all pixels; the encoder maps a sequence of feature
maps to one feature map and returns the per-head attention masks, which the
decoder reuses to collapse the skip connections.
"""

import numpy as np
import torch
import torch.nn as nn

from models.blocks import PositionalEncoder


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature: float):
        super().__init__()
        self.temperature = temperature

    def forward(self, q, k, v, pad_mask=None):
        scores = torch.matmul(q.unsqueeze(1), k.transpose(1, 2)) / self.temperature
        if pad_mask is not None:
            scores = scores.masked_fill(
                pad_mask.unsqueeze(1), torch.finfo(scores.dtype).min
            )
        attn = torch.softmax(scores, dim=2)
        return torch.matmul(attn, v), attn


class MultiHeadAttention(nn.Module):
    """Master-query attention: the queries are parameters, not projections."""

    def __init__(self, n_head: int, d_k: int, d_in: int):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in

        self.query = nn.Parameter(torch.zeros(n_head, d_k))
        nn.init.normal_(self.query, mean=0, std=np.sqrt(2.0 / d_k))
        self.to_key = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.to_key.weight, mean=0, std=np.sqrt(2.0 / d_k))
        self.attention = ScaledDotProductAttention(temperature=np.sqrt(d_k))

    def forward(self, v, pad_mask=None):
        n_batch, seq_len, _ = v.shape
        n_head, d_k = self.n_head, self.d_k

        q = self.query.repeat(n_batch, 1, 1).transpose(0, 1).reshape(-1, d_k)
        k = self.to_key(v).view(n_batch, seq_len, n_head, d_k)
        k = k.permute(2, 0, 1, 3).reshape(-1, seq_len, d_k)
        v = torch.stack(v.split(self.d_in // n_head, dim=-1)).view(
            n_head * n_batch, seq_len, -1
        )
        if pad_mask is not None:
            pad_mask = pad_mask.repeat(n_head, 1)

        out, attn = self.attention(q, k, v, pad_mask=pad_mask)
        attn = attn.view(n_head, n_batch, seq_len)
        out = out.view(n_head, n_batch, self.d_in // n_head)
        return out, attn


class LTAE2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_head: int = 16,
        d_k: int = 4,
        d_model: int = 256,
        mlp=(256, 128),
        period: int = 1000,
    ):
        super().__init__()
        assert mlp[0] == d_model
        self.n_head = n_head
        self.in_norm = nn.GroupNorm(num_groups=n_head, num_channels=in_channels)
        self.in_conv = nn.Conv1d(in_channels, d_model, kernel_size=1)
        self.positional_encoder = PositionalEncoder(
            d_model // n_head, period=period, repeat=n_head
        )
        self.attention = MultiHeadAttention(n_head=n_head, d_k=d_k, d_in=d_model)
        self.mlp = nn.Sequential(
            *[
                layer
                for i in range(len(mlp) - 1)
                for layer in (
                    nn.Linear(mlp[i], mlp[i + 1]),
                    nn.BatchNorm1d(mlp[i + 1]),
                    nn.ReLU(),
                )
            ]
        )
        self.out_norm = nn.GroupNorm(num_groups=n_head, num_channels=mlp[-1])

    def forward(self, x, positions, pad_mask=None):
        """(B, T, C, H, W) -> ((B, C_out, H, W), attention masks)."""
        b, t, c, h, w = x.shape

        out = x.permute(0, 3, 4, 1, 2).reshape(b * h * w, t, c)
        out = self.in_norm(out.transpose(1, 2)).transpose(1, 2)
        out = self.in_conv(out.transpose(1, 2)).transpose(1, 2)

        pixel_positions = positions[:, None, None, :].expand(b, h, w, t)
        out = out + self.positional_encoder(pixel_positions.reshape(b * h * w, t))

        if pad_mask is not None:
            pad_mask = pad_mask[:, None, None, :].expand(b, h, w, t)
            pad_mask = pad_mask.reshape(b * h * w, t)

        out, attn = self.attention(out, pad_mask=pad_mask)
        out = out.permute(1, 0, 2).reshape(b * h * w, -1)
        out = self.out_norm(self.mlp(out))
        out = out.view(b, h, w, -1).permute(0, 3, 1, 2)

        attn = attn.view(self.n_head, b, h, w, t).permute(0, 1, 4, 2, 3)
        return out, attn
