"""Temporal cross-attention fusion baselines.

Two variants share one implementation:

``s2_center``
    The optical series queries every SAR series and the enriched copies are
    concatenated with the original optical series.  Queries and keys live on
    their own temporal grid, so the modalities need not be aligned.

``full``
    Every modality queries the concatenation of the others.  All modalities must
    expose the same number of steps, which in practice means the series have been
    compressed into a common set of slots.

Both variants are time aware: queries and keys carry the sinusoidal encoding of
their positions, which are acquisition days when the fusion runs on raw series
and slot indices when it runs on compressed ones.  Keys coming from a padded
step are masked out.

Attention is applied independently at every pixel, so tensors are kept in a
``(batch, pixel, step, channel)`` layout: the position encoding is shared by all
pixels of a sample and broadcasts along the pixel axis instead of being copied.
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn

from models.blocks import PositionalEncoder

MODES = ("s2_center", "full")


class TemporalCrossAttention(nn.Module):
    """Cross-attention between two temporal sequences of the same pixel.

    The position encoding is added to the queries and the keys only, so the
    residual stream and the values stay free of positional content.
    """

    def __init__(self, dim: int, n_heads: int = 4):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = math.sqrt(self.head_dim)

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, query, key_value, query_pos, key_pos, key_mask=None):
        """Attend from ``query`` to ``key_value``.

        query     (B, P, T_q, D)     key_value (B, P, T_k, D)
        query_pos (B, 1, T_q, D)     key_pos   (B, 1, T_k, D)
        key_mask  (B, T_k) or None, True where the step is valid
        """
        b, n_pixels, n_query, dim = query.shape
        n_key = key_value.shape[2]
        heads, head_dim = self.n_heads, self.head_dim

        q_tokens = self.norm_q(query) + query_pos
        kv_tokens = self.norm_kv(key_value)

        q = self.to_q(q_tokens).view(b, n_pixels, n_query, heads, head_dim)
        k = self.to_k(kv_tokens + key_pos).view(b, n_pixels, n_key, heads, head_dim)
        v = self.to_v(kv_tokens).view(b, n_pixels, n_key, heads, head_dim)
        q, k, v = (t.transpose(2, 3) for t in (q, k, v))

        scores = torch.matmul(q, k.transpose(-1, -2)) / self.scale
        if key_mask is not None:
            scores = scores.masked_fill(
                ~key_mask[:, None, None, None, :], torch.finfo(scores.dtype).min
            )
        attn = torch.softmax(scores, dim=-1)

        out = torch.matmul(attn, v).transpose(2, 3)
        return query + self.to_out(out.reshape(b, n_pixels, n_query, dim))


class CrossAttentionFusion(nn.Module):
    def __init__(
        self,
        dim: int,
        out_channels: int,
        n_modalities: int,
        n_heads: int = 4,
        mode: str = "s2_center",
        period: int = 1000,
    ):
        super().__init__()
        assert mode in MODES, mode
        assert n_modalities >= 2
        self.mode = mode
        self.n_modalities = n_modalities

        n_blocks = n_modalities - 1 if mode == "s2_center" else n_modalities
        self.blocks = nn.ModuleList(
            [TemporalCrossAttention(dim, n_heads) for _ in range(n_blocks)]
        )
        self.pos_encoder = PositionalEncoder(dim // n_heads, period=period,
                                             repeat=n_heads)
        self.to_out = nn.Sequential(
            nn.GroupNorm(num_groups=1, num_channels=dim * n_modalities),
            nn.Conv2d(dim * n_modalities, out_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
        )

    def forward(
        self,
        modalities: List[torch.Tensor],
        positions: List[torch.Tensor],
        masks: Optional[List[torch.Tensor]] = None,
    ):
        """Fuse ``(B, T_i, C, H, W)`` tensors into ``(B, T_out, C_out, H, W)``.

        ``T_out`` follows the first modality in ``s2_center`` mode and is the
        common step count in ``full`` mode.
        """
        assert len(modalities) == self.n_modalities
        b, _, _, h, w = modalities[0].shape
        n_pixels = h * w

        tokens = [
            x.permute(0, 3, 4, 1, 2).reshape(b, n_pixels, x.shape[1], x.shape[2])
            for x in modalities
        ]
        encodings = [self.pos_encoder(pos.float()).unsqueeze(1) for pos in positions]

        if self.mode == "s2_center":
            fused = [tokens[0]]
            for i, block in enumerate(self.blocks, start=1):
                fused.append(
                    block(
                        tokens[0],
                        tokens[i],
                        encodings[0],
                        encodings[i],
                        None if masks is None else masks[i],
                    )
                )
        else:
            steps = {x.shape[1] for x in modalities}
            assert len(steps) == 1, f"'full' mode needs aligned sequences, got {steps}"
            assert masks is None, "'full' mode runs on compressed slots, never padded"
            fused = []
            for i, block in enumerate(self.blocks):
                others = [j for j in range(self.n_modalities) if j != i]
                fused.append(
                    block(
                        tokens[i],
                        torch.cat([tokens[j] for j in others], dim=2),
                        encodings[i],
                        torch.cat([encodings[j] for j in others], dim=2),
                    )
                )

        out = torch.cat(fused, dim=-1)
        n_steps, cat_dim = out.shape[2], out.shape[3]
        out = out.view(b, h, w, n_steps, cat_dim).permute(0, 3, 4, 1, 2)
        out = self.to_out(out.reshape(b * n_steps, cat_dim, h, w))
        return out.view(b, n_steps, -1, h, w)
