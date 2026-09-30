"""Convolutional blocks and the sinusoidal position encoder shared by the
temporal compression module, the fusion modules and the U-TAE backbone."""

import torch
import torch.nn as nn


class PositionalEncoder(nn.Module):
    """Sinusoidal encoding of scalar positions.

    Positions are acquisition days for raw image time series, and phenological
    slot indices once the series has been compressed.  ``repeat`` tiles the
    encoding so that every attention head receives the same signal.
    """

    def __init__(self, dim: int, period: int = 1000, repeat: int = 1):
        super().__init__()
        denominator = torch.pow(period, 2 * (torch.arange(dim).float() // 2) / dim)
        self.register_buffer("denominator", denominator, persistent=False)
        self.repeat = repeat

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """(B, T) positions -> (B, T, dim * repeat) encoding."""
        angles = positions[:, :, None] / self.denominator[None, None, :]
        encoding = torch.empty_like(angles)
        encoding[..., 0::2] = torch.sin(angles[..., 0::2])
        encoding[..., 1::2] = torch.cos(angles[..., 1::2])
        if self.repeat > 1:
            encoding = encoding.repeat(1, 1, self.repeat)
        return encoding


def _norm_layer(norm: str, channels: int, n_groups: int = 4):
    if norm == "batch":
        return nn.BatchNorm2d(channels)
    if norm == "instance":
        return nn.InstanceNorm2d(channels)
    if norm == "group":
        return nn.GroupNorm(num_groups=n_groups, num_channels=channels)
    return None


class ConvLayer(nn.Module):
    """Stack of convolutions, each followed by a normalisation and a ReLU."""

    def __init__(
        self,
        widths,
        norm: str = "batch",
        k: int = 3,
        s: int = 1,
        p: int = 1,
        n_groups: int = 4,
        last_relu: bool = True,
        padding_mode: str = "reflect",
    ):
        super().__init__()
        layers = []
        for i in range(len(widths) - 1):
            layers.append(
                nn.Conv2d(
                    widths[i],
                    widths[i + 1],
                    kernel_size=k,
                    stride=s,
                    padding=p,
                    padding_mode=padding_mode,
                )
            )
            norm_layer = _norm_layer(norm, widths[i + 1], n_groups)
            if norm_layer is not None:
                layers.append(norm_layer)
            if last_relu or i < len(widths) - 2:
                layers.append(nn.ReLU())
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class TemporallySharedBlock(nn.Module):
    """Applies a 2-D block to every frame of a (B, T, C, H, W) tensor.

    A frame whose content is exactly ``pad_value`` comes from the collate
    padding: it is left out of the convolution and its output is filled with
    ``pad_value``, so the padding stays recognisable further down the network.
    """

    def __init__(self, pad_value=None):
        super().__init__()
        self.pad_value = pad_value

    def smart_forward(self, x):
        if x.dim() == 4:
            return self.forward(x)

        b, t = x.shape[:2]
        frames = x.reshape(b * t, *x.shape[2:])

        if self.pad_value is None:
            out = self.forward(frames)
        else:
            valid = ~(frames == self.pad_value).all(-1).all(-1).all(-1)
            if bool(valid.all()):
                out = self.forward(frames)
            else:
                encoded = self.forward(frames[valid])
                out = frames.new_full(
                    (b * t, *encoded.shape[1:]), float(self.pad_value)
                )
                out[valid] = encoded

        return out.view(b, t, *out.shape[1:])


class ConvBlock(TemporallySharedBlock):
    def __init__(
        self,
        widths,
        pad_value=None,
        norm: str = "batch",
        last_relu: bool = True,
        padding_mode: str = "reflect",
    ):
        super().__init__(pad_value=pad_value)
        self.conv = ConvLayer(
            widths, norm=norm, last_relu=last_relu, padding_mode=padding_mode
        )

    def forward(self, x):
        return self.conv(x)


class DownConvBlock(TemporallySharedBlock):
    """Strided convolution followed by a residual pair of convolutions."""

    def __init__(
        self,
        d_in,
        d_out,
        k,
        s,
        p,
        pad_value=None,
        norm: str = "batch",
        padding_mode: str = "reflect",
    ):
        super().__init__(pad_value=pad_value)
        self.down = ConvLayer([d_in, d_in], norm=norm, k=k, s=s, p=p,
                              padding_mode=padding_mode)
        self.conv1 = ConvLayer([d_in, d_out], norm=norm, padding_mode=padding_mode)
        self.conv2 = ConvLayer([d_out, d_out], norm=norm, padding_mode=padding_mode)

    def forward(self, x):
        out = self.conv1(self.down(x))
        return out + self.conv2(out)


class UpConvBlock(nn.Module):
    """Transposed convolution, skip concatenation, residual refinement."""

    def __init__(
        self,
        d_in,
        d_out,
        k,
        s,
        p,
        norm: str = "batch",
        d_skip=None,
        padding_mode: str = "reflect",
    ):
        super().__init__()
        d_skip = d_out if d_skip is None else d_skip
        self.skip_conv = nn.Sequential(
            nn.Conv2d(d_skip, d_skip, kernel_size=1),
            nn.BatchNorm2d(d_skip),
            nn.GELU(),
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(d_in, d_out, kernel_size=k, stride=s, padding=p),
            nn.BatchNorm2d(d_out),
            nn.GELU(),
        )
        self.conv1 = ConvLayer([d_out + d_skip, d_out], norm=norm,
                               padding_mode=padding_mode)
        self.conv2 = ConvLayer([d_out, d_out], norm=norm, padding_mode=padding_mode)

    def forward(self, x, skip):
        out = torch.cat([self.up(x), self.skip_conv(skip)], dim=1)
        out = self.conv1(out)
        return out + self.conv2(out)


class TemporalAggregator(nn.Module):
    """Collapses the temporal axis of the skip connections.

    ``att_group`` reuses the L-TAE attention masks, splitting the channels into
    one group per head; ``mean`` averages the valid frames.
    """

    def __init__(self, mode: str = "att_group"):
        super().__init__()
        assert mode in ("att_group", "mean"), mode
        self.mode = mode

    def forward(self, x, pad_mask=None, attn_mask=None):
        if self.mode == "mean":
            if pad_mask is None:
                return x.mean(dim=1)
            valid = (~pad_mask).float()[:, :, None, None, None]
            return (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

        n_heads, b, t, h, w = attn_mask.shape
        attn = attn_mask.view(n_heads * b, t, h, w)
        if x.shape[-2] > h:
            attn = nn.Upsample(size=x.shape[-2:], mode="bilinear",
                               align_corners=False)(attn)
        else:
            attn = nn.AvgPool2d(kernel_size=h // x.shape[-2])(attn)
        attn = attn.view(n_heads, b, t, *x.shape[-2:])
        if pad_mask is not None:
            attn = attn * (~pad_mask).float()[None, :, :, None, None]

        out = torch.stack(x.chunk(n_heads, dim=2))
        out = (attn[:, :, :, None, :, :] * out).sum(dim=2)
        return torch.cat(list(out), dim=1)
