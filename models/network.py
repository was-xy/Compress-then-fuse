"""Segmentation network: a temporal front-end per modality, a fusion stage and
a U-TAE encoder / decoder.

The front-end either compresses the series into ``K`` phenological slots with
the TCM or applies a per-frame convolutional stem and leaves the temporal axis
untouched.  Where the fusion happens decides how many backbones are needed:
``late`` and ``decision`` fuse after the decoder and instantiate one backbone
per modality, every other strategy fuses before the encoder and shares one.
"""

from typing import Any, Dict

import torch
import torch.nn as nn

from configs.experiments import ExperimentConfig
from models.blocks import (
    ConvBlock,
    DownConvBlock,
    TemporalAggregator,
    UpConvBlock,
)
from models.cross_attention import CrossAttentionFusion
from models.ltae import LTAE2d
from models.sacf import SACF
from models.tcm import TemporalCompressionModule, share_parameters

JOINT_FUSIONS = ("single", "early")
PARALLEL_FUSIONS = ("late", "decision")


class ModalityEncoder(nn.Module):
    """Maps one raw time series to the sequence fed to the U-TAE body."""

    def __init__(self, in_channels: int, cfg: ExperimentConfig, alpha_init: float):
        super().__init__()
        self.pad_value = cfg.pad_value
        dim = cfg.encoder_widths[0]

        if cfg.use_tcm:
            self.tcm = TemporalCompressionModule(
                in_channels=in_channels,
                dim=dim,
                n_slots=cfg.tcm_slots,
                n_heads=cfg.tcm_heads,
                alpha_init=alpha_init,
                pad_value=cfg.pad_value,
            )
            self.stem = None
        else:
            self.tcm = None
            self.stem = ConvBlock(
                [in_channels, dim, dim],
                pad_value=cfg.pad_value,
                norm="group",
                padding_mode=cfg.padding_mode,
            )

    def forward(self, x, positions):
        """Returns (features, temporal positions, padding mask or None).

        The compressed sequence has no padding, hence the mask is dropped when
        the TCM is used.
        """
        pad_mask = (x == self.pad_value).all(-1).all(-1).all(-1)
        if self.tcm is not None:
            slots, slot_positions = self.tcm(x, pad_mask, positions)
            return slots, slot_positions, None
        return self.stem.smart_forward(x), positions, pad_mask


class UTAEBody(nn.Module):
    """U-Net encoder and decoder with an L-TAE bottleneck."""

    def __init__(self, cfg: ExperimentConfig):
        super().__init__()
        widths, up_widths = cfg.encoder_widths, cfg.decoder_widths
        n_stages = len(widths)

        self.down_blocks = nn.ModuleList(
            DownConvBlock(
                d_in=widths[i],
                d_out=widths[i + 1],
                k=cfg.str_conv_k,
                s=cfg.str_conv_s,
                p=cfg.str_conv_p,
                pad_value=cfg.pad_value,
                norm=cfg.encoder_norm,
                padding_mode=cfg.padding_mode,
            )
            for i in range(n_stages - 1)
        )
        self.temporal_encoder = LTAE2d(
            in_channels=widths[-1],
            n_head=cfg.ltae_n_head,
            d_k=cfg.ltae_d_k,
            d_model=cfg.ltae_d_model,
            mlp=(cfg.ltae_d_model, widths[-1]),
        )
        self.temporal_aggregator = TemporalAggregator(mode=cfg.agg_mode)
        self.up_blocks = nn.ModuleList(
            UpConvBlock(
                d_in=up_widths[i],
                d_out=up_widths[i - 1],
                d_skip=widths[i - 1],
                k=cfg.str_conv_k,
                s=cfg.str_conv_s,
                p=cfg.str_conv_p,
                norm="batch",
                padding_mode=cfg.padding_mode,
            )
            for i in range(n_stages - 1, 0, -1)
        )

    def forward(self, features, positions, pad_mask=None):
        pyramid = [features]
        for block in self.down_blocks:
            pyramid.append(block.smart_forward(pyramid[-1]))

        out, attn = self.temporal_encoder(pyramid[-1], positions, pad_mask=pad_mask)
        for i, block in enumerate(self.up_blocks):
            skip = self.temporal_aggregator(
                pyramid[-(i + 2)], pad_mask=pad_mask, attn_mask=attn
            )
            out = block(out, skip)
        return out


class CompressThenFuse(nn.Module):
    def __init__(self, cfg: ExperimentConfig):
        super().__init__()
        self.cfg = cfg.validate()
        self.modalities = cfg.modalities

        if cfg.fusion in JOINT_FUSIONS:
            self._build_joint_branch(cfg)
        elif cfg.fusion in PARALLEL_FUSIONS:
            self._build_parallel_branches(cfg)
        else:
            self._build_fused_branches(cfg)
        self._build_heads(cfg)

    # ---- construction ----

    def _build_joint_branch(self, cfg):
        """One branch over the channel-wise concatenation of every modality."""
        self.encoder = ModalityEncoder(
            sum(cfg.input_dims.values()),
            cfg,
            cfg.tcm_alpha_init[self.modalities[0]],
        )
        self.body = UTAEBody(cfg)

    def _build_modality_encoders(self, cfg):
        self.encoders = nn.ModuleDict(
            {
                modality: ModalityEncoder(dim_in, cfg, cfg.tcm_alpha_init[modality])
                for modality, dim_in in cfg.input_dims.items()
            }
        )
        if cfg.use_tcm:
            share_parameters([self.encoders[m].tcm for m in self.modalities])

    def _build_parallel_branches(self, cfg):
        """One full backbone per modality, fused after the decoder."""
        self._build_modality_encoders(cfg)
        self.bodies = nn.ModuleDict(
            {modality: UTAEBody(cfg) for modality in self.modalities}
        )
        if cfg.fusion == "late":
            width = cfg.decoder_widths[0]
            self.fusion = nn.Sequential(
                nn.Conv2d(width * len(self.modalities), width, 1, bias=False),
                nn.BatchNorm2d(width),
                nn.GELU(),
            )

    def _build_fused_branches(self, cfg):
        """One front-end per modality, fused before a shared backbone."""
        self._build_modality_encoders(cfg)
        dim = cfg.encoder_widths[0]
        n_modalities = len(self.modalities)

        if cfg.fusion == "aligned":
            self.fusion = nn.Conv2d(dim * n_modalities, dim, 1, bias=False)
        elif cfg.fusion in ("cross_s2", "cross_nn"):
            self.fusion = CrossAttentionFusion(
                dim=dim,
                out_channels=dim,
                n_modalities=n_modalities,
                n_heads=cfg.fusion_heads,
                mode="s2_center" if cfg.fusion == "cross_s2" else "full",
            )
        else:
            self.fusion = SACF(
                n_modalities=n_modalities,
                in_channels=dim,
                dim=dim,
                out_channels=dim,
                n_heads=cfg.fusion_heads,
            )
        self.body = UTAEBody(cfg)

    def _build_heads(self, cfg):
        def head():
            return ConvBlock(
                [cfg.decoder_widths[0], cfg.decoder_widths[0], cfg.num_classes],
                padding_mode=cfg.padding_mode,
            )

        if cfg.fusion == "decision":
            # Decision fusion averages the per-modality logits, so the heads are
            # the only prediction path and receive no auxiliary supervision.
            self.heads = nn.ModuleDict({m: head() for m in self.modalities})
            return

        self.head = head()
        if cfg.needs_aux_loss:
            self.aux_heads = nn.ModuleDict({m: head() for m in self.modalities})

    # ---- forward ----

    def forward(
        self,
        images: Dict[str, torch.Tensor],
        positions: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        """Returns the logits and, for late fusion, the auxiliary logits."""
        if self.cfg.fusion in JOINT_FUSIONS:
            return self._forward_joint(images, positions)
        if self.cfg.fusion in PARALLEL_FUSIONS:
            return self._forward_parallel(images, positions)
        return self._forward_fused(images, positions)

    def _forward_joint(self, images, positions):
        reference = self.modalities[0]
        lengths = {images[m].shape[1] for m in self.modalities}
        assert len(lengths) == 1, (
            "channel concatenation expects every modality on the reference "
            f"temporal grid, got sequence lengths {sorted(lengths)}"
        )
        x = torch.cat([images[m] for m in self.modalities], dim=2)
        features, pos, pad_mask = self.encoder(x, positions[reference])
        return {"prediction": self.head(self.body(features, pos, pad_mask)), "aux": []}

    def _forward_parallel(self, images, positions):
        decoded = {}
        for modality in self.modalities:
            features, pos, pad_mask = self.encoders[modality](
                images[modality], positions[modality]
            )
            decoded[modality] = self.bodies[modality](features, pos, pad_mask)

        if self.cfg.fusion == "decision":
            logits = [self.heads[m](decoded[m]) for m in self.modalities]
            return {"prediction": sum(logits) / len(logits), "aux": []}

        fused = self.fusion(torch.cat([decoded[m] for m in self.modalities], dim=1))
        return {
            "prediction": self.head(fused),
            "aux": [self.aux_heads[m](decoded[m]) for m in self.modalities],
        }

    def _forward_fused(self, images, positions):
        features, step_positions, masks = [], [], []
        for modality in self.modalities:
            feature, pos, pad_mask = self.encoders[modality](
                images[modality], positions[modality]
            )
            features.append(feature)
            step_positions.append(pos)
            masks.append(None if pad_mask is None else ~pad_mask)

        if self.cfg.fusion == "aligned":
            b, k, c, h, w = features[0].shape
            fused = self.fusion(torch.cat(features, dim=2).reshape(b * k, -1, h, w))
            fused = fused.view(b, k, c, h, w)
        elif self.cfg.fusion == "sacf":
            fused = self.fusion(features)
        else:
            fused = self.fusion(
                features, step_positions, None if masks[0] is None else masks
            )

        # The fused sequence follows the first modality, hence its positions.
        pad_mask = None if masks[0] is None else ~masks[0]
        out = self.body(fused, step_positions[0], pad_mask)
        return {"prediction": self.head(out), "aux": []}


def build_model(cfg: ExperimentConfig) -> CompressThenFuse:
    return CompressThenFuse(cfg)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
