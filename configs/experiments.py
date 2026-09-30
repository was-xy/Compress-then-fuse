"""Experiment registry.

An experiment is fully described by four axes:

    modality set       which satellite sources the network sees
    fusion             how those sources are combined
    temporal front-end whether the Temporal Compression Module (TCM) is used
    observation window full season, first 3 months, or first 6 months

The registry below is the cartesian product of those axes, so every reported
number has a configuration with a stable identifier.
"""

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Number of channels of each PASTIS-R source.
INPUT_DIM = {
    "S2": 10,
    "S1A": 3,
    "S1D": 3,
    "S1ABcoh": 2,
    "S1DBcoh": 2,
    "S1A_fused": 5,
    "S1D_fused": 5,
}

# Initial value of the TCM lag-mixing logit.  sigmoid(alpha) is the weight of
# the plain weighted standard deviation; 1 - sigmoid(alpha) is the weight of
# the lag-one covariance.  Optical sources start close to the plain deviation,
# speckled SAR sources start close to the lag-one covariance.
TCM_ALPHA_INIT = {
    "S2": 3.0,
    "S1A": -3.0,
    "S1D": -3.0,
    "S1ABcoh": 3.0,
    "S1DBcoh": 3.0,
    "S1A_fused": -3.0,
    "S1D_fused": -3.0,
}

# Modality sets.  "hybrid" keeps three branches but concatenates the
# interferometric coherence onto the matching SAR source channel-wise.
MODALITY_SETS: Dict[str, List[str]] = {
    "S2": ["S2"],
    "S1A": ["S1A"],
    "S1D": ["S1D"],
    "N3": ["S2", "S1A", "S1D"],
    "N5": ["S2", "S1A", "S1D", "S1ABcoh", "S1DBcoh"],
    "hybrid": ["S2", "S1A_fused", "S1D_fused"],
}

# Model-facing modality -> sources read from disk.
HYBRID_SOURCES = {
    "S1A_fused": ("S1A", "S1ABcoh"),
    "S1D_fused": ("S1D", "S1DBcoh"),
}

FUSIONS = (
    "single",     # one modality, no fusion
    "early",      # interpolate every source onto the S2 grid, concatenate channels
    "aligned",    # concatenate the compressed slots, reduce with a 1x1 convolution
    "late",       # one branch per modality, concatenate decoder features, auxiliary supervision
    "decision",   # one branch per modality, average the per-modality logits
    "cross_s2",   # S2 queries every other modality along the temporal axis
    "cross_nn",   # every modality queries every other modality along the slot axis
    "sacf",       # proposed cross-modal fusion
)

# Fusions that operate on slot-aligned features and therefore need the TCM.
FUSIONS_REQUIRING_TCM = ("aligned", "cross_nn", "sacf")

# Length of the observation window, counted in days from the reference date.
SEASON_WINDOWS = {None: "", 91: "3m", 182: "6m"}


@dataclass
class ExperimentConfig:
    exp_id: int
    modality_set: str
    fusion: str
    use_tcm: bool
    early_season_days: Optional[int] = None

    # Segmentation head.
    num_classes: int = 19

    # U-TAE encoder / decoder.
    encoder_widths: List[int] = field(default_factory=lambda: [64, 64, 64, 128])
    decoder_widths: List[int] = field(default_factory=lambda: [32, 32, 64, 128])
    str_conv_k: int = 4
    str_conv_s: int = 2
    str_conv_p: int = 1
    agg_mode: str = "att_group"
    encoder_norm: str = "group"
    padding_mode: str = "zeros"
    pad_value: int = 0

    # L-TAE temporal encoder.
    ltae_n_head: int = 16
    ltae_d_model: int = 256
    ltae_d_k: int = 4

    # Temporal Compression Module.
    tcm_slots: int = 4
    tcm_heads: int = 4

    # Fusion.
    fusion_heads: int = 4
    aux_loss_weight: float = 0.5

    # Temporal dropout applied to the training and validation splits.
    drop_rate_s2: float = 0.4
    drop_rate_s1: float = 0.2

    def validate(self) -> "ExperimentConfig":
        assert self.modality_set in MODALITY_SETS, self.modality_set
        assert self.fusion in FUSIONS, self.fusion
        assert (self.fusion == "single") == (len(self.modalities) == 1), (
            "'single' is the fusion of a one-modality experiment and nothing else"
        )
        if self.fusion in FUSIONS_REQUIRING_TCM:
            assert self.use_tcm, f"fusion '{self.fusion}' operates on compressed slots"
        assert self.early_season_days in SEASON_WINDOWS, self.early_season_days
        assert self.encoder_widths[0] % self.tcm_heads == 0
        assert self.encoder_widths[0] % self.fusion_heads == 0
        return self

    # ---- derived properties ----

    @property
    def modalities(self) -> List[str]:
        """Modalities the network sees."""
        return list(MODALITY_SETS[self.modality_set])

    @property
    def is_hybrid(self) -> bool:
        return self.modality_set == "hybrid"

    @property
    def sources(self) -> List[str]:
        """Sources read from disk."""
        if not self.is_hybrid:
            return self.modalities
        out = []
        for modality in self.modalities:
            out.extend(HYBRID_SOURCES.get(modality, (modality,)))
        return out

    @property
    def input_dims(self) -> Dict[str, int]:
        return {m: INPUT_DIM[m] for m in self.modalities}

    @property
    def align_to_s2(self) -> bool:
        """Early fusion needs every modality on a common temporal grid."""
        return self.fusion == "early"

    @property
    def needs_aux_loss(self) -> bool:
        return self.fusion == "late"

    @property
    def tcm_alpha_init(self) -> Dict[str, float]:
        return {m: TCM_ALPHA_INIT[m] for m in self.modalities}

    @property
    def name(self) -> str:
        parts = [
            f"exp{self.exp_id:03d}",
            self.modality_set,
            self.fusion,
            "tcm" if self.use_tcm else "notcm",
        ]
        window = SEASON_WINDOWS[self.early_season_days]
        if window:
            parts.append(window)
        return "_".join(parts)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_UNIMODAL_SETS = ("S2", "S1A", "S1D")
_MULTIMODAL_SETS = ("N3", "N5", "hybrid")

# (fusion, use_tcm), in reporting order.  Fusions that do not need slot
# alignment are run with and without the temporal compression module.
_MULTIMODAL_VARIANTS = (
    ("early", False),
    ("late", False),
    ("decision", False),
    ("cross_s2", False),
    ("early", True),
    ("aligned", True),
    ("late", True),
    ("decision", True),
    ("cross_s2", True),
    ("cross_nn", True),
    ("sacf", True),
)

_WINDOWS = (None, 91, 182)


def _build_registry() -> Dict[int, ExperimentConfig]:
    registry: Dict[int, ExperimentConfig] = {}
    exp_id = 1

    for modality_set in _UNIMODAL_SETS:
        for use_tcm in (False, True):
            for days in _WINDOWS:
                registry[exp_id] = ExperimentConfig(
                    exp_id=exp_id,
                    modality_set=modality_set,
                    fusion="single",
                    use_tcm=use_tcm,
                    early_season_days=days,
                ).validate()
                exp_id += 1

    for modality_set in _MULTIMODAL_SETS:
        for fusion, use_tcm in _MULTIMODAL_VARIANTS:
            for days in _WINDOWS:
                registry[exp_id] = ExperimentConfig(
                    exp_id=exp_id,
                    modality_set=modality_set,
                    fusion=fusion,
                    use_tcm=use_tcm,
                    early_season_days=days,
                ).validate()
                exp_id += 1

    return registry


ALL_EXPERIMENTS: Dict[int, ExperimentConfig] = _build_registry()


def get_config(exp_id: int) -> ExperimentConfig:
    if exp_id not in ALL_EXPERIMENTS:
        raise KeyError(
            f"unknown experiment {exp_id}; available: "
            f"{min(ALL_EXPERIMENTS)}-{max(ALL_EXPERIMENTS)}"
        )
    return copy.deepcopy(ALL_EXPERIMENTS[exp_id])


def parse_exp_ids(spec: str) -> List[int]:
    """Parse '7', '7,9', '7-12' or '7,9-11' into an ordered list of ids."""
    ids: List[int] = []
    for token in str(spec).replace("，", ",").split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            first, last = (int(v) for v in token.split("-", 1))
            step = 1 if last >= first else -1
            ids.extend(range(first, last + step, step))
        else:
            ids.append(int(token))

    ordered, seen = [], set()
    for exp_id in ids:
        if exp_id not in seen:
            seen.add(exp_id)
            ordered.append(exp_id)

    unknown = [i for i in ordered if i not in ALL_EXPERIMENTS]
    if unknown:
        raise KeyError(f"unknown experiments: {unknown}")
    if not ordered:
        raise ValueError(f"no experiment id parsed from {spec!r}")
    return ordered


def print_experiments() -> None:
    header = (
        f"{'ID':>4s}  {'modalities':<12s}  {'fusion':<10s}  {'TCM':<4s}  "
        f"{'window':<7s}  {'name'}"
    )
    print(header)
    print("-" * len(header))
    for exp_id in sorted(ALL_EXPERIMENTS):
        cfg = ALL_EXPERIMENTS[exp_id]
        window = SEASON_WINDOWS[cfg.early_season_days] or "full"
        print(
            f"{exp_id:>4d}  {cfg.modality_set:<12s}  {cfg.fusion:<10s}  "
            f"{'yes' if cfg.use_tcm else 'no':<4s}  {window:<7s}  {cfg.name}"
        )


if __name__ == "__main__":
    print_experiments()
    print(f"\n{len(ALL_EXPERIMENTS)} experiments")
