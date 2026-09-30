"""Shape and gradient check for every experiment, on synthetic tensors.

The script needs no data: it feeds random series with modality specific and
deliberately unequal sequence lengths, plus padded frames, through every
configuration of the registry and checks the prediction shape, the auxiliary
outputs and that gradients reach every input.

    python -m tools.check_models
    python -m tools.check_models --exp_id 4,19-29 --device cuda
"""

import argparse
import sys

import torch

from configs.experiments import ALL_EXPERIMENTS, get_config, parse_exp_ids
from models.network import build_model, count_parameters

# Sequence length per source; early fusion resamples everything onto S2.
SEQUENCE_LENGTH = {"S2": 11, "S1A": 9, "S1D": 8, "S1ABcoh": 7, "S1DBcoh": 6}
N_PADDED = 2


def make_batch(cfg, batch_size, size, device):
    """Random series, plus a padded tail on every modality of the last sample."""
    images, positions = {}, {}
    for modality, channels in cfg.input_dims.items():
        source = modality.replace("_fused", "")
        length = SEQUENCE_LENGTH["S2"] if cfg.align_to_s2 else SEQUENCE_LENGTH[source]
        x = torch.randn(batch_size, length, channels, size, size, device=device)
        x[-1, length - N_PADDED:] = cfg.pad_value
        images[modality] = x
        days = torch.arange(length, device=device) * 10 + 5
        positions[modality] = days[None, :].repeat(batch_size, 1)
    return images, positions


def check(cfg, batch_size, size, device):
    torch.manual_seed(0)
    model = build_model(cfg).to(device)
    images, positions = make_batch(cfg, batch_size, size, device)
    for tensor in images.values():
        tensor.requires_grad_(True)

    output = model(images, positions)
    prediction = output["prediction"]
    expected = (batch_size, cfg.num_classes, size, size)
    assert prediction.shape == expected, f"{prediction.shape} != {expected}"

    n_aux = len(cfg.modalities) if cfg.needs_aux_loss else 0
    assert len(output["aux"]) == n_aux, f"{len(output['aux'])} auxiliary outputs"
    for auxiliary in output["aux"]:
        assert auxiliary.shape == expected

    loss = prediction.mean() + sum(a.mean() for a in output["aux"])
    loss.backward()
    for modality, tensor in images.items():
        assert tensor.grad is not None, f"{modality} received no gradient"
        assert torch.isfinite(tensor.grad).all(), f"{modality} gradient is not finite"
        assert tensor.grad.abs().sum() > 0, f"{modality} gradient is zero"

    return count_parameters(model)


def check_cross_attention_invariants(device):
    """The cross-attention fusion must be time aware and ignore padded steps."""
    from models.cross_attention import CrossAttentionFusion

    torch.manual_seed(0)
    b, c, h, w, n = 2, 16, 8, 8, 3
    lengths = {"s2_center": (7, 5, 6), "full": (4, 4, 4)}

    for mode, steps in lengths.items():
        fusion = CrossAttentionFusion(
            dim=c, out_channels=c, n_modalities=n, n_heads=4, mode=mode
        ).to(device).eval()
        features = [
            torch.randn(b, t, c, h, w, device=device) for t in steps
        ]
        positions = [
            torch.arange(t, device=device)[None].repeat(b, 1).float() * 10 + 5
            for t in steps
        ]

        with torch.no_grad():
            reference = fusion(features, positions)
            shifted = list(positions)
            shifted[-1] = shifted[-1].flip(dims=(1,))
            moved = fusion(features, shifted)
        assert not torch.allclose(reference, moved, atol=1e-6), (
            f"{mode}: the output does not depend on the temporal positions"
        )

        if mode != "s2_center":
            continue

        masks = [torch.ones(b, t, dtype=torch.bool, device=device) for t in steps]
        masks[1][:, -2:] = False
        polluted = [f.clone() for f in features]
        polluted[1][:, -2:] = 17.0
        with torch.no_grad():
            masked = fusion(features, positions, masks)
            masked_polluted = fusion(polluted, positions, masks)
        assert torch.allclose(masked, masked_polluted, atol=1e-6), (
            f"{mode}: padded key steps leak into the fused output"
        )
        assert not torch.allclose(masked, reference, atol=1e-6), (
            f"{mode}: the key mask has no effect"
        )

    print("  ok  cross-attention is time aware and masks padded key steps")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp_id", default=None, type=str,
                        help="subset of experiments, default: all")
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--size", default=32, type=int)
    parser.add_argument("--device", default="cpu", type=str)
    args = parser.parse_args()

    exp_ids = (
        parse_exp_ids(args.exp_id) if args.exp_id else sorted(ALL_EXPERIMENTS)
    )
    device = torch.device(args.device)

    failures = []
    try:
        check_cross_attention_invariants(device)
    except Exception as error:
        failures.append(("cross-attention invariants", str(error)))
        print(f"FAIL  cross-attention invariants: {error}")

    for exp_id in exp_ids:
        cfg = get_config(exp_id)
        try:
            n_params = check(cfg, args.batch_size, args.size, device)
            print(f"  ok  {cfg.name:<40s} {n_params:>10,d} parameters")
        except Exception as error:
            failures.append((cfg.name, f"{type(error).__name__}: {error}"))
            print(f"FAIL  {cfg.name:<40s} {type(error).__name__}: {error}")

    n_config_failures = sum(1 for name, _ in failures if name.startswith("exp"))
    print(f"\n{len(exp_ids) - n_config_failures}/{len(exp_ids)} configurations passed")
    if failures:
        for name, message in failures:
            print(f"  {name}: {message}")
        sys.exit(1)


if __name__ == "__main__":
    main()
