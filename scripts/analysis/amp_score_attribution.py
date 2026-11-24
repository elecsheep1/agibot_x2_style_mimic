"""Offline attribution of AMP discriminator scores to AMP feature blocks.

The AMP discriminator is a linear-output network.  For a policy window we
integrate the score gradient along the straight line from the matched
reference window to the policy window (integrated gradients).  The resulting
per-dimension attribution sums (up to quadrature error) to
``score(policy) - score(matched reference)``, so negative attribution is the
part of the score gap that a feature block pushes down.

This script is read-only: it loads a captured successful-policy rollout
(``*_windows.pt``), the exact training checkpoint and reconstructs the
discriminator from the checkpoint state dict.  It does not import or modify
the training environment, and it never writes under ``source/``.

Typical use::

    python scripts/analysis/amp_score_attribution.py \
        --capture logs/acceptance/experimentA_9500_comparison/A_windows.pt \
        --label A_model9500

The recommended policy is the one validated by the acceptance comparison
(64/64 fixed-first-frame successes); do not point this at a run that cannot
stand up, because its windows mix failures into the attribution.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn

NUM_STEPS = 5
STATE_DIM = 175
BODY_COUNT = 11
JOINT_COUNT = 29

AMP_KEY_BODY_NAMES = (
    "torso_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "left_knee_link",
    "right_knee_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
)

# Keep in sync with scripts/common/amp_feature_layout.py (175D layout).
BLOCKS = {
    "root_height": (0, 1),
    "root_gravity": (1, 4),
    "root_linear": (4, 7),
    "root_angular_scaled": (7, 10),
    "body_position": (10, 43),
    "body_rotation6d": (43, 109),
    "body_linear": (109, 142),
    "body_angular_scaled": (142, 175),
}
BODY_BLOCK_WIDTHS = {
    "body_position": 3,
    "body_rotation6d": 6,
    "body_linear": 3,
    "body_angular_scaled": 3,
}
BODY_COMPONENTS = ("position", "rotation6d", "linear", "angular_scaled")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True,
                        help="Captured *_windows.pt from a successful policy evaluation.")
    parser.add_argument("--label", type=str, default=None,
                        help="Human-readable policy label; defaults to the capture stem.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory for the JSON/Markdown report.")
    parser.add_argument("--phase", choices=("getup", "standing", "all"), default="getup",
                        help="Primary phase reported first; the other phases are still summarized.")
    parser.add_argument("--ig-steps", type=int, default=16,
                        help="Midpoint quadrature steps for integrated gradients.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-windows", type=int, default=None,
                        help="Smoke-test limit for the primary phase.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_discriminator(checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    amp = checkpoint["amp_window"]
    state = amp["discriminator"]
    settings = amp["settings"]
    network_state = {
        key.removeprefix("network."): value
        for key, value in state.items()
        if key.startswith("network.")
    }
    indices = sorted(
        int(key.split(".")[0])
        for key in network_state
        if key.endswith(".weight")
    )
    layers = []
    for index in indices:
        weight = network_state[f"{index}.weight"]
        layers.append(nn.Linear(weight.shape[1], weight.shape[0]))
        if index != indices[-1]:
            layers.append(nn.ELU())
    network = nn.Sequential(*layers)
    network.load_state_dict(network_state)
    network.eval()
    for parameter in network.parameters():
        parameter.requires_grad_(False)
    return network, state, settings


def normalize(states, state_mean, state_std, clip):
    normalized = (states - state_mean) / state_std
    if clip is not None:
        normalized = normalized.clamp(-clip, clip)
    return normalized


def feature_names():
    names = [f"root_height"]
    for prefix, width in (
        ("root_gravity", 3),
        ("root_linear", 3),
        ("root_angular_scaled", 3),
    ):
        for axis in "xyz"[:width]:
            names.append(f"{prefix}[{axis}]")
    for block in ("body_position", "body_rotation6d", "body_linear", "body_angular_scaled"):
        start = BLOCKS[block][0]
        width = BODY_BLOCK_WIDTHS[block]
        for body in AMP_KEY_BODY_NAMES:
            for offset in range(width):
                if block == "body_rotation6d":
                    suffix = f"r6d[{offset}]"
                else:
                    suffix = "xyz"[offset]
                names.append(f"{body}/{block.removeprefix('body_')}[{suffix}]")
    if len(names) != STATE_DIM:
        raise RuntimeError(f"feature name count {len(names)} != {STATE_DIM}")
    return names


def block_of_dim(dim):
    for name, (start, end) in BLOCKS.items():
        if start <= dim < end:
            return name
    raise KeyError(dim)


def body_of_dim(dim):
    for block, width in BODY_BLOCK_WIDTHS.items():
        start, end = BLOCKS[block]
        if start <= dim < end:
            index = (dim - start) // width
            return AMP_KEY_BODY_NAMES[index], block
    return None, None


def integrated_gradients(network, policy, reference, steps):
    """Midpoint-rule integrated gradients in normalized input space."""
    delta = policy - reference
    accumulated = torch.zeros_like(delta)
    for step in range(steps):
        alpha = (step + 0.5) / steps
        point = (reference + alpha * delta).detach().requires_grad_(True)
        score = network(point.flatten(1)).flatten()
        gradient = torch.autograd.grad(score.sum(), point)[0]
        accumulated += gradient
    return delta * (accumulated / steps)


def quantiles(values):
    if len(values) == 0:
        return {}
    q = torch.quantile(
        values.float(),
        values.new_tensor([0.1, 0.5, 0.9]),
    )
    return {
        "mean": float(values.float().mean()),
        "p10": float(q[0]),
        "p50": float(q[1]),
        "p90": float(q[2]),
        "min": float(values.float().min()),
        "max": float(values.float().max()),
    }


def score_windows(network, windows, state_mean, state_std, clip, batch_size):
    values = []
    for start in range(0, len(windows), batch_size):
        batch = windows[start:start + batch_size]
        normalized = normalize(batch, state_mean, state_std, clip)
        values.append(network(normalized.flatten(1)).flatten().detach())
    return torch.cat(values) if values else torch.empty(0)


def attribute(network, windows, reference_windows, state_mean, state_std, clip,
              steps, batch_size):
    """Return per-window attribution [N, NUM_STEPS, STATE_DIM] and scores."""
    attributions = []
    policy_scores = []
    reference_scores = []
    for start in range(0, len(windows), batch_size):
        batch_windows = windows[start:start + batch_size]
        batch_reference = reference_windows[start:start + batch_size]
        policy = normalize(batch_windows, state_mean, state_std, clip).detach()
        reference = normalize(batch_reference, state_mean, state_std, clip).detach()
        with torch.no_grad():
            policy_scores.append(network(policy.flatten(1)).flatten())
            reference_scores.append(network(reference.flatten(1)).flatten())
        attributions.append(integrated_gradients(network, policy, reference, steps))
    return (
        torch.cat(attributions),
        torch.cat(policy_scores),
        torch.cat(reference_scores),
    )


def summarize_attribution(attribution, policy_scores, reference_scores, dim_names):
    """Aggregate [N, T, D] attribution into blocks, bodies and dimensions."""
    per_dim = attribution.sum(dim=1)  # [N, D]
    mean_per_dim = per_dim.mean(dim=0)
    score_gap = (policy_scores - reference_scores)
    mean_gap = float(score_gap.mean())
    completeness = (
        per_dim.sum(dim=1) - score_gap
    ).abs().mean() / score_gap.abs().mean().clamp_min(1e-8)

    block_rows = []
    for name, (start, end) in BLOCKS.items():
        value = float(mean_per_dim[start:end].sum())
        block_rows.append({
            "block": name,
            "dimensions": end - start,
            "mean_attribution": value,
            "share_of_gap_percent": 100.0 * value / mean_gap if abs(mean_gap) > 1e-12 else None,
        })
    block_rows.sort(key=lambda row: row["mean_attribution"])

    body_totals = {body: 0.0 for body in AMP_KEY_BODY_NAMES}
    component_totals = {
        (body, component): 0.0
        for body in AMP_KEY_BODY_NAMES
        for component in BODY_COMPONENTS
    }
    for dim, name in enumerate(dim_names):
        body, block = body_of_dim(dim)
        if body is None:
            continue
        body_totals[body] += float(mean_per_dim[dim])
        component = block.removeprefix("body_")
        component_totals[(body, component)] += float(mean_per_dim[dim])

    body_rows = [
        {"body": body, "mean_attribution": value}
        for body, value in sorted(body_totals.items(), key=lambda item: item[1])
    ]
    component_rows = [
        {
            "body": body,
            "component": component,
            "mean_attribution": value,
        }
        for (body, component), value in sorted(
            component_totals.items(), key=lambda item: item[1]
        )
    ]

    dim_rows = [
        {
            "dimension": dim,
            "feature": dim_names[dim],
            "mean_attribution": float(mean_per_dim[dim]),
        }
        for dim in range(len(dim_names))
    ]
    dim_rows.sort(key=lambda row: row["mean_attribution"])

    return {
        "count": int(len(policy_scores)),
        "score_policy": quantiles(policy_scores),
        "score_matched_reference": quantiles(reference_scores),
        "score_gap_mean": mean_gap,
        "completeness_relative_error": float(completeness),
        "blocks": block_rows,
        "bodies": body_rows,
        "body_components": component_rows,
        "top_negative_dimensions": dim_rows[:25],
        "top_positive_dimensions": dim_rows[-10:][::-1],
    }


def hybrid_substitution(network, windows, reference_windows, state_mean, state_std,
                        clip, batch_size):
    """Replace selected feature groups with the matched reference and rescore."""
    groups = {
        "body_linear": [BLOCKS["body_linear"]],
        "body_angular_scaled": [BLOCKS["body_angular_scaled"]],
        "all_body_relative_velocities": [
            BLOCKS["body_linear"],
            BLOCKS["body_angular_scaled"],
        ],
        "body_position": [BLOCKS["body_position"]],
        "body_rotation6d": [BLOCKS["body_rotation6d"]],
        "body_pose": [BLOCKS["body_position"], BLOCKS["body_rotation6d"]],
        "root_linear": [BLOCKS["root_linear"]],
        "root_angular_scaled": [BLOCKS["root_angular_scaled"]],
        "all_velocity_blocks": [
            BLOCKS["root_linear"],
            BLOCKS["root_angular_scaled"],
            BLOCKS["body_linear"],
            BLOCKS["body_angular_scaled"],
        ],
        "all_features": [(0, STATE_DIM)],
    }
    policy_base = normalize(windows, state_mean, state_std, clip)
    reference = normalize(reference_windows, state_mean, state_std, clip)
    with torch.no_grad():
        base_scores = []
        for start in range(0, len(windows), batch_size):
            batch = policy_base[start:start + batch_size]
            base_scores.append(network(batch.flatten(1)).flatten())
        base_scores = torch.cat(base_scores)
    result = {}
    for name, slices in groups.items():
        deltas = []
        for start in range(0, len(windows), batch_size):
            hybrid = policy_base[start:start + batch_size].clone()
            for lo, hi in slices:
                hybrid[..., lo:hi] = reference[start:start + batch_size, :, lo:hi]
            with torch.no_grad():
                score = network(hybrid.flatten(1)).flatten()
            deltas.append(score - base_scores[start:start + batch_size])
        delta = torch.cat(deltas)
        result[name] = {
            "mean_score_change": float(delta.mean()),
            "median_score_change": float(delta.median()),
        }
    return result


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(8, torch.get_num_threads()))

    capture_path = args.capture.resolve()
    capture = torch.load(capture_path, map_location="cpu", weights_only=False, mmap=True)
    checkpoint = Path(capture["checkpoint"]).resolve()
    label = args.label or capture_path.stem.removesuffix("_windows")

    if int(capture["state_dim"]) != STATE_DIM or int(capture["num_steps"]) != NUM_STEPS:
        raise ValueError("Capture does not use the expected 175D x 5-step layout")

    network, state, settings = load_discriminator(checkpoint)
    state_mean = state["state_mean"]
    state_std = state["state_std"]
    clip = settings.get("normalization_clip")

    windows = capture["windows"]
    reference_windows = capture["reference_windows"]
    matched = reference_windows[capture["nearest_reference_window_index"]]
    hold = capture["hold_s"]
    standing_flag = capture.get("standing", hold > NUM_STEPS * float(capture["step_dt"]))
    phases = {
        "getup": hold <= 0.10,
        "standing": standing_flag.bool(),
        "all": torch.ones_like(hold, dtype=torch.bool),
    }

    # Reconstruct captured scores to prove the checkpoint reconstruction is exact.
    reconstructed = score_windows(
        network, windows, state_mean, state_std, clip, args.batch_size
    )
    captured_scores = capture["score"].float()
    max_score_error = float((reconstructed - captured_scores).abs().max())
    if max_score_error > 1e-4:
        raise RuntimeError(
            f"Reconstructed discriminator scores differ from capture: {max_score_error}"
        )

    # Sanity: the reference windows themselves should score near +1.
    reference_self = score_windows(
        network, reference_windows, state_mean, state_std, clip, args.batch_size
    )

    dim_names = feature_names()
    report = {
        "method": (
            "Integrated gradients (midpoint quadrature) of the checkpoint AMP "
            "discriminator score with respect to the normalized 5-frame window, "
            "along the line from the pose-matched reference window to the policy "
            "window. Positive attribution raises the policy score; negative "
            "attribution lowers it. Attribution sums to score(policy)-score(ref)."
        ),
        "capture": str(capture_path),
        "checkpoint": str(checkpoint),
        "label": label,
        "feature_version": capture.get("feature_version"),
        "num_steps": NUM_STEPS,
        "state_dim": STATE_DIM,
        "ig_steps": args.ig_steps,
        "normalization": {"clip": clip},
        "sanity": {
            "max_reconstructed_score_error": max_score_error,
            "reference_self_score": quantiles(reference_self),
            "reference_count": int(len(reference_windows)),
        },
        "phase_counts": {name: int(mask.sum()) for name, mask in phases.items()},
        "phases": {},
        "phase_by_height": {},
    }

    phase_masks = {}
    for name, mask in phases.items():
        if not bool(mask.any()):
            continue
        phase_masks[name] = mask
    # The primary phase is computed first so a smoke test can limit it.
    primary = args.phase
    if primary not in phase_masks:
        primary = "all" if "all" in phase_masks else next(iter(phase_masks))

    for name in [primary] + [p for p in phase_masks if p != primary]:
        mask = phase_masks[name]
        selected_windows = windows[mask]
        selected_reference = matched[mask]
        if name == primary and args.max_windows is not None:
            selected_windows = selected_windows[:args.max_windows]
            selected_reference = selected_reference[:args.max_windows]
        attribution, policy_scores, reference_scores = attribute(
            network,
            selected_windows,
            selected_reference,
            state_mean,
            state_std,
            clip,
            args.ig_steps,
            args.batch_size,
        )
        summary = summarize_attribution(
            attribution, policy_scores, reference_scores, dim_names
        )
        summary["hybrid_substitution"] = hybrid_substitution(
            network,
            selected_windows,
            selected_reference,
            state_mean,
            state_std,
            clip,
            args.batch_size,
        )
        report["phases"][name] = summary

        if name in ("getup", "standing", "all") and name != primary:
            continue
        # Height bins are only meaningful for the getup phase; use endpoint root height.
        if name == "getup":
            endpoint_height = selected_windows[:, -1, 0]
            bins = {
                "lt_0.20m": endpoint_height < 0.20,
                "0.20_to_0.45m": (endpoint_height >= 0.20) & (endpoint_height < 0.45),
                "ge_0.45m": endpoint_height >= 0.45,
            }
            for bin_name, bin_mask in bins.items():
                if not bool(bin_mask.any()):
                    continue
                bin_attr = attribution[bin_mask]
                bin_policy = policy_scores[bin_mask]
                bin_reference = reference_scores[bin_mask]
                bin_summary = summarize_attribution(
                    bin_attr, bin_policy, bin_reference, dim_names
                )
                report["phase_by_height"][bin_name] = {
                    "count": int(bin_mask.sum()),
                    "score_policy": bin_summary["score_policy"],
                    "score_gap_mean": bin_summary["score_gap_mean"],
                    "blocks": bin_summary["blocks"],
                    "bodies": bin_summary["bodies"],
                    "top_negative_dimensions": bin_summary["top_negative_dimensions"][:10],
                }

    # Reference self-score by motion, for context.
    motion_ids = capture.get("motion_id")
    if motion_ids is not None:
        report["reference_self_score_by_motion"] = {}
        # The capture's reference_windows order concatenates clips; use counts when available.
        counts = capture.get("reference_window_counts")
        if counts:
            offset = 0
            for motion_index, count in enumerate(counts):
                segment = reference_self[offset:offset + count]
                offset += count
                report["reference_self_score_by_motion"][str(motion_index)] = quantiles(segment)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            Path("logs/analysis") /
            f"amp_score_attribution_{label}_20261124"
        )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "attribution.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    write_markdown(report, output_dir / "attribution.md")
    print(json.dumps({
        "output": str(output_dir),
        "primary_phase": primary,
        "primary_score_policy": report["phases"][primary]["score_policy"],
        "primary_score_gap": report["phases"][primary]["score_gap_mean"],
        "completeness_relative_error": report["phases"][primary]["completeness_relative_error"],
        "top_blocks": report["phases"][primary]["blocks"][:5],
        "reference_self_score": report["sanity"]["reference_self_score"],
    }, indent=2, ensure_ascii=False))


def write_markdown(report, path: Path) -> None:
    lines = []
    lines.append(f"# AMP score attribution: {report['label']}")
    lines.append("")
    lines.append(f"- capture: `{report['capture']}`")
    lines.append(f"- checkpoint: `{report['checkpoint']}`")
    lines.append(f"- feature version: `{report['feature_version']}`")
    lines.append(f"- method: {report['method']}")
    lines.append("")
    sanity = report["sanity"]
    lines.append("## Sanity")
    lines.append("")
    lines.append(f"- max reconstructed score error: {sanity['max_reconstructed_score_error']:.2e}")
    ref = sanity["reference_self_score"]
    lines.append(
        f"- reference self-score: mean {ref['mean']:.4f}, "
        f"p10/p50/p90 {ref['p10']:.4f}/{ref['p50']:.4f}/{ref['p90']:.4f}"
    )
    lines.append("")
    for phase_name, summary in report["phases"].items():
        score = summary["score_policy"]
        reference = summary["score_matched_reference"]
        lines.append(f"## Phase `{phase_name}` ({summary['count']} windows)")
        lines.append("")
        lines.append(
            f"- policy score: mean {score['mean']:.4f}, "
            f"p10/p50/p90 {score['p10']:.4f}/{score['p50']:.4f}/{score['p90']:.4f}"
        )
        lines.append(
            f"- matched reference score: mean {reference['mean']:.4f}, "
            f"p50 {reference['p50']:.4f}"
        )
        lines.append(f"- mean score gap: {summary['score_gap_mean']:+.4f}")
        lines.append(
            f"- completeness relative error: "
            f"{summary['completeness_relative_error']:.3f}"
        )
        lines.append("")
        lines.append("### Blocks (negative lowers the policy score)")
        lines.append("")
        lines.append("| block | dims | mean attribution | share of gap |")
        lines.append("|---|---:|---:|---:|")
        for row in summary["blocks"]:
            share = row["share_of_gap_percent"]
            share_text = "n/a" if share is None else f"{share:+.1f}%"
            lines.append(
                f"| {row['block']} | {row['dimensions']} | "
                f"{row['mean_attribution']:+.4f} | {share_text} |"
            )
        lines.append("")
        lines.append("### Bodies")
        lines.append("")
        lines.append("| body | mean attribution |")
        lines.append("|---|---:|")
        for row in summary["bodies"]:
            lines.append(f"| {row['body']} | {row['mean_attribution']:+.4f} |")
        lines.append("")
        lines.append("### Top negative dimensions")
        lines.append("")
        lines.append("| feature | mean attribution |")
        lines.append("|---|---:|")
        for row in summary["top_negative_dimensions"][:15]:
            lines.append(f"| {row['feature']} | {row['mean_attribution']:+.4f} |")
        lines.append("")
        hybrid = summary["hybrid_substitution"]
        lines.append("### Matched-reference substitution (score change)")
        lines.append("")
        lines.append("| replaced group | mean | median |")
        lines.append("|---|---:|---:|")
        for name, value in hybrid.items():
            lines.append(
                f"| {name} | {value['mean_score_change']:+.4f} | "
                f"{value['median_score_change']:+.4f} |"
            )
        lines.append("")
    if report["phase_by_height"]:
        lines.append("## Getup phase by endpoint root height")
        lines.append("")
        lines.append("| height bin | count | policy score p50 | score gap | top negative block |")
        lines.append("|---|---:|---:|---:|---|")
        for name, summary in report["phase_by_height"].items():
            top = summary["blocks"][0]
            lines.append(
                f"| {name} | {summary['count']} | "
                f"{summary['score_policy']['p50']:+.4f} | "
                f"{summary['score_gap_mean']:+.4f} | "
                f"{top['block']} ({top['mean_attribution']:+.4f}) |"
            )
        lines.append("")
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
