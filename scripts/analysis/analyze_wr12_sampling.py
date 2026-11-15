#!/usr/bin/env python3
"""Export WR-12 resolved pose draws and their distributions, without simulation."""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bench.tasks.wr12.config import load_task_config, resolve_configured_episode_plan
from bench.tasks.wr12.domain.episode import load_sampling_rules, load_station_circles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.count < 1 or args.seed_start < 0:
        parser.error("count must be positive and seed-start nonnegative")
    args.output.mkdir(parents=True, exist_ok=False)
    config = load_task_config()
    rules = load_sampling_rules(config.path(config.resources["sampling_rules"]))
    circles = load_station_circles(config.path(config.resources["stations"]))
    targets, rows, layouts = Counter(), [], set()
    for seed in range(args.seed_start, args.seed_start + args.count):
        plan = resolve_configured_episode_plan(config, seed)
        targets[plan.target_box] += 1
        layouts.add(
            (
                plan.block_offset_xy,
                plan.block_yaw,
                plan.placement_offset_xy,
                tuple(plan.base_xy_by_station.items()),
                tuple(plan.base_yaw_by_station.items()),
            )
        )
        poses = [
            ("block", plan.block_offset_xy, rules["block_radius_m"], plan.block_yaw),
            ("placement", plan.placement_offset_xy, rules["placement_radius_m"], None),
            *(
                (
                    name,
                    np.array(plan.base_xy_by_station[name]) - c["center_xy"],
                    c["radius_m"],
                    plan.base_yaw_by_station[name],
                )
                for name, c in circles.items()
            ),
        ]
        for name, xy, radius, yaw in poses:
            rows.append(
                {
                    "seed": seed,
                    "target_box": plan.target_box,
                    "id": name,
                    "dx_m": float(xy[0]),
                    "dy_m": float(xy[1]),
                    "radius_m": radius,
                    "yaw_rad": yaw,
                }
            )
    with (args.output / "samples.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "scope": "resolved draws only, not a physics success-rate evaluation",
        "profile_id": rules["profile_id"],
        "seed_start": args.seed_start,
        "count": args.count,
        "target_counts": dict(targets),
        "unique_pose_layouts": len(layouts),
        "supports": {},
    }
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), constrained_layout=True)
    for name, ax in zip(("block", "placement", *circles), axes, strict=True):
        values = [row for row in rows if row["id"] == name]
        xy = np.array([[row["dx_m"], row["dy_m"]] for row in values])
        radius = values[0]["radius_m"]
        radial = np.linalg.norm(xy, axis=1)
        if not np.all(radial <= radius + 1e-12):
            raise ValueError(f"{name}: out-of-region pose")
        stats = {
            "radius_m": radius,
            "maximum_radius_m": float(radial.max()),
            "mean_xy_m": xy.mean(axis=0).tolist(),
            "mean_normalized_squared_radius": float(np.mean((radial / radius) ** 2))
            if radius
            else 0.0,
            "uniform_area_expected_mean_normalized_squared_radius": 0.5,
        }
        if values[0]["yaw_rad"] is not None:
            yaw = np.array([row["yaw_rad"] for row in values])
            stats["yaw_min_rad"], stats["yaw_max_rad"] = float(yaw.min()), float(yaw.max())
        summary["supports"][name] = stats
        for target, color in (("left", "tab:blue"), ("right", "tab:orange")):
            selected = xy[[row["target_box"] == target for row in values]] * 100
            if len(selected):
                ax.scatter(
                    selected[:, 0],
                    selected[:, 1],
                    color=color,
                    s=12 if args.count <= 200 else 3,
                    alpha=0.6,
                    label=target,
                )
        ax.add_patch(plt.Circle((0, 0), radius * 100, fill=False, color="black"))
        ax.set(
            title=f"{name}\nR = {radius * 100:g} cm",
            xlabel="offset x (cm)",
            ylabel="offset y (cm)",
            aspect="equal",
            xlim=(-5.5, 5.5),
            ylim=(-5.5, 5.5),
        )
    axes[0].legend(title="remembered box", fontsize=8)
    fig.suptitle(f"WR-12 sampling-v2: {args.count} seeds / {len(layouts)} independent pose layouts")
    fig.savefig(args.output / "sampling_support.png", dpi=160)
    plt.close(fig)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
