#!/usr/bin/env python3
"""Audit resolved WR-11 draws; this is not a physics or policy evaluation."""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bench.tasks.wr11.config import load_task_config, resolve_configured_episode_plan
from bench.tasks.wr11.domain.contracts import ITEM_IDS, PACKAGING_RGB, SAMPLE_ID
from bench.tasks.wr11.domain.sampling import load_station_circles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.count < 30 or args.seed_start < 0:
        parser.error("count must be >=30 and seed-start nonnegative")
    args.output.mkdir(parents=True, exist_ok=False)
    config = load_task_config()
    circles = load_station_circles(config.path(config.resources["stations"]))
    rows, packagings, permutations, combinations = [], Counter(), Counter(), Counter()
    targets = Counter()
    slot_colors = {item: Counter() for item in ITEM_IDS}
    for seed in range(args.seed_start, args.seed_start + args.count):
        plan = resolve_configured_episode_plan(config, seed)
        permutation = tuple(plan.packaging_by_item[item] for item in ITEM_IDS)
        packagings[plan.sample_packaging] += 1
        permutations[permutation] += 1
        combinations[(plan.sample_packaging, *permutation)] += 1
        assert len(set(permutation)) == 3 and permutation.count(plan.sample_packaging) == 1
        targets[plan.matching_item] += 1
        for item in ITEM_IDS:
            slot_colors[item][plan.packaging_by_item[item]] += 1
        for item in (*ITEM_IDS, SAMPLE_ID):
            dx, dy = plan.offset_xy_by_item[item]
            rows.append(
                {
                    "seed": seed,
                    "kind": "object",
                    "id": item,
                    "dx_m": dx,
                    "dy_m": dy,
                    "radius_m": plan.item_sample_radius_m,
                    "yaw_rad": plan.yaw_by_item[item],
                    "sample_packaging": plan.sample_packaging,
                    "matching_item": plan.matching_item,
                }
            )
        for name, xy in plan.base_xy_by_station.items():
            dx, dy = np.array(xy) - circles[name]["center_xy"]
            rows.append(
                {
                    "seed": seed,
                    "kind": "station",
                    "id": name,
                    "dx_m": float(dx),
                    "dy_m": float(dy),
                    "radius_m": plan.base_radius_by_station[name],
                    "yaw_rad": plan.base_yaw,
                    "sample_packaging": plan.sample_packaging,
                    "matching_item": plan.matching_item,
                }
            )
    with (args.output / "samples.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ids = (*ITEM_IDS, SAMPLE_ID, *circles)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    summary = {
        "scope": "resolved draws only; seed order is not simulation time; no physics success-rate claim",
        "profile_id": plan.sampling_profile_id,
        "seed_start": args.seed_start,
        "count": args.count,
        "packaging_counts": dict(packagings),
        "matching_slot_counts": dict(targets),
        "slot_color_counts": {item: dict(counts) for item, counts in slot_colors.items()},
        "theoretical_ordered_stock_layouts": 336,
        "theoretical_request_stock_combinations": 1008,
        "permutation_counts": {"/".join(key): count for key, count in permutations.items()},
        "request_permutation_combinations": len(combinations),
        "supports": {},
    }
    for name, ax in zip(ids, axes.flat):
        values = [row for row in rows if row["id"] == name]
        xy = np.array([[row["dx_m"], row["dy_m"]] for row in values])
        radius = values[0]["radius_m"]
        radial = np.linalg.norm(xy, axis=1)
        assert np.all(radial <= radius + 1e-12)
        yaw = np.array([row["yaw_rad"] for row in values])
        summary["supports"][name] = {
            "count": len(values),
            "radius_m": radius,
            "maximum_radius_m": float(radial.max()),
            "xy_mean_m": xy.mean(axis=0).tolist(),
            "mean_normalized_squared_radius": float(np.mean((radial / radius) ** 2))
            if radius
            else 0.0,
            "uniform_area_expected_mean_normalized_squared_radius": 0.5,
            "yaw_min_rad": float(yaw.min()),
            "yaw_max_rad": float(yaw.max()),
        }
        ax.scatter(xy[:, 0] * 100, xy[:, 1] * 100, c=yaw, cmap="twilight", s=7, alpha=0.6)
        ax.add_patch(plt.Circle((0, 0), radius * 100, fill=False, color="black"))
        ax.set(title=name, xlabel="offset x (cm)", ylabel="offset y (cm)", aspect="equal")
    assert set(packagings) <= set(PACKAGING_RGB) and len(permutations) <= 336
    fig.suptitle(f"WR-11 sampling support: {args.count} seeds; points colored by yaw")
    fig.savefig(args.output / "sampling_support.png", dpi=160)
    plt.close(fig)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {
                "count": args.count,
                "packaging_counts": dict(packagings),
                "permutations": len(permutations),
                "request_permutation_combinations": len(combinations),
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
