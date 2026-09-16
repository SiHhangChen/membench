#!/usr/bin/env python3
"""Generate TS03-style base XY/yaw distributions over WR-07 keypoints."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from bench.execution.navigation.station_poses import StationPoseSampler, normalize_angle
from bench.execution.sampling.circle_sampling import sample_uniform_disk


RESET_CENTER = np.array([2.6327167854768594, -0.8387353492148965])
RESET_YAW = 1.569755536243165
SOURCE_STATION = {
    "center": np.array([2.131064026779481, -0.9289858865987279]),
    "yaw": 1.5697450892050213,
}
TARGET_STATION = {
    "center": np.array([1.6902007516630673, -2.20]),
    "yaw": -1.551592653589793,
}
RADIUS = 0.05
TURN_BASE = TARGET_STATION["yaw"] - SOURCE_STATION["yaw"]
if TURN_BASE > 0:
    TURN_BASE -= 2 * np.pi


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.count <= 0:
        raise ValueError("count must be positive")
    station_file = Path("bench/tasks/wr07/resources/calibration/stations.json")
    rows: list[dict[str, float | int | str]] = []
    seeds = np.arange(args.count)
    for seed in seeds:
        initial_xy = sample_uniform_disk(RESET_CENTER, RADIUS, seed=int(seed), salt=0x57523037)
        sampler = StationPoseSampler.from_file(station_file, seed=int(seed))
        source = sampler.sample("counter_pick_station", purpose="source_observe")
        target = sampler.sample("plate_station", purpose="target_place")
        source_yaw = float(np.arctan2(
            2 * (source.quaternion[3] * source.quaternion[2] + source.quaternion[0] * source.quaternion[1]),
            1 - 2 * (source.quaternion[1] ** 2 + source.quaternion[2] ** 2),
        ))
        target_yaw = float(np.arctan2(
            2 * (target.quaternion[3] * target.quaternion[2] + target.quaternion[0] * target.quaternion[1]),
            1 - 2 * (target.quaternion[1] ** 2 + target.quaternion[2] ** 2),
        ))
        turn_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0x57523037, 0x5455524E]))
        turn = TURN_BASE + float(turn_rng.uniform(-np.deg2rad(5), np.deg2rad(5)))
        turn_yaw = normalize_angle(source_yaw + turn)
        points = [("initial", initial_xy, RESET_YAW),
                  ("source", np.asarray(source.position[:2]), source_yaw),
                  ("turn_complete", np.asarray(source.position[:2]), turn_yaw),
                  ("target", np.asarray(target.position[:2]), target_yaw)]
        for index, (keypoint, xy, yaw) in enumerate(points):
            rows.append({"seed": int(seed), "keypoint_index": index, "keypoint": keypoint,
                         "x_m": float(xy[0]), "y_m": float(xy[1]),
                         "yaw_deg": float(np.rad2deg(yaw))})
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    with (output / "samples.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    quantile_rows = []
    for index, group in __import__("pandas").DataFrame(rows).groupby("keypoint_index"):
        for variable in ("x_m", "y_m", "yaw_deg"):
            values = group[variable]
            for q in (0.05, 0.25, 0.50, 0.75, 0.95):
                quantile_rows.append({"keypoint_index": int(index), "keypoint": group.iloc[0].keypoint,
                                      "variable": variable, "quantile": q, "value": float(values.quantile(q))})
    with (output / "timeseries_quantiles.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(quantile_rows[0])); writer.writeheader(); writer.writerows(quantile_rows)
    df = __import__("pandas").DataFrame(rows)
    fig, axes = __import__("matplotlib.pyplot", fromlist=["subplots"]).subplots(3, 1, figsize=(11, 10), sharex=True)
    for ax, variable, label, unit in zip(axes, ("x_m", "y_m", "yaw_deg"), ("Base x", "Base y", "Base yaw"), ("m", "m", "deg")):
        grouped = df.groupby("keypoint_index")[variable]
        xaxis = np.arange(4)
        for lo, hi, color, alpha in ((0.05, 0.95, "#9ecae1", .22), (0.25, 0.75, "#3182bd", .35)):
            ax.fill_between(xaxis, grouped.quantile(lo).reindex(xaxis), grouped.quantile(hi).reindex(xaxis), color=color, alpha=alpha,
                            label=f"P{int(lo*100):02d}–P{int(hi*100):02d}")
        ax.plot(xaxis, grouped.mean().reindex(xaxis), "o-", color="#222222", lw=2, label="mean")
        ax.plot(xaxis, grouped.median().reindex(xaxis), "s--", color="#6254ae", lw=1.5, label="median")
        ax.set_ylabel(f"{label} ({unit})"); ax.grid(True, alpha=.25); ax.legend(frameon=False)
    axes[-1].set_xticks(range(4), ["initial", "source", "turn complete", "target"])
    axes[-1].set_xlabel("WR-07 motion keypoint (0-based)")
    fig.suptitle(f"WR-07 training: {args.count} seeds, base x / y / yaw", y=.98, fontsize=16)
    fig.text(.5, .945, "Keypoint-aligned commanded poses; bands describe seed variation, not confidence intervals", ha="center", color=".4")
    fig.text(.5, .02, "Initial/source/target XY use 5 cm uniform-area disks. Turn yaw = −178.84° + Uniform(−5°, +5°); XY is held at source.", ha="center", fontsize=8, color=".4")
    fig.tight_layout(rect=(0, .05, 1, .92))
    fig.savefig(output / "01_training_xy_yaw.png", dpi=180); fig.savefig(output / "01_training_xy_yaw.pdf")
    __import__("matplotlib.pyplot", fromlist=["close"]).close(fig)
    summary = {"episodes": args.count, "keypoints": [p[0] for p in points], "radius_m": RADIUS,
               "turn_base_deg": float(np.rad2deg(TURN_BASE)), "turn_range_deg": [float(np.rad2deg(TURN_BASE-np.deg2rad(5))), float(np.rad2deg(TURN_BASE+np.deg2rad(5)))],
               "sampling": "production circle sampler; keypoint-aligned commanded poses", "source_observe_pick_reuse": True}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
