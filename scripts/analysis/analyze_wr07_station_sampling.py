#!/usr/bin/env python3
"""Audit WR-07 sampled docking targets, not executed robot trajectories."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bench.execution.navigation.station_poses import (
    StationPoseSampler, normalize_angle, quat_xyzw_to_yaw,
)
from bench.tasks import TaskCatalog
from bench.tasks.contracts import EpisodeRequest
from bench.tasks.wr07.config import DEFAULT_TASK_CONFIG_FILE, load_task_config
from bench.tasks.wr07.execution.navigation import SourcePlateStationSampler
from scripts.analysis.analyze_wa06_station_sampling import autocorrelation
from scripts.analysis.analyze_wa02_state_separability import analyze_point


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count', type=int, default=200)
    parser.add_argument('--permutations', type=int, default=2000)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.count < 30 or args.permutations < 1:
        parser.error('count must be >= 30 and permutations positive')
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    config = load_task_config(DEFAULT_TASK_CONFIG_FILE)
    station_file = config.path(config.resources['stations'])
    specs = json.loads(station_file.read_text())['stations']
    rows = []
    reuse = []
    for seed in range(args.count):
        facts = TaskCatalog.resolve(EpisodeRequest(task_id='WR-07', seed=seed)).facts
        source_slot = facts['source_slot_by_item']['milk_carton']
        target_slot = facts['target_slot_by_color'][facts['source_color_by_item']['milk_carton']]
        sampler = SourcePlateStationSampler(
            StationPoseSampler.from_file(station_file, seed=seed), None)
        observe = sampler.sample('counter_pick_station', purpose='source_observe')
        pick = sampler.sample('counter_pick_station', purpose='source_pick')
        reuse.append(observe == pick)
        target = sampler.sample('plate_station', purpose='target_place')
        for station, pose in [('counter_pick_station', observe), ('plate_station', target)]:
            circle = specs[station]['circle']
            x, y = pose.position[:2]
            dx, dy = np.asarray([x, y]) - circle['center_xy']
            yaw = quat_xyzw_to_yaw(pose.quaternion)
            rows.append(dict(seed=seed, station=station, source_slot=source_slot,
                             target_slot=target_slot, x=x, y=y, dx=float(dx), dy=float(dy),
                             radius=float(np.hypot(dx, dy)), yaw=yaw,
                             yaw_delta=normalize_angle(yaw - circle['base_yaw'])))
    with (output / 'samples.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(task='WR-07', seed_range=[0, args.count - 1],
                   scope='sampled docking targets; seed order is not simulation time',
                   source_observe_pick_identical_count=sum(reuse), stations={})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.stats import kstest, chisquare
    fig, axes = plt.subplots(2, 4, figsize=(19, 9), constrained_layout=True)
    arrays_by_station = {}
    for i, station in enumerate(specs):
        selected = [row for row in rows if row['station'] == station]
        arrays = {key: np.asarray([r[key] for r in selected])
                  for key in ('dx', 'dy', 'radius', 'yaw_delta')}
        arrays_by_station[station] = arrays
        circle = specs[station]['circle']
        radius = circle['radius_m']
        u = (arrays['radius'] / radius) ** 2
        quadrants = np.bincount((arrays['dx'] < 0).astype(int) * 2 +
                               (arrays['dy'] < 0).astype(int), minlength=4)
        stat = dict(circle=circle, count=len(selected),
                    expected_radius_mean_m=2 * radius / 3,
                    expected_xy_std_m=radius / 2,
                    quadrant_counts=quadrants.tolist(),
                    quadrant_chisquare_p=float(chisquare(quadrants).pvalue),
                    area_uniform_ks_p=float(kstest(u, 'uniform').pvalue),
                    metrics={}, target_slot_probes={})
        for key, values in arrays.items():
            stat['metrics'][key] = dict(mean=float(values.mean()),
                std=float(values.std(ddof=1)), min=float(values.min()), max=float(values.max()),
                lag_autocorrelation={str(lag): autocorrelation(values, lag)
                                     for lag in (1, 2, 5, 10, 20)},
                seed_pearson=float(np.corrcoef(np.arange(args.count), values)[0, 1]))
        stat['consecutive_xy_step_mean_m'] = float(np.hypot(
            np.diff(arrays['dx']), np.diff(arrays['dy'])).mean())
        # TS03's binary probes are reused pairwise for the three target slots.
        X = np.column_stack([arrays[k] for k in ('dx', 'dy', 'yaw_delta')])
        labels = np.asarray([row['target_slot'] for row in selected])
        for a, b in itertools.combinations(sorted(set(labels)), 2):
            valid = (labels == a) | (labels == b)
            probe = analyze_point(X[valid], (labels[valid] == b).astype(np.int8), args.permutations)
            for dimension, name in zip(probe['dimensions'], ('base_dx', 'base_dy', 'base_yaw_delta')):
                dimension['name'] = name
            probe['class_counts'] = {a: int(np.sum(labels == a)), b: int(np.sum(labels == b))}
            stat['target_slot_probes'][f'{a}_vs_{b}'] = probe
        summary['stations'][station] = stat
        axes[i, 0].scatter(arrays['dx'] * 1000, arrays['dy'] * 1000,
                           c=np.arange(args.count), s=14, cmap='viridis')
        axes[i, 0].add_patch(plt.Circle((0, 0), radius * 1000, fill=False, color='gray'))
        axes[i, 0].set(xlabel='dx (mm)', ylabel='dy (mm)', title=station, aspect='equal')
        axes[i, 1].plot(arrays['dx'] * 1000, label='dx', lw=0.8)
        axes[i, 1].plot(arrays['dy'] * 1000, label='dy', lw=0.8)
        axes[i, 1].set(xlabel='seed', ylabel='offset (mm)', title='seed-order XY series')
        axes[i, 1].legend()
        axes[i, 2].plot(arrays['radius'] * 1000, lw=0.8)
        axes[i, 2].set(xlabel='seed', ylabel='radius (mm)', title='seed-order radial series')
        axes[i, 3].plot(np.rad2deg(arrays['yaw_delta']), lw=0.8)
        axes[i, 3].set(xlabel='seed', ylabel='yaw offset (deg)', title='seed-order yaw series')
    a, b = list(arrays_by_station.values())
    summary['cross_station_pearson'] = {
        key: float(np.corrcoef(a[key], b[key])[0, 1]) for key in a}
    summary['target_slot_counts'] = {
        slot: sum(r['target_slot'] == slot for r in rows[::2])
        for slot in sorted(set(r['target_slot'] for r in rows))}
    fig.savefig(output / 'timeseries.png', dpi=160)
    plt.close(fig)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
