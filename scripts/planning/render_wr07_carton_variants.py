#!/usr/bin/env python3
"""Render every WR-07 carton type/finish in a fixed comparison scene."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from PIL import Image, ImageDraw
import yaml

from render_wr_scene import ROOT, _render_free_camera
from bench.runtime import RoboCasaRuntimeFactory
from bench.tasks import TaskCatalog, resolve_builtin_task_plugin
from bench.tasks.contracts import EpisodeRequest
from bench.tasks.wr07.config import DEFAULT_TASK_CONFIG_FILE
from bench.tasks.wr07.domain.contracts import MILK_CARTON_TYPES, MILK_CARTON_MATERIALS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(DEFAULT_TASK_CONFIG_FILE.read_text())
    config['resources'] = {
        key: str((DEFAULT_TASK_CONFIG_FILE.parent / value).resolve())
        for key, value in config['resources'].items()
    }
    api = resolve_builtin_task_plugin('WR-07').load_runtime_api()
    sheets = {view: Image.new('RGB', (len(MILK_CARTON_MATERIALS) * 480,
                                      len(MILK_CARTON_TYPES) * 384), 'white')
              for view in ('closeup', 'scene')}
    records = []
    for row, carton_type in enumerate(MILK_CARTON_TYPES):
        config['episode']['carton_type_candidates'] = [carton_type]
        config['episode']['carton_material_candidates'] = ['matte']
        config_path = output / f'{carton_type}_preview.yaml'
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        episode = TaskCatalog.resolve(EpisodeRequest(
            task_id='WR-07', seed=args.seed, task_config_file=config_path))
        runtime = RoboCasaRuntimeFactory(api).create(episode, execute=False, expert=True)
        try:
            env = runtime.env
            env.sim.model._model.vis.global_.offwidth = 960
            env.sim.model._model.vis.global_.offheight = 720
            pos = env._obj_pos('milk_carton').tolist()
            for col, material in enumerate(MILK_CARTON_MATERIALS):
                # Preview-only appearance changes reuse identical poses and lighting.
                env.carton_material = material
                env._apply_carton_material()
                env.sim.forward()
                paths = {}
                for view in sheets:
                    close = view == 'closeup'
                    path = _render_free_camera(
                        env, output, label=f'{carton_type}_{material}_{view}',
                        lookat=[pos[0], pos[1], pos[2] + 0.03] if close else [2.15, -0.55, 1.0],
                        distance=0.48 if close else 2.25,
                        azimuth=90.0, elevation=-25.0 if close else -32.0,
                        width=960, height=720)
                    with Image.open(path) as image:
                        panel = image.resize((480, 360))
                    sheets[view].paste(panel, (col * 480, row * 384 + 24))
                    ImageDraw.Draw(sheets[view]).text(
                        (col * 480 + 10, row * 384 + 6),
                        f'{carton_type} / {material}', fill='black')
                    paths[view] = str(path)
                records.append({'carton_type': carton_type, 'carton_material': material,
                                'object_position': pos, 'images': paths})
                print(f'Rendered {carton_type} / {material}', flush=True)
        finally:
            runtime.close()
    for view, sheet in sheets.items():
        sheet.save(output / f'all_{view}.jpg', quality=95)
    (output / 'manifest.json').write_text(json.dumps(
        {'seed': args.seed, 'preview_only': True, 'variants': records}, indent=2))
    print(f'All {len(records)} variants saved to {output}', flush=True)


if __name__ == '__main__':
    main()
