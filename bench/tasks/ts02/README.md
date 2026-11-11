# TS02

TS02 places one high-saturation red cube in one of three closed drawers. The
agent searches without revisiting drawers, closes every inspected drawer, stops
when the cube is found, and then presses the spring-loaded completion button
exactly once per inspected drawer. The target location and expert search order
are deterministically sampled from separate seed streams.

After every expert button press, including the last, the gripper retreats above
the button and the arm returns to its regular pose with the gripper open before
the next press or the end of the plan.

Physical drawer operations require every executed prehook, hook, pull and
push stage to pass. A successful plan or an opened drawer cannot override
a failed arm execution.

Version 1.1.0 adds independent seeded drawer variation while keeping layout,
target appearance, prompt, handle friction and all other dynamics fixed:

- Each drawer independently selects a body finish from the eight-entry
  `episode.drawer_material_candidates` pool. The default pool contains light
  wood, walnut, gray wood, bamboo, white paint, blue-gray paint, green paint and
  brushed steel. The fixed cabinet frame and the drawer front, bottom, back,
  and side walls share that finish; the handle and handle mounts remain dark.
- Before each drawer check, `drawer_station` samples uniformly over a disk of
  radius 0.075 m centered on the calibrated (4.85, -1.37) m base XY. Its base
  yaw is sampled uniformly in -pi/2 +/- 5 degrees. The three pure manual-path
  points (start, forward waypoint, and completion-button station) independently
  sample XY on 0.10 m diameter disks; the same episode seed is used for reset
  and navigation so the route is stable within an episode.
  The proximity guard allows 1.15 m to include the disk and navigation tolerance.
- The prehook is selected from the calibrated orientation and conservative
  tool-frame pitch offsets of -5 and +3 degrees. Hook and pull orientations
  remain calibrated. Larger positive pitch and roll candidates are excluded
  because physical validation showed less reliable handle engagement from
  sampled base stations.
- The entire prehook/hook/pull approach is shifted along the fixture-local X
  axis (the long axis of the handle), uniformly in +/-0.015 m. The same offset
  is reapplied when recomputing the pull from the live handle position.
- The calibrated pull distance is 0.18 m so the full station disk retains
  margin above the 0.70 open threshold without driving the 0.22 m drawer slide
  into its mechanical limit, preserving reliable contact during close.

Materials and per-drawer grasp variations are stored in resolved episode facts
and expert plan metadata, never inserted into policy-visible prompts. Station
samples and selected target offsets are also recorded in expert stage results.
Adding candidates does not change the existing target/search-order seed streams.

The appearance catalog in `drawer_materials.py` supports `original`, all 25
`wood/*.png` finishes, all 15 `flat/*.png` paints, all four `metals/*.png`
finishes, and `steel-brushed.png` from RoboCasa's asset texture directory.
Examples beyond the default pool include warm wood, dark wood, parquet,
greenheart wood, black/gray/light-blue paint, brass and bright metal. Red wood
is available but excluded from the default pool to keep the red cube distinct.
The handle-offset half-range is configurable up to 0.04 m; use smaller ranges
when changing the handle or gripper geometry. To disable appearance/grasp
variation, use `[original]`, only the baseline prehook, and zero half-range.
Remove the station's `circle` entry to restore its calibrated fixed sample.

Real contact-motion validation (with the RoboCasa environment activated):

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python scripts/planning/validate_ts02_randomization.py \
  --seeds 2 --full-episode --output-dir outputs/debug/ts02_randomization
```

This package directly owns TS02's canonical episode, RoboCasa scene, physical
edge observer, search-and-count semantics, expert plan, runtime composition,
and resources.

Package layout version: `membench.task_package/v1`.
