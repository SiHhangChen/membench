# WA-05

This package directly owns WA-05's canonical episode, RoboCasa scene, semantics, expert plan, runtime composition, and resources.

Water is dispensed by two task-owned touchless faucets above the sink. The
robot-left faucet is hot (red), the robot-right faucet is cold (blue), and each
stream is active only while a cup is inside that faucet's proximity zone.

正式评测 v2 由专家完成 `pick_cup` 和 `fill_cup`，之后一次性交给 policy 完成放杯和按键。
专家期间 policy 持续接收观测，交接保留模型历史和完整指令。
配置、生成固定动作和计分方式见[固定专家前置评测说明](../../../docs/wa_expert_prefix_evaluation.md)。

Each episode contains one cup and one fixed service slot. In legacy v1 the policy
may choose either faucet; in formal v2 the expert prefix determines the fill.
The policy must stage the filled cup, and then physically press the compact
red or blue answer button matching the water actually dispensed. No faucet or
button is highlighted, and the prompt does not reveal the expert's choice.

Expert recording retains the original placement constraints: the cup center must
be inside the 12 x 12 cm service rectangle and within 12 cm of its target, with
the cup upright, supported by the counter, and released. This also applies when
recording expert prefixes for evaluation-split episodes.

Policy evaluation accepts upright, released placement anywhere on the task's
island counter. It requires counter contact, cup tilt at most 15 degrees, and
the right end effector more than 12 cm from the cup center. Rectangle membership
and distance to the service target do not affect evaluation staging. Canonical
evaluation episodes select this rule via `split="evaluation"`; attaching the
expert runtime restores the stricter recording rule. Diagnostics identify the
active rule in `service.placement_rule`.

The cup-pick (P4), straight-ahead intermediate (P6), sink-fill (P8), and
service (P10) base positions are sampled from 5 cm-radius disks once per
episode seed. P5 rotates in place 90 +/- 5 degrees to the left, and P7 rotates
in place 90 +/- 5 degrees to the right. P8 then moves directly to P10; legacy
P9 is not active. All station torso heights stay fixed.

Package layout version: `membench.task_package/v1`.
