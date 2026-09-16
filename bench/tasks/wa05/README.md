# WA-05

This package directly owns WA-05's canonical episode, RoboCasa scene, semantics, expert plan, runtime composition, and resources.

Water is dispensed by two task-owned touchless faucets above the sink. The
robot-left faucet is hot (red), the robot-right faucet is cold (blue), and each
stream is active only while a cup is inside that faucet's proximity zone.

Each episode contains one cup and one fixed service slot. The policy may choose
either faucet, must stage the filled cup, and then physically press the compact
red or blue answer button matching the water actually dispensed. No faucet or
button is highlighted, and the prompt does not reveal the expert's choice.

The cup-pick (P4), straight-ahead intermediate (P6), sink-fill (P8), and
service (P10) base positions are sampled from 5 cm-radius disks once per
episode seed. P5 rotates in place 90 +/- 5 degrees to the left, and P7 rotates
in place 90 +/- 5 degrees to the right. P8 then moves directly to P10; legacy
P9 is not active. All station torso heights stay fixed.

Package layout version: `membench.task_package/v1`.
