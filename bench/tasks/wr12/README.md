# WR-12：遮挡后的记忆投放

机器人从侧面观察两个带顶的收纳盒，其中一个装有罐头。开局观察 1 秒后，
两个箱子的开口由环境同步自动封闭，之后保持锁定。机器人移到正面，
抓起公共方块，放入记住的盒子前方托盘。
机器人无需关门、柜内取物或携物导航。场景为 layout 33/style 34/seed 0。

方块与放置点对齐 WR07/WR08，在半径 4.5 cm 圆内按面积均匀采样；方块 yaw
在 `[−π, π]` 内均匀采样。站位半径为 5 cm，朝向独立扰动 ±5°。每个 seed
独立采样物体姿态、放置点和两个站位；相邻 seed 仅保持左右目标各一次，不复用姿态。
固定 prompt 不含目标、seed 或专家阶段。三路 RGB、state37、action13
使用现有数据接口。盒子尺寸为 34×40×30 cm，托盘为 30×27 cm，方块为 5×5×7 cm。
侧开口带相同的前缘挡边，以阻断腕部相机从操作区斜视罐头。

罐头使用原资产的 3 倍尺寸，实际包围盒约为直径 18.2 cm、高 24.3 cm，由盒底真实支撑。
左箱比右箱后移 15 cm，避免左箱遮住右箱开口；两个箱子的朝向相同，前缘挡边均为
12 cm。罐头相对箱心的 XY 偏移分别为左箱 `(−5, −9)` cm、右箱 `(6, −9)` cm，
兼顾侧面观察与正面遮挡。箱体布置与目标无关，每集仅罐头所在箱不同。
观察窗口为 20 个控制步（1 秒），之后挡板用 6 步（0.3 秒）落下；
第 26 步完全封闭。仍只从同一侧观察，确认封口和记忆屏障后再移至正面选择。

判分要求先观察罐头，随后两箱封闭、三路相机完整隐藏且存在盒壁遮挡连续三步，
建立记忆屏障；专家通常在第 28 步通过。屏障建立后重开箱口、重新看到罐头或
可见性未知会终止。抓取还需到达操作区、真实持有并抬升，
首次与托盘接触且脱离夹爪的释放锁定选择；稳定支撑三步后判定放置与目标正确性。
放错、换盘或移动罐头均不能通过。公共报告仅有抓取、放置、成功三个阶段，
使用 `WR-12/report-v2`；观察和遮挡仍作为内部判分条件保留在 diagnostics。

盒体和托盘为固定碰撞物体；方块与罐头使用自由关节。初始化之后不写这两者的位置，
不使用抓取附着。挡板为带碰撞的不透明 mocap 物体，由模拟时间统一驱动，
不依赖目标标签、专家或策略动作，也不添加机器人关节。录制与策略共用物理评测，
专家目标仅用于生成动作。

固定 prompt：

> Observe which box contains the can during the one-second viewing period. After both boxes close, move to the front, pick up the block, and place it in the tray in front of the remembered box.

```bash
source scripts/tools/activate_sim_env.sh
python scripts/record_entrypoint.py --task-id WR12 --seeds 0,1 --dataset-id wr12_smoke --output-root outputs/datasets --no-record-depth
python -m pytest -q tests/test_wr12_task.py tests/test_wr12_evaluation.py tests/simulator/test_wr12_visibility.py
```

评测 suite：`configs/eval/suites/wr12_v1.yaml`。任务版本：`1.4.0`；
采样 profile：`WR-12/sampling-v2`。旧版轨迹不能直接作为当前版本的数据。
