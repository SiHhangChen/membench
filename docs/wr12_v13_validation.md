# WR-12 v1.3 历史记录：固定开口版本

此历史记录对应 `1.3.0`，采样 profile 为 `WR-12/sampling-v2`。
仍从同一侧观察两个朝左开口的箱子，转向并沿岛台移动约 1.27 m 到正面，
抓取公共方块并放入记住的箱子前方托盘。机械操作只有一次抓取、一次放置。
场景为 layout 33/style 34/scene seed 0。

![修正前、箱体错位后、移动后的遮挡；上为左目标，下为右目标](../outputs/diagnostics/wr12/clear_view_v13b/observation_comparison.jpg)

200 seeds 正式录制后为 198 成功、2 失败，seeds 70、72 在抓取时未能保持完整遮挡。
此版本已由定时封口设计替代；旧记录单独保留。

## 观察增强

罐头使用原资产的 **3 倍尺寸**，实际包围盒约为直径 **18.2 cm**、高 **24.3 cm**。
v1.2 虽放大了罐头，但两箱同深度排列，左箱会挡住右箱开口；右目标在原观察图中
仅露出约三成轮廓。此前仅检查可见像素数量，没有充分检查物体露出的比例。

本版将左箱后移 **15 cm**：箱心分别为左 `(3.00, −2.60)`、右 `(3.65, −2.45)` m。
罐头相对对应箱心的 XY 偏移分别为左 `(−0.05, −0.09)`、右 `(0.06, −0.09)` m，
均由盒底真实支撑。箱子保持 34×40×30 cm，顶部和前后右侧封闭、左侧开口；
开口前缘挡边加宽至 **12 cm**，阻断操作时腕部相机的斜视路径。
两个托盘为 30×27 cm，公共方块为 5×5×7 cm。
专家在观察阶段停留 20 个控制步，即 1 秒。

两个箱子始终使用同一朝向、相同资产，布置与目标标签无关。机器人仍从同一侧观察，
再移至正面抓取公共方块并选择托盘。
盒子与托盘固定，方块与罐头使用自由关节；初始化之后不写物体位置、不附着抓取。

## 与已有任务对齐的随机化

| 变量 | 当前规则 | 对齐情况 |
| --- | --- | --- |
| 方块 XY | 台面中心 `(3.32, −2.02)` 附近，半径 4.5 cm 圆内面积均匀 | 与 WR07/WR08 范围及分布一致 |
| 方块 yaw | `[−π, π]` 均匀 | 与 WR07/WR08 一致 |
| 专家放置 XY | 对应托盘中心附近，独立半径 4.5 cm 圆内面积均匀 | 与 WR07/WR08 一致 |
| 专家放置方向 | 固定顺序尝试 `0°/90°/180°/270°`，选可行路径 | 与 WR07/WR08 当前执行规则一致；不保证四方向等频 |
| 观察站位 XY | `(2.05, −1.62)` 附近半径 5 cm 圆内面积均匀 | 与 WR07/WR08 范围一致 |
| 操作站位 XY | `(3.32, −1.62)` 附近半径 5 cm 圆内面积均匀 | 同上 |
| 站位 yaw | 观察中心 `−0.62`、操作中心 `−π/2`，各独立扰动 ±5° | 与 WR07/WR08 扰动范围一致 |
| 固定部分 | 场景、盒子与托盘中心、罐头尺寸及盒内相对位置、躯干高度 | 保留已校准遮挡关系 |

左右托盘中心分别为 `(3.00, −2.10)`、`(3.65, −2.10)`，单位为米。
圆采样使用 `r = R√u`，保证面积均匀；物体、放置点、站位 XY 与 yaw 采用独立随机流。
全部采样在 episode 解析时确定，导航重复调用复用该 episode 的点，不重新抽取。
实际物理导航允许 2.5 cm 停止误差；采样圆描述目标分布。

每个 seed 独立采样姿态和站位。相邻 `2k/2k+1` 只保证左右目标各一次，顺序随机，
不再复用方块姿态或站位。因此 seeds 0–199 为 **200 组独立姿态**，左右各 100 条。
初版 v1.0 为 100 组姿态，每组左右各一次，现已更改。
固定 prompt 与三路 RGB、state37、action13 不包含目标盒或 seed。

采样参数来源：`bench/tasks/wr12/resources/sampling/regions.json`；
站位参数来源：`bench/tasks/wr12/resources/calibration/stations.json`；
罐头尺寸来源：`bench/tasks/wr12/resources/scene/placements.json` 的 `can_scale`。

seeds 0–9999 的解析采样审计中，10000 组姿态均不同，左右各 5000 次，全部点位于规定
圆内；归一化半径平方均值为 0.495–0.500，面积均匀理论值为 0.5。
CSV、统计与图位于 `outputs/diagnostics/wr12/sampling_v2/draws200/` 和 `draws10000/`。
这些是解析分布检查，不能当作物理轨迹成功率。

![200 seeds 的物体、放置点与站位分布](../outputs/diagnostics/wr12/sampling_v2/draws200/sampling_support.png)

## 验收结果

最终产物目录：`outputs/diagnostics/wr12/clear_view_v13b/`。其他 `clear_view_*`、
`box_stagger_probe/` 及 `larger_can_*` 目录保留几何校准和历史版本，不用于当前数据集。

左右目标各检查两个站位、八个圆边界点、三种 yaw（−5°/0°/+5°），
共 **96 个组合通过**。操作位要求三路几何证明完全隐藏，且至少一路因盒壁遮挡。
观察位要求至少一路 agent 相机同时满足：罐头可见面积 ≥200 像素、可见轮廓占比
≥65%、宽 ≥10 像素、高 ≥18 像素。占比分母来自同一相机下仅保留罐头的诊断渲染；
渲染后恢复全部几何分组，不影响 RGB 录制和物理状态。遮挡几何证明与箱沿资产同步更新，
仍要求整个罐头包围盒被遮住；没有以“射线未命中”代替完整隐藏证明。

| 目标 | 观察边界上最佳 agent 相机可见面积 | 可见轮廓占比 | 最小可见宽 | 最小可见高 |
| --- | ---: | ---: | ---: | ---: |
| 左盒 | 372–480 像素 | 72.7%–82.4% | 17 像素 | 26 像素 |
| 右盒 | 225–258 像素 | 68.9%–77.6% | 13 像素 | 19 像素 |

上表使用原始 **224×224** 图像，检查每个边界点至少一路 agent 相机可见。
几何隐藏证明与 MuJoCo 渲染像素使用不同判据；不声称两个目标的 RGB 逐像素相同。
修正前后及移动后图像使用同一路原始相机，展示时以最近邻放大，见本文首图。
边界检查是场景证据，不能证明连续圆域中的每条抓放路径均可达。

| seed | 目标 | 抓取通过步 | 成功步 | 录制帧数 |
| --- | --- | ---: | ---: | ---: |
| 0 | left | 444 | 524 | 627 |
| 1 | right | 458 | 552 | 686 |
| 34 | right | 413 | 504 | 614 |
| 90 | left | 371 | 462 | 574 |

四条录制全部成功；seeds 34、90 的方块和放置点都接近 4.5 cm 圆边缘。
全过程逐控制步检查遮挡。所有 12 段视频逐帧解码通过，帧数与动作 `(N,13)`、
状态 `(N,37)` 及元数据一致，数值有限。

seeds 0、1 在全新策略运行时进行原生动作回放，均成功；episode identity 和所有阶段
首次通过步与录制完全一致。审计见
[recording_audit.json](../outputs/diagnostics/wr12/clear_view_v13b/recording_audit.json)。

最终相关联合测试 **132 passed**，包括真实 MuJoCo 射线测试；Ruff、格式、
diff whitespace 及评测 suite preflight 通过。尚未开展正式 200 seeds 采集或 policy 训练。

公共判分阶段保持 `can_observed → memory_barrier → block_picked → block_placed → success`。
观察、遮挡、真实托盘支撑各需连续三个控制步；遮挡后重新观察或可见性未知失败。
首次释放锁定选择，错误托盘失败；移动罐头超过 1.5 cm 失败。

## 复现

```bash
source scripts/tools/activate_sim_env.sh
export MUJOCO_EGL_DEVICE_ID=4

python scripts/planning/validate_wr12.py --seed 0 --scene-only --sampling-boundaries \
  --output outputs/diagnostics/wr12/new_scene0
python scripts/planning/validate_wr12.py --seed 0 \
  --output outputs/diagnostics/wr12/new_record0
python scripts/planning/validate_wr12.py --seed 0 \
  --replay-episode outputs/diagnostics/wr12/new_record0/episode \
  --output outputs/diagnostics/wr12/new_replay0

python scripts/record_entrypoint.py --task-id WR12 --seeds 0-9 \
  --dataset-id wr12_v13_pilot --output-root outputs/datasets --no-record-depth
python scripts/evaluate.py --suite configs/eval/suites/wr12_v1.yaml --preflight-only
```

`--scene-only` 使用站位重设进行诊断，不能作为成功轨迹。
历史记录：[v1.0](wr12_v1_validation.md)、[v1.2（右箱观察不足）](wr12_v12_validation.md)。
旧版与当前版本的 episode identity 不同，不能混作当前版本的数据。
