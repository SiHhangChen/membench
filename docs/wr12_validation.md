# WR12 v1.4：观察一秒后实体封口

任务版本 `1.4.0`，任务 revision `WR12-v1.4-one-second-view-then-seal`。
保留同侧观察、移动到正面、抓公共方块、投放对应托盘的操作流程。
开局观察 20 步（1 秒），两个开口随后同步关闭 6 步（0.3 秒），第 26 步完全封口，
之后不再开放。自动封口由模拟时间驱动，与目标标签、专家和策略动作无关。

挡板使用不透明、带真实碰撞的 mocap 物体；盒子其他面原本封闭，落板后构成完整外壳。
机器人无需关门。方块和罐头仍使用自由关节与正常物理，不写入初始化后的位姿、不使用抓取附着。
旧版仅靠站位遮挡时仍有重观察机会，因此 v1.3 轨迹不混入新版本。

固定 prompt：

> Observe which box contains the can during the one-second viewing period. After both boxes close, move to the front, pick up the block, and place it in the tray in front of the remembered box.

## 判分

先在观察位看到罐头；两箱真实封闭、三路相机完整隐藏且存在盒壁遮挡连续三步，
建立 `memory_barrier`。屏障可在观察站位建立，专家随后移动到操作位。
屏障后箱口重新开放、再次看到罐头或无法证明完整隐藏均失败。
抓取还要求到达操作位、真实持有并抬升；首次释放锁定托盘选择，真实稳定支撑三步后判分。
公共阶段仅为 `block_picked → block_placed → success`（抓取、放置、成功），
使用 `WR-12/report-v2`。`can_observed` 与 `memory_barrier` 继续作为内部判分条件，
保留在 diagnostics 中。旧五阶段报告通过显式 `WR-12/report-v1` 读取。

## 随机化与观察

随机化继续对齐 WR07/WR08，不改变已校准的箱体几何：

| 变量 | 规则 |
| --- | --- |
| 方块 XY | 中心 `(3.32, −2.02)` m 附近半径 4.5 cm 圆内面积均匀 |
| 方块 yaw | `[−π, π]` 均匀 |
| 专家放置 XY | 对应托盘中心附近独立半径 4.5 cm 圆内面积均匀 |
| 观察与操作站位 | 各独立半径 5 cm 圆内面积均匀，yaw 独立 ±5° |
| 目标 | 每 seed 独立姿态，相邻 seed 平衡左右；0–199 左右各 100 |
| 固定部分 | 箱体、托盘、盒内罐头相对位置、场景和躯干高度 |

罐头仍为原资产 3 倍，直径约 18.2 cm、高 24.3 cm。
左箱后移 15 cm，保证右箱观察开口露出。观察验收阈值保持不变：至少一路 agent 相机
可见罐头 ≥200 像素、轮廓占比 ≥65%、宽 ≥10 像素、高 ≥18 像素。
采样参数和历史解析分布见 [v1.3 验证](wr12_v13_validation.md)。

## 验证证据

最终场景诊断位于 `outputs/diagnostics/wr12/timed_seal_v14b/scene_seed0/` 与 `scene_seed1/`。
两个目标各覆盖 72 个站位与朝向边界组合：开放观察位 24、封口后返回观察位 24、
封口操作位 24，共 **144 个组合通过**。每个目标的 `shutter_timeline.json` 检查前 20 步开放、
第 26 步封闭；`sampling_boundaries.json` 保存边界结果，`validation.json` 均为通过。

封口状态的分割图候选罐头像素还逐点使用真实相机射线确认，排除多重采样导致的 ID 假像素；
所有封口边界组合的确认像素为 0，并通过完整几何遮挡证明。
真实 MuJoCo 射线测试另覆盖封闭盒接缝和侧面等 5 种视点，每种 125 条射线均不命中罐头。

完整试录 `outputs/diagnostics/wr12/timed_seal_v14/record_seed1/validation.json` 通过：
观察第 3 步、屏障第 28 步、抓取第 447 步、成功第 541 步。
该试录属于诊断，不继承到正式批次。相关联合测试 **144 passed**，Ruff、格式和 whitespace 检查通过。
边界诊断不能替代完整 200-seed 的实际抓放结果；正式运行以 [录制记录](wr12_recording.md)
链接的增量进度、最终 summary 与完整审计为准。

历史：[v1.0](wr12_v1_validation.md)、[v1.2](wr12_v12_validation.md)、[v1.3](wr12_v13_validation.md)。
