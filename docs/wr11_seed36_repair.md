# WR-11 seed 36：抓取规划修复

首轮 200 seeds 录制中，seed 36 的专家 stage 3 返回 `no_reachable_topdown_grasp_candidate`：四个方向都没有可执行的完整抓取链。原失败记录位于 `outputs/datasets/wr11_200seeds_v3_2_colors8_run2_part00/debug_failed/WR-11/seed_000036_attempt_00/`。

原策略使用 6 cm 预抓取、12 cm 抬升。保持同一场景诊断后，4 cm 预抓取、8 cm 抬升的 0°/180° 接近方向可通过预抓取、抓取、抬升的全部规划检查。诊断结果见 `outputs/diagnostics/wr11/seed36_grasp_probe.json`。

`bench/tasks/wr11/execution/geometry.py` 中增加通用后备策略 `WR-11/grasp-v2`：优先尝试原路径；只有所有原候选完整链都失败时，再规划较低高度候选，且仍优先原生 Y 轴的 0°/180° 方向。每个候选必须按顺序通过三个规划阶段，执行后仍需真实接触、持有与抬升证据。颜色、物体姿态、站位、物理评测与任务版本保持原样；专家路径有新的 profile 标识。

seed 36 的目标是右侧黄色牛奶盒，修复选择 180° 接近方向。正式重录使用独立 attempt 1，原失败 attempt 0 保留。新 episode 的 facts 和 identity 与原记录完全相同，identity 为 `sha256:06e6cb5b5664cf51e57fdd94e424575ab7d493a69a13d474d77e6eb41eb2c72a`。

## 验证

- 相关任务、物理评测、视野、依赖契约与新增后备路径测试：58 passed；最终完整链条件强化后，后备路径 3 项测试通过。Ruff、格式和 `git diff --check` 通过。
- 诊断完整录制全部 9 个 stage 成功，并在全新策略环境回放成功，成功步 1445。
- 正式 attempt 1 全部 9 个 stage 成功，总 1546 步，首次正确抓取步 389、物理成功步 1444。

正式重录目录为 `outputs/datasets/wr11_200seeds_v3_2_colors8_seed36_retry_graspv2/train/WR-11/seed_000036_attempt_01/`。正式回放位于 `outputs/diagnostics/wr11/seed36_repair/replay_production/`，最终审计写入 `outputs/diagnostics/wr11/seed36_repair/repair_audit.json`。

正式回放成功步为 1444，五个评测检查点与录制步数一致，episode identity 一致。三段正式视频逐帧解码通过，各 1546 帧，与动作和状态对齐。全部 yaw 指令为零，实测朝向变化最大 0.093°。

## 成功轨迹集合

后续整理已将全部原始文件集中到 `outputs/datasets/wr11_raw/`：正式成功
轨迹位于 `train/WR-11/`，seed 36 原失败轨迹位于 `debug_failed/WR-11/`。
旧分片和成功集合路径保留兼容链接；下述记录描述整理前的来源关系。

`outputs/datasets/wr11_200seeds_v3_2_colors8_complete/` 将原 199 条成功轨迹与 seed 36 成功重录通过目录链接组成一个 200 seeds 集合，保留各自源 episode 路径，不复制或改写原始产物。`manifest.json` 每个 seed 选取一条成功轨迹，`summary.json` 保留首次录制 199/200、一次 seed 36 重录的说明。

该集合是一次修复重录后得到的 200 条成功轨迹，首轮成功率仍为 99.5%。本次验证两个 seed 36 回放，不代表全数据集回放已完成。

[正式重录视频](../outputs/datasets/wr11_200seeds_v3_2_colors8_seed36_retry_graspv2/train/WR-11/seed_000036_attempt_01/videos/robot0_agentview_left.mp4)
