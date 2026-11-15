# WR12 v1.4：完整重录 200 seeds

当前批次：`wr12_200seeds_v14`，开始时间 `2026-11-12T17:06:58.893291+00:00`，supervisor PID `3315618`。
任务版本 `1.4.0`，采样 profile `WR-12/sampling-v2`。

本批次从头录制 **seeds 0–199**，左右目标各 100 条；`resume_from` 为空，
`inherited_items` 为 0。每个 seed 都重新生成物理轨迹，旧版或诊断轨迹不计入本批次。

开局开放观察 20 个控制步（1 秒），随后两个箱子的挡板同步落下 6 步（0.3 秒），
第 26 步完全封闭并保持锁定。确认遮挡连续三步后，机器人移动到正面抓公共方块，
放入记住的箱子前方托盘。机器人无需关门或从柜内取物。

- 并发：16 个录制进程；GPU 4、5、6、7 各 4 个，外加 1 个 supervisor。
- 相机：`robot0_agentview_left`、`robot0_agentview_right`、`robot0_eye_in_hand`。
- 图像：224×224、20 FPS，仅保存 RGB；state 37 维，action 13 维。
- 方块与放置点：独立半径 4.5 cm 圆内面积均匀；站位：半径 5 cm，yaw ±5°。
- 每条轨迹前检查任务源码哈希与 episode identity；记录独立分片、增量 manifest，失败产物保留。
- 正式运行期间冻结 `bench/tasks/wr12/`，避免混合版本；任务源码与录制/审计脚本已快照。

运行目录：`outputs/record/wr12_200seeds_v14/`。
Supervisor 日志：`outputs/record/wr12_200seeds_v14_supervisor.log`。
分片：`outputs/datasets/wr12_200seeds_v14_part00/` 至 `part15/`。

进度以 `partXX_progress.json` 与各分片 `manifest.json` 为准。
所有进程结束后，supervisor 写入 `summary.json`；全部 seed 成功时自动执行完整审计：
版本、prompt、identity、阶段顺序、封口状态、三路完整隐藏、HDF5 有限数与维度、
逐帧视频解码及帧数。审计通过才生成 `native_audit.json` 和
`outputs/datasets/wr12_200seeds_v14_complete/`。

启动命令（已执行，重复执行会拒绝覆盖当前批次）：

```bash
RUN_ID=wr12_200seeds_v14 \
RENDER_GPUS=4,5,6,7,4,5,6,7,4,5,6,7,4,5,6,7 \
RESUME_FROM= bash scripts/record/launch_wr12_200seeds.sh
```

旧 v1.3 全部数据和日志保留，198 成功、2 失败；不会混入 v1.4。
旧录制记录见 [v1.3 记录](wr12_v13_recording.md)，当前设计验收见
[WR12 验证](wr12_validation.md)。本流程录制原生数据，不自动训练或上传。

## 补录与最终选择

原始 seeds 0–199 全部完成，199 成功，seed 70 在放置阶段机械臂碰箱后失败，原始失败产物保留。
按用户要求，另外录制 seed 200 成功，不重录 seed 70；新 seed 的物体、站位及放置点按现有规则采样。
任务源码未修改，所有轨迹保持同一版本和源文件哈希。

最终选择为 seeds 0–69、71–200，共 200 条成功轨迹，左右各 100 条。
选择清单位于 `outputs/record/wr12_200success_v14/`；补录原始目录为
`outputs/record/wr12_v14_supplement_seed200/`。
最终逐帧审计结果以选择目录中的 `native_audit.json` 为准；通过后数据入口为
`outputs/datasets/wr12_200success_v14_complete/`。
