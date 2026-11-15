# WR12 v1.3 数据录制

录制固定 WR12 `1.3.0`、`WR-12/sampling-v2`，沿用同侧观察后移动到正面、抓取方块并投放的流程。

目标为 seeds 0–199，各 seed 独立采样姿态，左右目标各 100 条。
先导录制 `wr12_v13_pilot8` 使用 seeds 0–7；通过原生产入口及逐帧审计后，
正式运行 `wr12_200seeds_v13` 继承这 8 条并续录剩余 192 条。

- 相机：`robot0_agentview_left`、`robot0_agentview_right`、`robot0_eye_in_hand`。
- 图像：224×224、20 FPS；保存 RGB 视频，不保存深度。
- 动作：13 维；状态：37 维。运行时观测 profile 为共享 `rgbd_v3`。
- 当前并发：16 个独立录制进程，EGL 设备 4、5、6、7 各 4 个；失败产物保留在 debug_failed。
- 每条轨迹前检查任务源文件哈希和 episode identity，保留源码快照与增量 manifest。

原 8 进程运行目录：`outputs/record/wr12_200seeds_v13/`。
扩容后当前运行目录：`outputs/record/wr12_200seeds_v13_16workers/`。
新增分片目录：`outputs/datasets/wr12_200seeds_v13_16workers_part00/` 至 `part15/`。
扩容时保留已完成的 160 条记录，将未完成 40 个 seed 分配给 16 个进程；
旧分片中完成记录继续作为实际来源，未完成文件保留。
完成后自动逐帧解码视频，核对动作、状态、版本、prompt、阶段判分、目标分布。
仅当全部通过时生成 `outputs/datasets/<run_id>_complete/`，
其 train 目录链接到已校验的原始轨迹，manifest 记录实际来源。

```bash
source scripts/tools/activate_sim_env.sh
python scripts/record/record_wr12_200seeds.py --dry-run
RESUME_FROM=outputs/record/wr12_v13_pilot8 bash scripts/record/launch_wr12_200seeds.sh
```

此流程只录制原生数据；不自动转换 LeRobot、训练 policy 或上传。
先导录制已完成：8/8 成功，左右各 4 条，共 5,413 帧，24 段视频逐帧解码及
动作/状态/identity/prompt/判分检查通过。

当前扩容运行 supervisor PID 为 1349497，旧 PID 528102 已停止。
扩容时完成记录中 158 条成功，seeds 70、72 因 `can_reobserved_after_barrier` 失败，
尚需分析；它们不会计入成功集合，当前 200 条全通过自动审计不会被触发。
实际完成数量和验收结果以运行目录中的 summary.json、native_audit.json 为准。
