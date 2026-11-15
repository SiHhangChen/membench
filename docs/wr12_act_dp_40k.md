# WR12 ACT / DP 40k 训练

数据为 `data/final/wr12-final`，200 episodes、131,328 帧，WR12 v1.4 观察一秒后封口规则。
配置沿用 TM05 / TS02 / WR11 40k 训练：batch 16、stride 10、训练 40,000 steps，
每 10,000 步保存，seed 1000、8 workers、prefetch 4、spawn、无 AMP、无图像增强。
`training_trim_video_tails=false`，无验证集与仿真评测，关闭 Hub/WandB 上传。

| 模型 | GPU | PID | 关键配置 |
| --- | --- | --- | --- |
| ACT | 6 | 1663936 | chunk/action steps 50，AdamW lr 1e-5，KL weight 10，TorchCodec |
| DP | 7 | 1663943 | obs 2，horizon 64，action steps 32，Adam lr 1e-4，cosine warmup 500，PyAV，共享 RGB encoder，gradient checkpointing |

2026-11-12 17:56 UTC 从第 0 步后台启动；使用
`/home/chensihang/miniconda3/envs/membench/bin/python` 与 `policy/lerobot/src`。
启动记录 `outputs/train/wr12_40k_v1/launch.json`，数据工厂预检结果
`outputs/train/wr12_40k_v1/preflight.json`：两配置解析通过，首/中/末帧三路图像解码与
有限值检查通过，动作形状分别 50×13、64×13，观测长度 1、2。

配置：`configs/train/wr12_40k_v1/act_wr12.json` 与 `diffusion_wr12.json`。
单任务入口 `scripts/train/train_wr12_40k.sh`，已有输出会拒绝覆盖。

日志：

```bash
tail -f outputs/train/act_wr12_40k_v1/wr12.log
tail -f outputs/train/diffusion_wr12_40k_v1/wr12.log
```

checkpoint 分别写入：

```text
outputs/train/act_wr12_40k_v1/wr12/checkpoints/040000/pretrained_model/
outputs/train/diffusion_wr12_40k_v1/wr12/checkpoints/040000/pretrained_model/
```

两个 40000-step checkpoint 均已完成训练并通过数据/模型接口静态检查。

2026-11-13 评测使用 seeds 1050–1099，各 50 条，每条最多 2400 步，录制三路视频。
报告 profile 为 `WR-12/report-v2`，只显示抓取、放置、成功；观察与遮挡记忆条件仍参与成功判定。
ACT 执行 50 步动作块；DP 每步接收新观测，模型内部沿用 2 帧观测和 32 步动作缓存。
两者均不使用专家前缀。

入口为 `scripts/eval/eval_wr12_40k.sh act|diffusion GPU PORT RUN_ID`。
后台资源队列 `scripts/eval/queue_wr12_40k.py` 每 30 秒检查 GPU，选取空闲显存至少
16000 MiB 且利用率不超过 10% 的卡，依次评测 ACT、DP，不中断已有训练。
队列记录在 `outputs/eval/logs/wr12-40k-queue-20261113/status.json`，
各模型日志与视频结果路径可按其中 `run_id` 定位。

按用户要求，已取消等待队列，直接启动共享资源评测。
ACT 使用 GPU 6 推理、GPU 4 渲染，run ID 为
`act-wr12-40000-stages-v2-shared-20261113`。
DP 在 GPU 7 加载成功，但首个动作推理因剩余显存不足而 OOM；该次运行已停止，
其基础设施错误结果不作为模型表现。DP 改用 CPU 推理、GPU 5 渲染，重新从 seed 1050
开始。CPU 首次尝试超过原先 30 秒通信超时，调整为 4 个推理线程与 300 秒
通信/推理超时后再次重启，正式 run ID 为
`diffusion-wr12-40000-stages-v2-cpu-v2-20261113`。
任务、seed、动作执行与成功条件不变，仅放宽通信超时；此次配置保存在对应日志目录的 `suite.yaml`。
CPU 使用同一 checkpoint 与评测配置，但推理速度与 GPU、随机数实现存在差别。
两者最新启动记录位于各自 `outputs/eval/logs/<run_id>/launch.json`。

用户暂停后，2026-11-13 06:48 UTC 空闲 GPU 已恢复，终止旧的暂停进程并重新启动。
ACT 改为 GPU 4 推理与渲染，保留原 run ID，续评已完成 6 条之外的 seed；
DP 改回 GPU 5 推理与渲染，使用标准 30 秒通信超时配置，从 seed 1050 开始，
run ID 为 `diffusion-wr12-40000-stages-v2-gpu-20261113`。
当前有效启动记录为 `outputs/eval/logs/wr12-40k-queue-20261113/gpu_restart.json`。

## 最终评测结果

两个模型均完成 seeds 1050–1099 的 50 条评测，无运行错误、无缺失结果。

| 模型 | 抓取 | 放置 | 成功 |
| --- | --- | --- | --- |
| ACT 40000 | 31/50（62%） | 17/50（34%） | 7/50（14%） |
| DP 40000 | 21/50（42%） | 18/50（36%） | 7/50（14%） |

ACT 失败为 33 条到达 2400 步上限、10 条放错箱子；DP 分别为 32 条、11 条。
放置阶段只要求完成物理放置，目标是否正确由成功阶段判定。
结果与视频位于 `outputs/eval/<run_id>/`，最终汇总为 `summary.json`。
