# WR12 pi0.5 checkpoint 29999

模型：`outputs/pi05-ckpts/wr12/29999`，配置：
`pi05_membench_wr12_3view_full_stride10_30k`。
归一化从 checkpoint 的 `assets/wr12-final/norm_stats.json` 加载。
三路 224×224 RGB、state37 输入；训练变换保留前 30 维状态；输出 50×13 动作。

评测使用 WR12 v1.4、`WR-12/report-v2`，报告抓取、放置、成功。
正式 seeds 1050–1099，每条最多 2400 步，20 Hz，每 50 步重新推理一次，
成功或任务终止时提前结束，保存三路视频。不使用专家前缀。
每个 episode 按 seed 重置策略随机数。

仓库根目录启动，默认推理 GPU 0、EGL 渲染设备 1、端口 59212：

```bash
bash scripts/eval/pi05/eval_pi05_wr12.sh serve
tail -f outputs/eval/logs/serve_wr12_59212.log
```

等日志出现 `server listening` 后退出 tail，运行：

```bash
bash scripts/eval/pi05/eval_pi05_wr12.sh preflight
bash scripts/eval/pi05/eval_pi05_wr12.sh smoke
bash scripts/eval/pi05/eval_pi05_wr12.sh eval
```

可用 `GPU`、`RENDER_GPU`、`PORT`、`CKPT_DIR`、`RUN_ID` 覆盖默认设置。
需要前台服务时使用 `serve-fg`。相同 run ID 再次运行会续评已提交结果；
基础设施错误条目需设置 `RETRY_ERRORS=1` 才会重试。

```bash
bash scripts/eval/pi05/eval_pi05_wr12.sh status
bash scripts/eval/pi05/eval_pi05_wr12.sh summary
bash scripts/eval/pi05/eval_pi05_wr12.sh stop
```

2026-11-15 已完成模型加载和在线预检。独立 seed 1049 试跑完成 2400 步、
48 次推理，无运行错误；三路视频各 2400 帧、20 FPS，已抽帧检查。
观察与记忆屏障通过，但未抓起方块，最终达到步数上限；此条不计入正式结果。

正式评测于 2026-11-15 08:25 UTC 后台启动。
run ID：`wr12-pi05-3view-full-stride10-s29999-r50-seedrng-v1`。
启动记录：`outputs/eval/logs/<run-id>.launch.json`；日志：
`outputs/eval/logs/<run-id>.log`；产物：`outputs/eval/<run-id>/`。
`summary.json` 在每条 episode 提交后自动更新；最终结果以完成 50 条后的汇总为准。

## 最终结果

50 条已全部完成，无运行错误、无缺失或重复 seed，汇总 `publishable=true`。

| 模型 | 抓取 | 放置 | 成功 |
| --- | --- | --- | --- |
| pi0.5 29999 | 38/50（76%） | 33/50（66%） | 12/50（24%） |
| ACT 40000 | 31/50（62%） | 17/50（34%） | 7/50（14%） |
| DP 40000 | 21/50（42%） | 18/50（36%） | 7/50（14%） |

pi0.5 的失败为 21 条放错盒、17 条达到 2400 步上限。
最终成功率比 ACT、DP 各高 10 个百分点；各模型使用同一组 50 个 seed。
放置阶段只要求完成物理放置，正确目标由成功阶段判定。

## 代码依赖与验证

OpenPI 训练配置已包含在 `SiHhangChen/openpi` 的 `99e7d7d` 提交中。
本次评测使用 `Policy.reseed()` 在每条 episode 开始时重置 JAX 随机数；
配套改动见该仓库的 `wr12-eval-reseed-20261115` 分支。
复现实验时需要此方法，单独重置 NumPy / Torch 不会重置 JAX 策略采样。

可移植的完整汇总随代码保存于：

- `docs/results/wr12/pi05_29999_summary.json`
- `docs/results/wr12/act_40000_summary.json`
- `docs/results/wr12/dp_40000_summary.json`

本次独立检出仅加入 WR12 及其必要依赖。选择的测试共 174 项，170 项通过；
4 项失败已在未修改的远程基线 `9f44b9d` 上复现：既有任务包文件不全、
既有任务互相导入、报告目录测试缺少 WA-06、WR11 测试仍预期旧的 3600 步。
WR12 的任务、物理判分、MuJoCo 遮挡与 ACT/DP 配置测试全部通过。
