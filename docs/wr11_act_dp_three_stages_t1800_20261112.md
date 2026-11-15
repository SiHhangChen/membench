# WR11 三阶段重测（2026-11-12）

用户指定只报告 item_picked → item_delivered → success，评测上限统一为 1800 步。

- item_picked：首次实际抓住并抬高一个库存纸盒至少 2 cm，不检查颜色。
- item_delivered：首次抓起的纸盒释放并与目标托盘接触连续 3 步，不检查颜色。
- success：放置通过，首次抓取纸盒颜色匹配参考，且原有观察、库存区视野隔离及参考未扰动等要求成立。

抓错颜色后继续运行，在放置通过后才以 wrong_color_delivered 结束；首次纸盒固定，额外抓起其他候选仍失败。
报告 profile 为 WR-11/report-v2；历史 report-v1 保留可读性。

同一批 ACT/DP 40k 模型，seeds 1050–1099 各 50 条，三视角视频开启，无专家前缀。
ACT GPU 4 / port 57140，DP GPU 5 / port 57141，2026-11-12 06:22 UTC 后台启动。
两组实际运行清单和 preflight 均确认 effective_rollout_max_steps=1800、report-v2。

相关检查：57 项测试通过；全任务 report catalog 测试因既有其他任务的目录/版本期望与当前代码不一致而排除。WR11 当前三阶段与历史 profile 单独验证通过，100 条旧 WR11 结果仍可解析。
最初启动的非 t1800 目录已停止，保留启动检查记录；本次只使用下列 t1800 目录。

- ACT 结果：`outputs/eval/act-wr11-40000-bs16-stride10-final20261107-stages-v2-t1800-20261112/`
- ACT 日志：`outputs/eval/logs/act-wr11-40000-bs16-stride10-final20261107-stages-v2-t1800-20261112/eval.log`
- DP 结果：`outputs/eval/diffusion-wr11-40000-bs16-stride10-final20261107-r1-seedrng-stages-v2-t1800-20261112/`
- DP 日志：`outputs/eval/logs/diffusion-wr11-40000-bs16-stride10-final20261107-r1-seedrng-stages-v2-t1800-20261112/eval.log`

## π0.5 同逻辑重测

2026-11-12 07:11:54 UTC 启动，GPU 6 / port 59312。沿用 WR11 30k checkpoint 和 `pi05_membench_wr11_3view_full_stride10_40k`，每局 RNG 按 seed 重置，seeds 1050–1099；1800 步、replan 50。已通过 preflight，正式评测有实际推理记录。

- 结果：`outputs/eval/wr11-pi05-3view-full-stride10-s30000-r50-t1800-seedrng-stages-v2-20261112/`
- 日志：`outputs/eval/pi05-wr11-stages-v2-t1800-20261112/wr11/formal.log`
- 运行状态：`outputs/eval/pi05-wr11-stages-v2-t1800-20261112/status.json`
- 启动记录：`outputs/eval/pi05-wr11-stages-v2-t1800-20261112/launch.json`

## 代码发布验证（2026-11-15）

三阶段判分、历史 report-v1 兼容、DP 1800 步上限、ACT/DP 运行名称与
配套测试整理在 `wr11-three-stage-evaluation-20261115` 分支。
独立检出验证 88 项通过，包括 WR11 判分、任务、MuJoCo 可见性、
报告格式与任务上下文；1 项全任务报告目录测试因已在远程基线复现的
WA-06 目录期望缺失而排除。两个启动脚本的 shell 语法检查通过。
