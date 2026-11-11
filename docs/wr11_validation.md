# WR-11 v3.2：八色包装验证

当前任务版本 **3.2.0**，采样 profile 为 `WR-11/sampling-v2`，场景为 layout 33/style 34/scene seed 0。包装池为红、橙、黄、绿、青、蓝、紫、黑；每次从八色中抽取三种不同色候选，再用独立请求随机流等概率选择其中一种作为样品，保证恰好一个匹配。

物体圆半径仍为 2.5 cm、物体 yaw 为 `[−π,π]`；两处站位圆半径仍为 5 cm，机器人保持 `−π/2` 朝向沿岛台平移。采样细节和复现命令见 [正式采样规则](wr11_sampling.md)。

## 颜色与采样

在同一姿态、光照和相机下逐色渲染，八种包装颜色清楚可分，保留原有白色图案。几何、碰撞与光泽参数一致。

![八色实际渲染对照](../outputs/diagnostics/wr11/colors8/scene_seed0/palette_comparison.png)

`--palette-preview --scene-only` 只改变诊断环境里的样品纹理，不录制训练轨迹。三路机器人相机截图也保存在 `outputs/diagnostics/wr11/colors8/scene_seed0/`。

10000 个解析种子 `0–9999` 覆盖全部八色、336 种有序候选排列和 1008 种样品与候选排列组合。红/橙/黄/绿/青/蓝/紫/黑样品计数为 `1228/1255/1287/1204/1267/1276/1242/1241`，目标位于左/中/右的次数为 `3218/3423/3359`。全部物体和站位点在指定圆内，归一化半径平方均值为 `0.495–0.502`，理论值为 `0.5`。CSV、图和统计位于 `outputs/diagnostics/wr11/colors8/draws10000/`。

八色版本 seed 0 两处站位圆各八个边界点，共 16 点通过样品可见/不可见、全部候选可见与托盘真实接触检查。解析审计不运行物理轨迹，边界检查不执行抓取，不能据此推断完整圆域或全部种子的物理成功率。

## 完整专家录制

使用生产 `RecordingRunner` 完成两条八色版本轨迹，均被接受。

| seed | 样品 | 正确候选 | 首次正确抓取步 | 成功步 | 录制总步数 | 最大朝向变化 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 0 | 蓝 | 中 | 334 | 1350 | 1447 | 0.088° |
| 4 | 绿 | 右 | 355 | 1374 | 1467 | 0.087° |

两条轨迹全部 yaw 动作指令为零，状态实测朝向变化小于 0.1°。样品观察和返回目标 XY 相同，与解析站位一致；到达取货区时样品完整离开全部三路相机视锥。首次正确抓取和释放到送回托盘均通过真实物理判定。

产物位于 `outputs/diagnostics/wr11/colors8/record_seed{seed}/`，含 `episode/trajectory.h5`、三路视频、录制元数据、专家阶段结果和 `validation.json`。六段视频逐帧解码通过，帧数、动作 `(N,13)`、状态 `(N,37)` 与元数据步数一致，动作和状态数值均有限。审计见 `outputs/diagnostics/wr11/colors8/recording_audit.json`。

- [seed 0 左相机完整视频](../outputs/diagnostics/wr11/colors8/record_seed0/episode/videos/robot0_agentview_left.mp4)
- [seed 4 左相机完整视频](../outputs/diagnostics/wr11/colors8/record_seed4/episode/videos/robot0_agentview_left.mp4)

## 全新策略环境动作回放

通过 `scripts/eval/replay_recorded_actions.py` 在全新策略环境回放，不调用专家控制器。结果位于 `outputs/diagnostics/wr11/colors8/replay_seed{seed}/replay/`；汇总见 `outputs/diagnostics/wr11/colors8/validation_summary.json`。

两条回放均成功，episode identity 及五个公共阶段通过步数与录制一致，成功后停止执行。

| seed | 回放成功步 | 五阶段一致 | episode identity 一致 |
| --- | ---: | --- | --- |
| 0 | 1350 | 是 | 是 |
| 4 | 1374 | 是 | 是 |

## 自动检查与范围

相关采样、任务、契约、物理评测、相机视野、注册、回放和停止执行测试 **229 passed**。任务与验证/采样分析脚本 Ruff 检查、格式检查、任务包与配置检查、正式 suite preflight 和 `git diff --check` 通过。

本次未运行完整 50 种子 rollout 或训练策略评测。任务上限为 3600 步，正式 suite 为 `configs/eval/suites/wr11_v3.yaml`。八种颜色的渲染和解析覆盖均已检查，完整物理轨迹只验证蓝、绿两种样品。

历史版本：[v3.1 三色正式采样](wr11_v3_1_sampling_validation.md)、[v3.0 平移初版](wr11_v3_translation_validation.md)、[v2 转身版本](wr11_v2_validation.md)、[v1 旧房间](wr11_v1_validation.md)。八色版本改变了相同 seed 的颜色及 episode identity，旧录制不能作为 v3.2 结果。
