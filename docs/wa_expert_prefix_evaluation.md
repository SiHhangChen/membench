# WA01 / WA05 正式评测的固定专家前置段

正式评测 v2 每个 episode 只交接一次，之后由 policy 连续完成所有剩余动作。

| 任务 | 专家完成 | 交接状态 | Policy 完成 |
| --- | --- | --- | --- |
| WA01 | `add_sugar`，共 1 个子任务 | 糖已投到 seed 指定杯中，机械臂完成退让和复位 | 运送两杯、使两杯就位、选择加糖杯 |
| WA05 | `pick_cup`、`fill_cup`，共 2 个子任务 | 已接水并完成接水后的机械臂回收与离开水槽的抬杯，仍真实夹持杯子 | 放杯、按与所接水温一致的按钮 |

专家执行时，policy 按 suite 的 `replan_steps` 接收当前观测并进行推理，其动作被丢弃。整个 episode 只在开始时 reset 一次模型；交接时清空未执行的动作块，使用交接观测重新推理，保留模型历史和完整任务指令。杯子身份、水温等答案只保存在评测器及结果文件中，不加入策略输入。

WA01 的首次真实加糖事件锁存答案。WA05 的首次真实接水事件锁存答案；之后接另一种水会触发原有的 `mixed_temperature_fill` 失败，不能改写答案。WA05 交接后的放置仍使用正式 policy 评测的岛台放杯规则。

## 配置与运行

标准 suite 为 `configs/eval/suites/wa01_v2.yaml` 和 `configs/eval/suites/wa05_v2.yaml`，正式 seeds 为 1050–1099。GR00T、pi0.5、MemER、ACT 和 diffusion 的 WA01/WA05 主评测入口已指向相应的 v2 suite；默认 run ID 使用 `prefix-v2` 区分旧结果。

先在仿真环境中为正式种子生成固定动作文件。每个任务的所有模型复用同一份前置段：

```bash
source scripts/tools/activate_sim_env.sh
python scripts/eval/build_evaluation_prefixes.py --suite configs/eval/suites/wa01_v2.yaml
python scripts/eval/build_evaluation_prefixes.py --suite configs/eval/suites/wa05_v2.yaml
```

生成单个调试 seed 时加 `--seed 1049`；生成器复用已通过身份与内容校验的文件。前置段存放于 `outputs/evaluation_prefixes/wa01_sugar_v1` 和 `outputs/evaluation_prefixes/wa05_pick_fill_v1`，不使用旧的逐阶段前置段目录。

随后使用原有模型 launcher 的 `eval` 命令，或直接运行：

```bash
python scripts/eval/run_remote.py --endpoint ws://127.0.0.1:8000 \
  --suite configs/eval/suites/wa01_v2.yaml --run-id wa01-model-prefix-v2
```

远程评测在连接模型前校验 suite 全部 seeds 的前置段；缺文件、错误 episode 身份、错误 profile、损坏动作或错误边界会阻止启动。Run manifest 保存整套动作的指纹，修改动作后不能续跑到同一结果目录。真实回放还会在交接时检查物理状态；前置段无效记为 `error`，阻止 policy 接管，结果不能作为完整评测发布。

## 计分与回放

`rollout.max_steps` 只计算交接后的 policy 环境步数。`env_steps` 和 `success_step` 包含专家段；`policy_actions_consumed` 仅计算真正执行的 policy 动作。`policy_calls` 包含专家期间为保留历史而进行的推理。

每个结果的 `episode_metadata.expert_prefix` 记录执行步数、交接步、动作 SHA256、有效性和内部答案。阶段报告沿用物理事件与权威评测器，前置阶段的通过由专家完成；v2 成功率表示固定专家前置条件下剩余任务的成功率。

保存的动作和视频覆盖整个 episode，包括专家段。动作回放不会再次插入专家前置段。

旧 v1 suite 与逐阶段诊断保留。Launcher 的 `stage-*` / `full-stage` / 旧 `prefixes` 命令继续使用 v1；需要自定义旧结果目录时设置 `MAIN_ROLLOUT_RUN`。逐阶段诊断拒绝把 v2 专家完成的阶段导入为模型阶段成绩。

## 当前验证

已为 WA01、WA05 各生成并在全新正式评测环境中回放 seeds 1049、1050，覆盖两只加糖杯以及冷、热水分支。四次都通过物理交接检查，模型仅 reset 一次，交接后正常执行 1 步策略动作并耗尽测试预算。这验证了前置段与交接，不代表策略完成任务的成功率。正式 50-seed 全量运行前仍需用上面的生成命令补齐剩余前置段。
