# WR-12 v1：遮挡后的记忆投放

这是初版 `1.0.0` 的历史归档。当前规则与验证见 [当前版本](wr12_validation.md)。

WR12 已实现并接入统一任务注册、专家录制、物理评测和动作回放。
机器人从侧面观察盒内罐头，转向并沿台边移动约 1.27 m 至操作位，
抓取公共方块后放入对应盒子前方托盘。整个任务只有一次抓取和一次放置。
场景为 layout 33/style 34/scene seed 0，版本为 `1.0.0`。

![两种目标的观察、遮挡、抓取、投放三视角](../outputs/diagnostics/wr12/wr12_preview.jpg)

## 场景与采样

两个固定遮挡盒为 34×40×30 cm，顶部及前后右侧封闭，左侧留观察开口。
开口前缘增加相同的 8 cm 挡边，阻断操作区腕部相机的斜视路径。
罐头位于盒中心向内 8 cm 的位置，由真实盒底支撑。两个托盘为 30×27 cm，
方块为 5×5×7 cm；盒子与托盘固定，方块与罐头为自由物体。

方块位置和专家放置点在半径 4 cm 的圆内按面积均匀采样，方块 yaw 在
`[-π, π]` 内均匀采样。两个底盘站位各在半径 5 cm 的圆内采样，
观察位 yaw 为 −0.65，操作位为 −π/2。站位和物体位置通过独立随机流生成。
运行中不重设物体姿态、不附着抓取、不关闭碰撞。

相邻 seeds `2k`、`2k+1` 共享方块位置、朝向、站位和放置扰动，只交换目标盒。
每对内的目标顺序随机；完整 seed 对确保左右各一次。训练、验证、测试按整对划分。
例如训练 seeds 0–199 对应 100 个布局，每个布局包含左右两种历史。
固定 prompt 与统一三路 RGB、state37、action13 不包含目标盒或专家阶段。

## 判分

公共阶段为 `can_observed → memory_barrier → block_picked → block_placed → success`。
观察要求连续三个控制步可见；遮挡要求到达操作区，全部相机证明罐头完全隐藏，
且至少一路视野中的罐头被盒体遮挡，连续三个控制步确认。

遮挡判断先检查整个罐头包围盒是否被同一面板挡住；多面板交界处使用封闭盒体和
侧开口的几何投影证明。仅罐头中心射线被挡不能构成完全遮挡。
配置加载验证盒体尺寸、不透明材质和碰撞几何，防止改资产后继续使用无效证明。

遮挡确认后重新看见罐头、或无法证明全部相机隐藏，会终止本条。
抓取必须夹爪闭合、真实持有并抬升至少 2 cm。首次脱离夹爪并落入托盘的释放
锁定选择；连续三个控制步真实支撑后通过放置阶段，然后检查目标盒。
放错也可记录抓取/放置通过，但成功失败；换盘不能恢复。移动罐头超过 1.5 cm 失败。

## 完整录制与回放

最终场景 seeds 0–7 和正式采集入口 seeds 1050、1051 共 **10 条录制全部成功**。
seeds 0–7 的数据位于 `outputs/diagnostics/wr12/record*_v*/episode/`；最终版本
分别为 seed 0 `record0_v2`、seed 1 `record1_v5`、其余 `record{seed}_v1`。
此前同目录中的其他版本属于调试记录，不用于数据集。

| seed | 目标 | 抓取通过步 | 成功步 | 录制帧数 |
| --- | --- | ---: | ---: | ---: |
| 0 | left | 407 | 488 | 591 |
| 1 | right | 407 | 501 | 617 |
| 2 | left | 429 | 518 | 648 |
| 3 | right | 429 | 524 | 660 |
| 4 | right | 411 | 510 | 655 |
| 5 | left | 411 | 508 | 651 |
| 6 | left | 441 | 523 | 659 |
| 7 | right | 441 | 536 | 684 |
| 1050 | right | 386 | 483 | 607 |
| 1051 | left | 386 | 474 | 590 |

20 Hz 下成功发生于约 24–27 秒。录制包含成功后的释放确认和收臂，因而帧数更多。
seeds 0、1 在全新策略运行时回放，无专家接管；成功步分别为 488、501，
所有公共阶段的首次通过步与录制完全一致，episode identity 一致。
回放产物在 `outputs/diagnostics/wr12/replay0_v2/replay/` 和 `replay1_v2/replay/`。

正式入口产物：`outputs/datasets/wr12_entrypoint_smoke_v1/`，含两条被接受的 episode
及 `manifest.json`。这是入口验证样本；正式 benchmark seeds 1050–1099 应保留作评测。

## 遮挡与数据验收

最终左右目标各检查两个站位圆边界的 8 个点，共 **32 个站位/目标组合通过**。
结果为 `scene0_v6/validation.json` 与 `scene1_v2/validation.json`。
成对 seeds 0、1 在操作位的三路 RGB **逐像素完全相同**；观察位的罐头位置不同。
整条物理评测持续检查遮挡，抓放运动未触发重新观察。

10 条录制的 30 段三路视频均逐帧解码通过；视频帧数、动作 `(N,13)`、状态 `(N,37)`
和元数据帧数一致，数值有限。审计：[recording_audit.json](../outputs/diagnostics/wr12/recording_audit.json)。
最终联合测试 **130 passed**，覆盖任务、采样、错误首选、遮挡证明、注册、接口、架构，
以及 2 项真实 MuJoCo 射线/盒体检查。Ruff 检查、格式检查、diff whitespace 和 suite preflight 通过。

更广泛的冻结测试存在原有的 TM04 缺失资源/跨任务导入、WA06 报告目录及旧 suite
命名/步数预期问题，未作为 WR12 验收通过项。为使环境注册可运行，补充了新增 TM05
环境缺失的 `typing.Any` 导入。

尚未采集正式 200 seeds 或训练 policy；10 条专家成功不能作为学习策略成功率。
站位边界检查仅验证场景可见性，不能证明整个随机域内所有抓放路径均可达。

## 使用

```bash
source scripts/tools/activate_sim_env.sh
export MUJOCO_EGL_DEVICE_ID=4

# 少量采集，--seeds 接受逗号或范围。
python scripts/record_entrypoint.py --task-id WR12 --seeds 0-9 \
  --dataset-id wr12_pilot_v1 --output-root outputs/datasets --no-record-depth

# 独立诊断录制，保留每阶段三视角图片。
python scripts/planning/validate_wr12.py --seed 8 \
  --output outputs/diagnostics/wr12/check_seed8

# 在全新策略运行时回放刚录制的原生动作。
python scripts/planning/validate_wr12.py --seed 8 \
  --replay-episode outputs/diagnostics/wr12/check_seed8/episode \
  --output outputs/diagnostics/wr12/replay_seed8

python scripts/evaluate.py --suite configs/eval/suites/wr12_v1.yaml --preflight-only
```

任务配置：[task.yaml](../bench/tasks/wr12/resources/task.yaml)。
评测 suite 为 `configs/eval/suites/wr12_v1.yaml`，上限 2400 步、每次推理执行 50 步。
本次保留改动与运行产物在工作区，未提交或推送。
