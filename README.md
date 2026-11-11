# MobileMemBench

MobileMemBench 是一个建立在 RoboCasa、robosuite 和 MPlib 之上的移动操作记忆基准。本仓库统一了任务定义、专家数据录制和闭环评测，并通过 WebSocket 将仿真客户端与策略服务端解耦。

本文档面向一台全新的 Linux 电脑，给出从克隆代码、安装仿真环境和资产，到录制数据、转换 LeRobot 数据集、启动策略服务和运行评测的完整流程。除非特别说明，所有命令都应在仓库根目录执行。

## 1. 复现范围

本仓库只保存基准代码、任务配置、测试和文档，不保存以下本地内容：

- RoboCasa 和 robosuite 源码及厨房资产；
- 专家轨迹、视频、HDF5 和 LeRobot 数据集；
- OpenPI、LeRobot、AC-DiT 等策略仓库；
- 模型 checkpoint、评测输出、缓存和日志。

推荐的本地目录结构如下：

```text
membench/
├── bench/                 # 任务、runtime、执行、录制、评测和协议
├── configs/               # observation profile、release lock、评测 suite
├── docs/                  # 架构与任务说明
├── scripts/               # 录制、转换、训练、服务和评测入口
├── tests/
├── robocasa/              # 单独克隆；被 Git 忽略
├── robosuite/             # 单独克隆；被 Git 忽略
├── policy/                # 可选策略源码；被 Git 忽略
├── outputs/               # 原生录制和评测结果；被 Git 忽略
└── lerobot_data/          # 转换后的数据集；建议只保存在本地
```

录制和评测共享同一套 canonical task/runtime。每个 episode 都由任务 ID、seed、variant、split 和固定配置解析得到；策略只接收三路 RGB-D、37 维机器人状态和 prompt，不接收任务 oracle。

## 2. 已验证平台与版本

推荐使用 Ubuntu 22.04 x86-64。无显示器服务器需要支持 EGL 的 NVIDIA 驱动；有桌面的电脑也可以使用 GLFW。

当前代码所依据的核心版本为：

| 组件 | 版本或提交 |
| --- | --- |
| Python | 3.11 |
| robosuite | `232ce7d4a6ed89c949a9aba024a05c8c32fdd08b`（运行时 1.5.2） |
| RoboCasa | `8f124f0e263de0c583ba5d2acc87137bfd905986`（运行时 1.0.1） |
| MuJoCo | 3.3.1 |
| NumPy | 2.2.5 |
| MPlib | 0.1.1 |
| LeRobot（RoboCasa 环境） | 0.3.3 |
| LeRobot（ACT 训练/服务环境） | `d4a229444b9a09742b4df5af079d308d1aa9a1d0`（0.5.2） |

新的实验如果更换这些版本，应同时记录第三方 Git commit、Conda 环境和本仓库 commit，不能只记录包名。

## 3. 准备新电脑

### 3.1 系统依赖

先安装 Git、编译工具、FFmpeg 和常用 OpenGL/EGL 运行库：

```bash
sudo apt update
sudo apt install -y \
  git git-lfs build-essential cmake ffmpeg \
  libgl1 libegl1 libglvnd0 libglib2.0-0
git lfs install
```

另外安装 Miniconda 或 Miniforge，并确认：

```bash
conda --version
```

在无显示器 GPU 服务器上还应确认驱动正常。`nvidia-smi` 必须能够列出 GPU；仅安装 CUDA toolkit 不能替代宿主机 NVIDIA 驱动。

### 3.2 克隆主仓库

私有仓库需要提前配置 GitHub SSH key：

```bash
git clone git@github.com:SiHhangChen/MomaMemBench.git membench
cd membench
git rev-parse HEAD
```

保存最后一条命令输出的 commit。不同机器复现实验时应 checkout 同一个 commit。

### 3.3 克隆固定的仿真依赖

代码会优先从仓库根目录下的 `robosuite/` 和 `robocasa/` 导入源码，因此目录名和位置不能改变：

```bash
git clone https://github.com/ARISE-Initiative/robosuite.git robosuite
git -C robosuite checkout 232ce7d4a6ed89c949a9aba024a05c8c32fdd08b

git clone https://github.com/robocasa/robocasa.git robocasa
git -C robocasa checkout 8f124f0e263de0c583ba5d2acc87137bfd905986
```

检查固定版本：

```bash
git -C robosuite rev-parse HEAD
git -C robocasa rev-parse HEAD
```

不要把第三方目录复制进本仓库，也不要在未记录 commit 的情况下修改它们。确需修改时，应在对应仓库单独建立分支并提交。

## 4. 安装仿真与规划环境

任务配置默认让 MPlib 在名为 `membench` 的 Conda 环境中运行，而 RoboCasa runtime 使用 `robocasa` 环境。为了与配置完全一致，新机器应建立两个环境。

### 4.1 RoboCasa 主环境

```bash
conda create -c conda-forge -n robocasa python=3.11 -y
conda activate robocasa

python -m pip install --upgrade pip
python -m pip install -e ./robosuite
python -m pip install -e ./robocasa
python -m pip install -e ".[dev,video]"
python -m pip install "mplib==0.1.1"
```

该环境负责：

- 创建 RoboCasa 仿真环境；
- 执行专家策略；
- 录制原生 episode；
- 运行评测客户端和轻量协议服务；
- 在 LeRobot 0.3.3 API 兼容时转换数据。

不要在这个环境中直接升级到策略仓库使用的新版 LeRobot。策略训练和真实模型服务应使用独立环境。

### 4.2 MPlib 子进程环境

```bash
conda create -c conda-forge -n membench python=3.11 -y
conda run -n membench python -m pip install --upgrade pip
conda run -n membench python -m pip install "numpy==2.2.5" "mplib==0.1.1"
```

录制进程仍从 `robocasa` 环境启动。规划器看到任务配置中的 `conda_env: membench` 后，会自动通过 `conda run -n membench` 启动 MPlib；本仓库和两个仿真源码目录会通过 `PYTHONPATH` 传给该子进程。

## 5. 初始化 RoboCasa 并下载资产

回到主环境：

```bash
conda activate robocasa
python -m robosuite.scripts.setup_macros
python -m robocasa.scripts.setup_macros
python -m robocasa.scripts.download_kitchen_assets
```

资产下载命令会交互式确认，并下载约 10 GB 的厨房纹理、fixture 和物体模型。资产位于 `robocasa/robocasa/models/assets/`，不会被本仓库提交。

用一个典型物体检查资产是否存在：

```bash
test -f robocasa/robocasa/models/assets/objects/objaverse/cup/cup_2/model.xml
```

该命令没有输出且退出码为 0 表示检查通过。

如果需要在多台机器之间复制而不携带 RoboCasa 的完整资产下载，可以使用仓库内
的最小资产收集器。它会根据所有任务的固定 layout/style、fixture registry 和
代码中的固定物体引用，解析出完整的 mesh/texture 依赖，并记录 SHA-256：

```bash
python scripts/tools/collect_sim_assets.py \
  --manifest /tmp/membench-sim-assets.json \
  --tar /tmp/membench-sim-assets.tar.gz
```

当前固定任务集合生成的包约为 0.489 GiB（压缩后约 225 MiB）。完整说明、恢复
命令和校验脚本见 [`docs/sim_asset_bundle.md`](docs/sim_asset_bundle.md)。

## 6. 配置渲染后端

每个新终端在运行仿真前都要进入仓库、激活环境并设置渲染变量。

无显示器服务器使用 EGL：

```bash
cd /path/to/membench
conda activate robocasa
export PYTHONPATH=.
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

有桌面且需要窗口时使用 GLFW：

```bash
cd /path/to/membench
conda activate robocasa
export PYTHONPATH=.
export MUJOCO_GL=glfw
unset PYOPENGL_PLATFORM
```

不要把 EGL 和 GLFW 配置混用，也不建议把这些变量永久写入全局 shell 配置。

## 7. 验证安装

### 7.1 导入与控制器检查

```bash
python - <<'PY'
from bench.env.robocasa_runtime import (
    bootstrap_local_robocasa_sources,
    make_joint_position_controller_config,
)

bootstrap_local_robocasa_sources()
import mujoco
import robocasa
import robosuite

config = make_joint_position_controller_config()
print("RoboCasa:", list(robocasa.__path__))
print("robosuite:", list(robosuite.__path__))
print("MuJoCo:", mujoco.__version__)
print("right arm controller:", config["body_parts"]["right"]["type"])
PY
```

`robocasa` 和 `robosuite` 路径应指向当前仓库中的两个源码目录，控制器类型应为 `JOINT_POSITION`。

### 7.2 无仿真的配置检查

```bash
python scripts/record_entrypoint.py \
  --task-id WA01 \
  --seeds 0-2 \
  --dry-run

python scripts/evaluate.py \
  --suite configs/eval/suites/wa01_v2.yaml \
  --preflight-only
```

第一个命令应展开 3 个录制 work item；第二个命令应完成任务 catalog 和 suite 检查。

### 7.3 测试

先运行不需要外部策略服务的测试：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q \
  -m "not integration and not external_policy and not policy_e2e"
```

完成资产和 EGL/GLFW 配置后，再运行 simulator 测试：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q -m simulator
```

## 8. 录制专家数据

统一录制入口为 `scripts/record_entrypoint.py`。下面录制 WA-01 的 seed 0–49：

```bash
PYTHONPATH=. MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
python scripts/record_entrypoint.py \
  --task-id WA01 \
  --seeds 0-49 \
  --output-root /data/membench/datasets \
  --dataset-id wa01_part1
```

最终目录为：

```text
/data/membench/datasets/wa01_part1/
├── train/WA-01/                    # 成功且被接受的 episode
├── debug_failed/WA-01/             # 失败 episode
├── _staging/WA-01/                 # 未完成的临时 episode
└── manifest.json
```

`--output-root` 是数据集父目录，`--dataset-id` 是其下一级目录名。已有 episode 默认不会被覆盖；只有明确需要重录时才使用 `--overwrite`。

### 8.1 四进程录制 200 个 seed

可以在四个终端并行运行，但每个进程必须使用不同的 `dataset-id`，否则顶层 `manifest.json` 会互相覆盖：

```bash
PYTHONPATH=. MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python scripts/record_entrypoint.py --task-id WA01 --seeds 0-49 --output-root /data/membench/datasets --dataset-id wa01_200seeds_part1
```

```bash
PYTHONPATH=. MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python scripts/record_entrypoint.py --task-id WA01 --seeds 50-99 --output-root /data/membench/datasets --dataset-id wa01_200seeds_part2
```

```bash
PYTHONPATH=. MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python scripts/record_entrypoint.py --task-id WA01 --seeds 100-149 --output-root /data/membench/datasets --dataset-id wa01_200seeds_part3
```

```bash
PYTHONPATH=. MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python scripts/record_entrypoint.py --task-id WA01 --seeds 150-199 --output-root /data/membench/datasets --dataset-id wa01_200seeds_part4
```

四进程会同时占用 CPU、内存和 GPU 显存。遇到 EGL context 创建失败或 OOM 时，减少为两个进程。

## 9. 转换为 LeRobot 数据集

转换器不读取录制分片根目录的 `manifest.json`。它直接读取每个成功 episode 中的 `episode.json`、`trajectory.h5` 和 `videos/`，并检查 seed 重复、相机列表和数据 schema。

下面将四个分片一次性转换为一个 LeRobot v3 数据集：

```bash
episode_args=()

for shard in \
  /data/membench/datasets/wa01_200seeds_part1 \
  /data/membench/datasets/wa01_200seeds_part2 \
  /data/membench/datasets/wa01_200seeds_part3 \
  /data/membench/datasets/wa01_200seeds_part4
do
  while IFS= read -r -d '' episode_dir; do
    episode_args+=(--episode-dir "$episode_dir")
  done < <(
    find "$shard/train/WA-01" \
      -mindepth 1 -maxdepth 1 -type d \
      -name 'seed_*_attempt_*' -print0
  )
done

python scripts/record/convert_bench_episodes_to_lerobot.py \
  "${episode_args[@]}" \
  --output-root lerobot_data/wa01_200seeds \
  --repo-id local/wa01_200seeds
```

默认只转换 RGB 图像、状态与动作，不包含深度图；需要深度时加 `--depth` 参数（`--no-depth` 可显式关闭）。`--depth` 开启时要求原生 episode 录制了深度数据及其相机标定，并把深度保存为 `observation.depth.*` 特征。相机内参/外参（`observation.camera_intrinsics.*`、`observation.camera_extrinsics_c2w.*`）在原生 episode 中存在时始终导出，与 `--depth` 无关。

v061 转换器（`scripts/record/convert_bench_episodes_to_lerobot_v061.py`）同样支持 `--depth` 开关，默认不转换深度；对应 shell 脚本用 `DEPTH=1` 环境变量开启。

转换器会按 seed 排序并生成 LeRobot 自己的 `meta/info.json` 和 `meta/seeds.jsonl`。如果目标目录已存在，默认拒绝覆盖；追加新 seed 使用 `--write-mode append`，完全重建才使用 `--overwrite`。

原始四个分片可以长期保留，无需先手动合并。若手动复制 episode 到同一目录，不能简单覆盖四份原生 `manifest.json`；这不影响 LeRobot 转换，但会使合并后的原生数据集 manifest 不完整。

## 10. Server/client 评测

策略服务端与 RoboCasa 客户端可以位于同一台机器，也可以位于不同机器。服务端只负责加载 checkpoint 和推理；客户端负责创建环境、发送 observation、执行 action chunk、判断任务成功并保存评测产物。

### 10.1 用 fake policy 验证通信

先在服务端终端启动一个不需要 GPU 和 checkpoint 的测试服务。suite 的 `replan_steps` 为 50，因此 fake policy 的 horizon 也应设为 50：

```bash
conda activate robocasa
cd /path/to/membench
PYTHONPATH=. python scripts/serve/serve_fake.py \
  --host 0.0.0.0 \
  --port 8000 \
  --horizon 50 \
  --image-size 224
```

在客户端终端只做协议 preflight：

```bash
conda activate robocasa
cd /path/to/membench
PYTHONPATH=. python scripts/eval/run_remote.py \
  --endpoint ws://SERVER_IP:8000 \
  --suite configs/eval/suites/wa01_v2.yaml \
  --preflight-only
```

WA01 / WA05 正式评测 v2 使用一次固定专家前置段。运行前须生成对应 seeds 的动作文件；具体边界、生成命令与计分口径见[专家前置评测说明](docs/wa_expert_prefix_evaluation.md)。

同机测试时把 `SERVER_IP` 换成 `127.0.0.1`。跨机器测试时开放 TCP 8000 端口，但不要把未鉴权服务直接暴露到公网。

需要鉴权时，在服务端和客户端设置相同环境变量：

```bash
export MEMBENCH_POLICY_API_KEY='replace-with-a-random-secret'
```

密钥不得写进配置文件、命令脚本或 Git。

### 10.2 准备 ACT/LeRobot 策略环境

ACT 训练和服务使用独立的 Python 3.12 环境。下面的 LeRobot 提交是当前训练脚本和 backend 所依据的版本：

```bash
mkdir -p policy
git clone https://github.com/huggingface/lerobot.git policy/lerobot
git -C policy/lerobot checkout d4a229444b9a09742b4df5af079d308d1aa9a1d0

conda create -c conda-forge -n lerobot python=3.12 -y
conda activate lerobot
python -m pip install --upgrade pip
python -m pip install -e "./policy/lerobot[training]"
python -m pip install -e .
```

PyTorch 必须能识别服务端 GPU：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
PY
```

如果 `CUDA available` 为 `False`，先按 LeRobot 固定提交中的安装说明选择与机器驱动匹配的 PyTorch wheel，再训练或启动服务。

### 10.3 启动真实策略服务

真实模型建议在单独的策略环境中运行，并把策略源码放在 `policy/` 下。不要把新版 OpenPI 或 LeRobot 强行安装到 `robocasa` 环境。

ACT/LeRobot checkpoint 服务示例：

```bash
PYTHONPATH=. python scripts/serve/serve_lerobot.py \
  --checkpoint /path/to/pretrained_model \
  --device cuda \
  --host 0.0.0.0 \
  --port 8000
```

OpenPI 服务端使用项目 fork 的独立环境。源码可以按固定提交放入约定目录：

```bash
mkdir -p policy
git clone --recurse-submodules --branch membench-openpi \
  git@github.com:SiHhangChen/openpi.git policy/openpi
git -C policy/openpi checkout 986ddd4e26561c1d488813f1b72d159d8e5435bf
```

完成 `policy/openpi/README.md` 中的安装和 checkpoint 准备后启动：

```bash
PYTHONPATH=. python scripts/serve/serve_openpi.py \
  --config OPENPI_CONFIG_NAME \
  --checkpoint /path/to/openpi/checkpoint \
  --image-size 224 \
  --state-schema membench.robot_state37.v2 \
  --host 0.0.0.0 \
  --port 8000
```

AC-DiT backend 当前仍是未实现占位，不能用于正式评测。

### 10.4 运行闭环评测

先运行 preflight，确认服务端声明的相机、图像尺寸、state/action schema、action horizon 和 20 Hz 控制频率与 suite 一致：

```bash
PYTHONPATH=. python scripts/eval/run_remote.py \
  --endpoint ws://SERVER_IP:8000 \
  --suite configs/eval/suites/wx01_v1.yaml \
  --preflight-only
```

通过后，在 RoboCasa 客户端运行闭环评测：

```bash
PYTHONPATH=. MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
python scripts/eval/run_remote.py \
  --endpoint ws://SERVER_IP:8000 \
  --suite configs/eval/suites/wx01_v1.yaml \
  --output-root outputs/eval \
  --run-id wx01-smoke-model-name
```

评测默认支持按 seed 恢复。使用相同 `--run-id` 重跑时会跳过已完成 episode；需要重试错误 episode 时增加 `--retry-errors`。

每次运行会保存 run manifest、逐 episode 结果、视频和 `summary.json`。中断或复制结果后可重建汇总：

```bash
python scripts/eval/summarize_run.py outputs/eval/wx01-smoke-model-name
```

正式实验应保存：

- 本仓库 commit；
- RoboCasa、robosuite 和策略仓库 commit；
- suite 文件及其 hash；
- checkpoint URI/fingerprint；
- `outputs/eval/<run-id>/` 完整目录。

## 11. 训练入口

本项目维护的训练入口统一位于 `scripts/train/<policy>/`，完整索引见
`scripts/train/README.md`。训练日志统一放在 `outputs/train/<policy>/`。

### ACT

WX-01 的 ACT 配置位于 `configs/policies/act_wx01_v1.yaml`。先修改其中的数据集根目录、repo ID 和输出目录，然后在安装了兼容 LeRobot 源码、PyTorch 和 CUDA 的独立环境中检查配置：

```bash
PYTHONPATH=. python scripts/train/act/train_act_wx01.py \
  --config configs/policies/act_wx01_v1.yaml \
  --dry-run
```

确认生成的 resolved config 正确后开始训练：

```bash
PYTHONPATH=. python scripts/train/act/train_act_wx01.py \
  --config configs/policies/act_wx01_v1.yaml
```

训练脚本会验证 LeRobot 数据集的 state37、action13、三路相机和四元数连续性，并给 checkpoint 写入 `membench_policy.json`。策略环境的 PyTorch/CUDA/LeRobot 安装以对应策略仓库的固定 commit 和 README 为准。

## 12. 数据合同摘要

固定 observation profile 为 `configs/benchmark/observation_profiles/rgbd_v3.yaml`：

- 相机：`robot0_agentview_left`、`robot0_agentview_right`、`robot0_eye_in_hand`；
- 图像：OpenCV/top-left，RGB-D，224×224；
- state：`membench.robot_state37.v2`；
- action：`membench.pandaomron_hybrid13.v2`；
- 控制频率：20 Hz。

Action 是单一 13 维向量：7 维 Panda 绝对关节目标、1 维夹爪、3 维移动底盘、1 维 torso 和 1 维 base/arm mode。State 是 37 维向量，包含机械臂、夹爪、底盘、torso 和末端执行器状态。完整字段与单位由 `bench/runtime/data_contract.py` 定义，并写入原生 HDF5 和 LeRobot metadata。

## 13. 常见问题

### `ModuleNotFoundError: robocasa` 或导入了系统版本

确认当前目录是仓库根目录，已执行 `export PYTHONPATH=.`，并检查 `robocasa/`、`robosuite/` 的目录名和固定 commit。

### `EnvironmentNameNotFound: membench`

任务正在按配置启动 MPlib 子进程，但没有找到规划环境。按第 4.2 节创建名为 `membench` 的 Conda 环境。

### EGL 初始化失败

先检查 `nvidia-smi`，再确认 NVIDIA 驱动提供 EGL。桌面机器可改用 `MUJOCO_GL=glfw` 并执行 `unset PYOPENGL_PLATFORM`。

### 视频写入失败

确认 `ffmpeg -version` 正常，并在 `robocasa` 环境中检查 `imageio` 和 `imageio-ffmpeg`。

### 录制重跑时报目录已存在

这是防止误覆盖的预期行为。改用新的 `dataset-id` 或 seed 范围；只有确定要删除并重录已有结果时才使用 `--overwrite`。

### 四个录制进程能否使用同一个 `dataset-id`

不能安全共用。episode 目录虽然按 seed 隔离，但四个进程都会写同一份顶层 `manifest.json`。应使用不同分片 ID，最终在 LeRobot 转换阶段统一合并。

### LeRobot 转换是否依赖原生 `manifest.json`

不依赖。转换器读取每个成功 episode 自己的元数据、轨迹和视频，并生成新的 LeRobot metadata。

## 14. 主要入口

| 目的 | 命令 |
| --- | --- |
| 展开录制计划 | `python scripts/record_entrypoint.py --task-id WA01 --seeds 0-2 --dry-run` |
| 录制专家 episode | `python scripts/record_entrypoint.py ...` |
| 转换 LeRobot 数据集 | `python scripts/record/convert_bench_episodes_to_lerobot.py ...` |
| 检查评测 suite | `python scripts/evaluate.py --suite <suite.yaml> --preflight-only` |
| 启动协议测试服务 | `python scripts/serve/serve_fake.py --horizon 50` |
| 启动 ACT 服务 | `python scripts/serve/serve_lerobot.py ...` |
| 启动 OpenPI 服务 | `python scripts/serve/serve_openpi.py ...` |
| 运行远程闭环评测 | `python scripts/eval/run_remote.py ...` |
| 重建评测汇总 | `python scripts/eval/summarize_run.py <run-dir>` |

架构说明见 `docs/design.md` 和 `docs/bench_migration_status.md`；任务定义与资源位于 `bench/tasks/<task_id>/`。
