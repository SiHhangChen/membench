# WR-11 正式采样规则

当前任务版本为 **3.2.0**，采样 profile 为 `WR-11/sampling-v2`。使用 layout 33/style 34/scene seed 0，保留沿岛台平移的任务设计。颜色池扩展为红、橙、黄、绿、青、蓝、紫、黑；每次仍有三盒候选。

| 变量 | 分布或规则 | 范围 |
| --- | --- | --- |
| 三盒候选包装 | 八色池中无放回抽取三色，按顺序放到左、中、右 | 336 种排列，每种概率 1/336 |
| 样品包装 | 用独立请求随机流，从本次三种候选颜色中等概率选择 | 每候选概率 1/3；每种颜色边际概率 1/8 |
| 样品和三盒候选 XY | 各自托盘中心的圆内，按面积均匀抽取 | 半径 0.025 m |
| 四盒牛奶的 yaw | 每盒独立均匀抽取 | `[−π, π]` |
| 样品/返回站位 XY | 样品站位圆内按面积均匀抽取 | 中心 `(1.00, −1.62)`，半径 0.05 m |
| 取货站位 XY | 取货站位圆内按面积均匀抽取 | 中心 `(3.30, −1.62)`，半径 0.05 m |
| 机器人 yaw | 两站保持同一朝向 | `−π/2`，角度随机范围为零 |
| 固定部分 | 房间、风格、托盘中心、牛奶盒几何、躯干高度 | 躯干高度 0 |

样品托盘中心 `(0.85, −2.15)`，送回托盘中心 `(1.22, −2.15)`；左、中、右候选托盘中心分别为 `(2.90, −2.15)`、`(3.30, −2.15)`、`(3.70, −2.15)`。这些是世界坐标的 XY，单位为米；高度根据台面与托盘几何计算。托盘本身不随机移动，送回目标托盘固定。

![站位与采样范围俯视示意](../outputs/diagnostics/wr11/colors8/station_sampling_diagram.png)

上图圆盘中心和半径使用当前配置，台面与托盘轮廓为示意。下方三个面板使用相同厘米尺度，散点来自 seeds 0–199 的实际解析结果，星号表示 seed 0。站位圆描述机器人基座参考点的采样范围；物体圆描述各自托盘中心附近的 XY 位移范围。

## 圆采样与独立性

圆采样使用 `r = R√u`、`θ ∼ Uniform(0, 2π)`、`u ∼ Uniform(0, 1)`，因此圆面积内概率密度均匀。半径直接均匀抽取会造成中心过密，这里使用面积均匀采样。

候选排列、条件于候选的样品选择、每个物体的 XY、每个物体的 yaw、样品站位和取货站位分别使用带名字或 salt 的随机流。样品与候选颜色必须关联，以保证恰好一个匹配；给定候选排列时每盒成为目标的概率仍为 1/3，不能只看候选布局确定目标。共有 56 种无序候选色组、336 种有序候选排列和 1008 种样品与排列组合。

姿态与站位不依据颜色或正确候选生成。修改物体位移半径不会改变包装、排列、物体角度或机器人站位；导航调用次数与用途也不会改变站位。

每个 seed 在 episode 解析时一次性确定全部采样。初始机器人直接位于该 episode 的样品站位；观察、返回送货复用同一点，取货使用该 episode 的另一个站位点。专家通过 `EpisodeStationSampler` 读取已解析点，不在运行中继续抽取。到达目标仍使用真实物理平移，允许配置中的 8 cm 导航停止误差；采样圆描述的是目标分布，实测终点不应误作采样点。

独立流并不承诺有限种子集合内各类别计数严格相等。没有根据正确候选、可抓取性或轨迹成功重新抽取随机数；运行失败应保留原 seed 和失败原因，不能通过挑选成功样本宣称原始采样分布已全部验证。

## 配置与校验

- `bench/tasks/wr11/resources/sampling/regions.json`：包装、排列、物体圆和角度的唯一参数来源。
- `bench/tasks/wr11/resources/calibration/stations.json`：机器人两处圆的唯一参数来源。
- `bench/tasks/wr11/domain/scene.py`：已校准台面、托盘与站位中心。

加载时校验两处圆的中心、场景编号、朝向和区域描述一致；拒绝物体半径超过 2.5 cm、站位半径超过 5 cm、额外站位、不同朝向、区域与 circle 的半径不一致。可以缩小半径；扩大半径需要重新校准和验证。现阶段不直接使用其他任务的 4.5 cm 物体半径，因为当前托盘支撑和抓取范围的证据来自 2.5 cm 范围。

reset 后验证样品与三盒候选均与各自托盘真实接触，初始样品在至少一路相机可见。取货和首次选择继续要求样品完整离开全部三路相机视锥。缺失可见性或接触证据不能成功。

采样 profile、物体偏移/角度、两处站位 XY、共同 yaw、站位半径均写入私有 episode facts；初始化记录额外注明样品站位复用。固定策略 prompt 不含目标包装或正确候选。包装和位置的身份信息不进入策略观测。

## 验证证据

10000 个解析种子 `0–9999` 中，红/橙/黄/绿/青/蓝/紫/黑样品计数为 `1228/1255/1287/1204/1267/1276/1242/1241`，336 种候选排列和 1008 种样品与排列组合均出现。目标位于左/中/右的次数为 `3218/3423/3359`。全部物体和站位点位于指定圆内，归一化半径平方均值为 `0.495–0.502`，面积均匀分布的理论值为 `0.5`。

![10000 个种子的物体与站位圆分布](../outputs/diagnostics/wr11/colors8/draws10000/sampling_support.png)

抽样 CSV 和统计位于 `outputs/diagnostics/wr11/colors8/draws10000/`。统计仅针对解析后的采样，不能当作 10000 条物理轨迹成功率。

八色版本 seed 0 检查两处站位圆每 45° 的八个边界点，共 16 个点，样品站位可见性、取货站位样品不可见性、全部候选可见性及托盘真实支撑均通过。证据位于 `outputs/diagnostics/wr11/colors8/scene_seed0/sampling_boundaries.json`。这是几何和接触检查，不是每个边界点的抓取可达性证明。

实际渲染保留白色图案和一致的几何、光泽，八色对照见下图；对照模式只修改诊断环境的样品纹理，不生成训练轨迹。

![八种包装实际渲染](../outputs/diagnostics/wr11/colors8/scene_seed0/palette_comparison.png)

相关自动测试 **229 passed**。完整录制、原生动作回放和具体步数见 [当前验证记录](wr11_validation.md)。完整 50 种子 rollout 尚未执行，不能把小样本结果当作完整圆域或全种子成功率。旧三色版本的证据见 [v3.1 验证归档](wr11_v3_1_sampling_validation.md)。

## 复现

生成站位示意图（PNG、SVG、PDF）：

```bash
PYTHONPATH="$PWD/.venv-mplib/lib/python3.11/site-packages" \
MPLCONFIGDIR=/tmp/wr11-sampling-mpl \
.venv-robocasa/bin/python scripts/analysis/plot_wr11_sampling_layout.py \
  --output outputs/diagnostics/wr11/colors8/station_sampling_diagram
```

纯采样审计入口：

```bash
.venv-robocasa/bin/python scripts/analysis/analyze_wr11_sampling.py \
  --count 10000 --output outputs/diagnostics/wr11/colors8/new_draws
```

该命令需要 Python 环境中有 NumPy、PyYAML 和 Matplotlib。本机仿真环境缺少 Matplotlib，已生成图使用现有规划环境的包：

```bash
PYTHONPATH="$PWD/.venv-mplib/lib/python3.11/site-packages" \
MPLCONFIGDIR=/tmp/wr11-sampling-mpl \
.venv-robocasa/bin/python scripts/analysis/analyze_wr11_sampling.py \
  --count 10000 --output outputs/diagnostics/wr11/colors8/new_draws
```

物理边界检查使用 GPU/EGL：

```bash
.venv-robocasa/bin/python scripts/planning/validate_wr11.py \
  --seed 0 --scene-only --sampling-boundaries --palette-preview \
  --output outputs/diagnostics/wr11/colors8/new_scene_seed0
```

`--scene-only` 重设机器人站位，仅属于场景诊断，不生成训练成功轨迹。完整录制去掉 `--scene-only --sampling-boundaries`；动作回放方式见 [任务说明](../bench/tasks/wr11/README.md)。
