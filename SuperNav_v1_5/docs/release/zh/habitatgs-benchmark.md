[English](../en/habitatgs-benchmark.md) | **简体中文**

# 运行 HabitatGS-Benchmark：单目标与多目标导航

[公开数据集](https://huggingface.co/datasets/yyh929/HabitatGS-Benchmark)包含
ObjectNav 与 Ordered Multi-ObjectNav 各 150 条任务，合计覆盖 70 个 InteriorGS 场景。
本指南使用公开的 `habitat_gs/objectnav/v1` 格式；本地历史 `new2` 或
`multi_objectnav_global_task/v2` 文件需使用其对应的历史流程。

**公开 `episodes.json` 不能直接传给 `supernav run --instructions`。**
先运行 `supernav prepare habitatgs` 转换，再运行导航、导出逐动作证据并独立评分。
`score-objectnav` 用于 HM3D/OVON；旧 `score-multi-objectnav` 用于 goal-mark/bbox
协议，均不实现本公开 GS benchmark 的评分规则。

## 1. 安装和下载

先完成 [Habitat-GS 安装](habitat.md)和 [Agent 配置](configuration.md)。
Agent 与模拟器环境均安装这次更新后的 SuperNav；仅更新 Agent 环境无法获得新的
仿真器逐动作记录功能。以下命令从仓库根目录执行：

```bash
python -m pip install -e '.[agents,evaluation]'
python -m pip install huggingface_hub
export SUPERNAV_HABITAT_PYTHON=/path/to/habitat-env/bin/python
"$SUPERNAV_HABITAT_PYTHON" -m pip install '.[evaluation]'

export GS_DATA=/path/to/storage/habitatgs-benchmark
export GS_SCENES=/path/to/storage/habitatgs-scenes
export GS_WORK=/path/to/storage/supernav-gs
export GS_REVISION=4a8fe5cf654c1d19f0c3c992720d3c4f57844f12
hf download yyh929/HabitatGS-Benchmark --repo-type dataset \
  --revision "$GS_REVISION" --local-dir "$GS_DATA"
python "$GS_DATA/evaluation/verify.py" "$GS_DATA"
```

`GS_REVISION` 是本适配器核验过的数据版本。保留实际使用的完整 commit SHA 和
manifest 哈希；更新数据时重新校验格式和评分合同。下载标注、GT 和运行输出应保存在
源码 checkout 之外。不要将 GT 文件或评测器的文件访问权限提供给被测策略。
转换器生成的任务清单仅含指令、起点、场景、目标文字顺序和数据版本标识，
不会复制 bbox、viewpoints、实例标签或参考路径。

## 2. 准备匹配场景

HF 发布包只含标注、评测脚本和场景哈希清单，**不含场景二进制文件**。
取得符合 [InteriorGS 数据条款](https://github.com/manycore-research/InteriorGS)
且与 `scene-assets/v1/manifest.json` 匹配的运行资产：

```text
habitatgs-scenes/
└── interior_<scene-id>/
    ├── <scene-id>.gs.ply
    └── <scene-id>.navmesh
```

允许在这个目录结构中链接本机已有资产，转换器会校验实际文件大小和 SHA256。
公开发布说明尚未提供完整的 source-to-runtime 转换及匹配 NavMesh 的分发途径；
资产未备齐时，不能仅靠下载标注运行仿真。原始压缩 PLY 改名、任意重烘焙的 NavMesh
都不能代替经过验证的运行资产。

下面的上游检查要求 `GS_SCENES` 中备齐全部 140 个场景文件，并拒绝解析到该目录外的
符号链接。使用外部符号链接或只准备部分场景时，继续执行下一节的转换：
`supernav prepare habitatgs` 会在写入输出前验证所选任务每个 PLY/NavMesh 的大小和 SHA256。

```bash
python "$GS_DATA/evaluation/verify.py" "$GS_DATA" --scene-root "$GS_SCENES"
```

## 3. 转换任务与运行配置

`GS_TASK` 使用 `objectnav` 或 `multi-objectnav`：

```bash
export GS_TASK=multi-objectnav
supernav prepare habitatgs \
  --dataset-root "$GS_DATA" --task "$GS_TASK" \
  --dataset-revision "$GS_REVISION" --scene-root "$GS_SCENES" \
  --template-config configs/experiments/habitat-geo-based-executor.json \
  --sensor-height 1.25 --output-dir "$GS_WORK/$GS_TASK"
```

转换器验证任务/schema/场景清单哈希和所选场景的 PLY/NavMesh，生成：

| 输出 | 内容 |
| --- | --- |
| `instructions.json` | SuperNav 任务清单；保留原始 episode ID、指令、起点与目标顺序 |
| `run.json` | 已解析的实验配置、输入/输出路径、primitive trace 设置 |
| `scenes.scene_dataset_config.json`、`stages/`、`scenes/` | 引用本机场景资产的 Habitat 配置 |
| `preparation.json` | 数据版本、哈希、选中任务 ID、相机高度与资产验证记录 |

多目标的零基 target index 会转换成 `hab_nav_goals` 所需的一基索引，目标顺序保持不变。
标记工具用于 Agent 记忆，**最终成绩不依赖标记**。多目标转换会开放 `hab_nav_goals`，
按 Geo 或学习型执行器选择多目标原生 Skill，并安装其依赖的全局与局部 Skill。
多目标转换支持这两种原生执行器模板；不支持的 arm/Skill 组合会在写文件前报错。
转换器保留数据集 NavMesh，禁止自动
按 Agent 参数重建。输出目录必须不存在，以免覆盖已有任务或运行证据。

相机高度 `1.25` 是此示例的显式设置，不是数据集规定的唯一设置。模型、图像参数、
动作步长/转角、滑动、执行器以及所有预算均需在实验记录中报告。
使用 Learned Executor 时，先按 [策略服务指南](habitat.md#learned-executor-weights)
启动服务，再将模板换为 `habitat-learned-executor.json`，使用另一输出目录。

只准备一个已安装场景中的任务时，可增加 `--task-ids <episode-id>`；默认转换完整任务集。

## 4. 预检、单任务和完整运行

```bash
export GS_ID=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["instructions"][0]["task_id"])' \
  "$GS_WORK/$GS_TASK/instructions.json")

supernav run --config "$GS_WORK/$GS_TASK/run.json" --task-ids "$GS_ID" \
  --dry-run --output-dir "$GS_WORK/$GS_TASK/dry-run"
supernav run --config "$GS_WORK/$GS_TASK/run.json" --task-ids "$GS_ID" \
  --output-dir "$GS_WORK/$GS_TASK/smoke-runs"

# 首条真实运行确认观测、动作和日志正常后，执行全部已转换任务。
supernav run --config "$GS_WORK/$GS_TASK/run.json"
```

`--dry-run` 不调用模型或仿真器；不能用它证明导航或权重加载成功。
同一评分目录只放一种方法、一个 arm、一次 repetition。重复运行使用独立目录，
导出器会拒绝重复 episode ID，避免无意选择成功重试。并行 runner 还需独立 bridge 端口。
导出器还会核验 `canonical.jsonl` 中只有一个已初始化的 session，且与 metrics 一致；
不能通过运行中重启会话悄悄替换轨迹。

仿真器在 `visuals/<session_id>.primitive.jsonl` 记录真实起点、每个底层动作后的位置及
显式关闭会话对应的 STOP。转向、碰撞和微小位移都会保留；普通清理/进程退出不会制造 STOP。
中途重设 Agent 位姿会记录为 runtime error。该文件属于 evaluator 证据，不返回给策略。
MCP artifact 资源的列表与读取接口均排除逐动作轨迹文件。

## 5. 导出并评分

以下默认评估完整运行目录 `runs/`。单任务检查时改成 `smoke-runs/`：

```bash
supernav export-habitatgs \
  --dataset-root "$GS_DATA" --task "$GS_TASK" \
  --runs-root "$GS_WORK/$GS_TASK/runs" \
  --visuals-root "$GS_WORK/$GS_TASK/visuals" \
  --output "$GS_WORK/$GS_TASK/submissions.json"

"$SUPERNAV_HABITAT_PYTHON" -m supernav score-habitatgs \
  --dataset-root "$GS_DATA" --task "$GS_TASK" --scene-root "$GS_SCENES" \
  --submissions "$GS_WORK/$GS_TASK/submissions.json" \
  --reference global_shortest --output "$GS_WORK/$GS_TASK/scores.json"
```

导出的 submission 也可交给同版本 HF `evaluation/score.py`。该上游命令即使只提交部分任务，
也要求完整场景目录，并拒绝解析到场景根目录外的资产符号链接。
SuperNav 保留公开评分器的逐任务 SR/SPL 合同，并额外输出多目标 `stage_sr`。
评分只需正确的 NavMesh，不需要渲染场景或调用模型；加载前校验 NavMesh 哈希。
旧运行只有工具终点或去重轨迹时，导出器会拒绝伪造缺失动作，需要重新采集合规证据。
文件缺失/不一致时会报错；不得为了导出成功删除失败任务或用 `goal_marks` 替代轨迹。

输出文件必须使用新路径。未提交的任务按失败计入完整数据集，因此只跑一条时仍会报告
其余 149 条 missing submissions；这不是完整 benchmark 的性能评测。
SuperNav 仅加载已提交任务的 NavMesh，并在输出 provenance 中列出已验证的网格。
如果基础设施故障导致没有 session/trace，需要人工核验错误证据；导出器会报错，
不会猜测其状态。

## 6. 成功规则、SPL 与分阶段 SR

- **Single**：执行 STOP 时，到任一 approved viewpoint 的 NavMesh 测地距离严格 `<0.1m`。
- **Multi**：沿真实轨迹按顺序命中所有目标区域，再 STOP；终点不必继续停留在最后目标附近。
  重叠的连续区域可在同一记录位置完成。不使用 bbox、单个参考终点或欧式距离替代测地距离。
- **SPL**：`S × L / max(L, P)`。`L` 默认使用公开 `global_shortest` 参考，`P` 使用逐动作路长。
  失败为 0；成功且 `L=P=0` 时为 1。选择 `local_shortest` 时需要明确报告参考变体。
- **SRk**：至少有 k 个目标的任务中，按顺序完成前 k 个目标的比例，阶段完成不要求 STOP。
  各任务目标数不同，因此 SR5 不等于整体 SR。

完整 Multi150 的阶段资格数为：

| 指标 | SR1 | SR2 | SR3 | SR4 | SR5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `eligible_total` | 150 | 150 | 112 | 75 | 38 |

公开合同将 `runtime_error` 排除出 SR/SPL 均值，其 success/SPL 为 null；预算耗尽、
未 STOP、缺少 submission 则为失败。分阶段统计沿用相同异常排除规则，逐阶段输出
`eligible_total`、`evaluated`、`runtime_errors`、`successes`、`success_rate`。
必须同时报告资格数、实际分母和错误数；有 runtime errors 时不能只报告成功率。
发生 STOP、预算耗尽或 runtime error 后的事件不再计入。

运行生成的 `metrics.json.success` 是 Agent 完成声明/工程状态，论文指标应读取独立的
`scores.json`。HM3D/OVON 属于单目标任务，其运行与 SR/SPL 见
[HM3D v2 与 OVON 指南](hm3d-and-ovon.md)，没有多目标 SR2～SR5。
