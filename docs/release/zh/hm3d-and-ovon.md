[English](../en/hm3d-and-ovon.md) | [简体中文](hm3d-and-ovon.md)

# 运行 HM3D v2 ObjectNav 与 HM3D-OVON

本教程介绍如何转换外部 episode 数据、使用 SuperNav 几何执行器运行任务，以及对
保存的轨迹进行评分。请先完成 [Habitat-GS 安装](habitat.md)与
[agent 配置](configuration.md#模型与凭证)。以下命令均在 SuperNav 仓库根目录执行，
agent 环境中应能使用 `supernav` 和 `python`。模拟器环境也应安装相同版本的 SuperNav。

HM3D mesh 场景直接使用 SuperNav Habitat 后端所用的 Habitat-GS `habitat_sim` SDK
加载。

## 选择数据

场景与导航 episode 需要分别下载：

| 输入 | 内容 | 来源 |
| --- | --- | --- |
| HM3D/HM3DSem **v0.2** 场景 | 可渲染场景、scene-instance 配置、语义资产和 NavMesh | [Matterport HM3D 下载说明](https://github.com/matterport/habitat-matterport-3dresearch#-downloading-hm3d-v02) |
| `objectnav_hm3d_v2` episodes | 使用 HM3DSem v0.2 场景的 ObjectNav 任务 | [Habitat-Lab 任务数据表](https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md#task-datasets) |
| HM3D-OVON episodes | 使用 HM3D 场景的开放词汇导航任务 | [OVON 作者说明](https://github.com/naokiyokoyama/ovon#-downloading-the-datasets)与 [episode 压缩包](https://huggingface.co/datasets/nyokoyama/hm3d_ovon/tree/main) |
| OVON 选择清单 | 当前转换器使用的外部 `our-set/ovon_full_set.json` | [MTU3D 数据说明](https://github.com/MTU3D/MTU3D#prepare-data)与[基准数据压缩包](https://huggingface.co/datasets/bigai/MTU3D/tree/main) |

本教程中的“HM3D v2”指 **ObjectNav v2 episode 数据与 v0.2 场景的组合**。

`supernav prepare objectnav` 读取按场景保存的 ObjectNav split。
`supernav prepare ovon` 根据外部 MTU3D 格式清单选择 OVON episodes，下面使用其中的
`val_unseen` 选择集。

## 准备存储与资产

在容量足够的数据盘上设置绝对路径。每次实验使用新的工作目录，任务清单、目标与生成的
运行证据保存在公开源码目录之外。已有安装可直接填写自己的路径。

```bash
export SUPERNAV_DATA_ROOT=/path/to/storage/datasets
export SUPERNAV_WORK_DIR=/path/to/storage/supernav-hm3d-ovon
export SUPERNAV_HABITAT_PYTHON=/path/to/storage/habitat-env/bin/python

mkdir -p "$SUPERNAV_DATA_ROOT" "$SUPERNAV_WORK_DIR/.cache" \
  "$SUPERNAV_WORK_DIR/.cache/tmp" \
  "$SUPERNAV_WORK_DIR/private" "$SUPERNAV_WORK_DIR/configs" \
  "$SUPERNAV_WORK_DIR/runs"
export XDG_CACHE_HOME="$SUPERNAV_WORK_DIR/.cache"
export PIP_CACHE_DIR="$SUPERNAV_WORK_DIR/.cache/pip"
export HF_HOME="$SUPERNAV_WORK_DIR/.cache/huggingface"
export TORCH_HOME="$SUPERNAV_WORK_DIR/.cache/torch"
export TMPDIR="$SUPERNAV_WORK_DIR/.cache/tmp"

"$SUPERNAV_HABITAT_PYTHON" -c 'import habitat_sim, supernav; print(habitat_sim.__file__, supernav.__file__)'
```

通过[数据提供方](https://github.com/matterport/habitat-matterport-3dresearch)获取 HM3D
访问权限。将自己的 Matterport token ID 和 secret 设置在以下环境变量中，然后使用
Habitat 下载器获取 v0.2 验证场景：

```bash
"$SUPERNAV_HABITAT_PYTHON" -m habitat_sim.utils.datasets_download \
  --username "$MATTERPORT_TOKEN_ID" --password "$MATTERPORT_TOKEN_SECRET" \
  --uids hm3d_val_v0.2 --data-path "$SUPERNAV_DATA_ROOT"
```

`hm3d_val_v0.2` 下载组包含 Habitat 场景资产、配置和语义资产。若所选任务使用训练场景，
还需下载 `hm3d_train_v0.2`。保留原始 `.basis.navmesh` 文件。

ObjectNav 请从 Habitat-Lab 数据表下载 `objectnav_hm3d_v2.zip`，解压后使 episode
根目录下存在 `val/content/`。OVON 请解压作者提供的 `hm3d.tar.gz`，使 episode
根目录下存在 `val_unseen/content/`。使用下述 OVON 选择集时，从 MTU3D 的
`embodied_bench_data.tar.gz` 中取得 `our-set/ovon_full_set.json`，保存到私有目录。

目录应具有以下结构；各 split 上层的目录名称可以不同，通过变量指定实际位置即可：

```text
datasets/
├── scene_datasets/hm3d/
│   ├── hm3d_annotated_basis.scene_dataset_config.json
│   └── val/<number>-<scene_hash>/
│       ├── <scene_hash>.basis.glb
│       ├── <scene_hash>.basis.navmesh
│       ├── <scene_hash>.basis.scene_instance.json
│       └── ... semantic and stage configuration assets ...
├── objectnav_hm3d_v2/
│   └── val/content/<scene_hash>.json.gz
└── hm3d_ovon/hm3d/
    └── val_unseen/content/<scene_hash>.json.gz

supernav-hm3d-ovon/private/our-set/ovon_full_set.json
```

```bash
export SUPERNAV_HM3D_SCENES_ROOT="$SUPERNAV_DATA_ROOT/scene_datasets/hm3d"
export SUPERNAV_SCENE_DATASET_CONFIG="$SUPERNAV_HM3D_SCENES_ROOT/hm3d_annotated_basis.scene_dataset_config.json"
test -f "$SUPERNAV_SCENE_DATASET_CONFIG"
```

场景数据配置必须注册所选场景及其 NavMesh、语义资产。转换器保留区分大小写的场景
hash，用于查找 scene instance；不要把它替换为 `.glb` 路径，否则直接加载 mesh
可能绕过已注册的 NavMesh。

## 转换 HM3D v2 ObjectNav 任务

```bash
export SUPERNAV_TASK_SET=hm3d-v2
export SUPERNAV_SPLIT=val
export SUPERNAV_EPISODES_ROOT="$SUPERNAV_DATA_ROOT/objectnav_hm3d_v2"

supernav prepare objectnav \
  --episodes-root "$SUPERNAV_EPISODES_ROOT" \
  --split "$SUPERNAV_SPLIT" \
  --scene-dataset-config "$SUPERNAV_SCENE_DATASET_CONFIG" \
  --sensor-height 1.25 \
  --template-config configs/experiments/habitat-geo-based-executor.json \
  --instructions-dir "$SUPERNAV_WORK_DIR/private/$SUPERNAV_TASK_SET" \
  --config-dir "$SUPERNAV_WORK_DIR/configs/$SUPERNAV_TASK_SET"

export SUPERNAV_INSTRUCTIONS="$SUPERNAV_WORK_DIR/private/$SUPERNAV_TASK_SET/hm3d_v2_${SUPERNAV_SPLIT}.instructions.json"
export SUPERNAV_PREPARED_CONFIG="$SUPERNAV_WORK_DIR/configs/$SUPERNAV_TASK_SET/hm3d-${SUPERNAV_SPLIT}-localnav.json"
```

首次只准备少量任务时，可加上 `--max-episodes 1`。如果只安装了部分场景，先把
`SUPERNAV_SCENE_HASH` 设为已安装的场景 hash，再增加
`--scenes "$SUPERNAV_SCENE_HASH"`；该参数也接受逗号分隔的多个 hash。去掉限制即可转换
完整 split；若需要保留已有实验，应写入新的私有目录。

相机高度设为 1.25 m，可通过 `--sensor-height` 调整。

接下来执行下文的“先运行单任务，再批量运行”。若要运行 OVON，则改用以下转换命令。

## 转换 OVON 选择集

选择文件在 `val_unseen` 键下保存列表，每一项含 `scan_id_suffix`、`episode_index`
与 `object_category`。其中 `episode_index` 是对应场景 episode 数组的从零开始的
下标，不是 `episode_id`。转换器会检查选择清单中的类别与原始 episode 是否一致。

本流程使用 `children_object_categories` 为空或不存在的 episodes。
转换器与评分器读取 episode 自身 `object_category` 对应的目标视点。

```bash
export SUPERNAV_TASK_SET=ovon
export SUPERNAV_SPLIT=val_unseen
export SUPERNAV_EPISODES_ROOT="$SUPERNAV_DATA_ROOT/hm3d_ovon/hm3d"
export SUPERNAV_OVON_SELECTION="$SUPERNAV_WORK_DIR/private/our-set/ovon_full_set.json"

supernav prepare ovon \
  --episodes-root "$SUPERNAV_EPISODES_ROOT" \
  --manifest "$SUPERNAV_OVON_SELECTION" \
  --split "$SUPERNAV_SPLIT" \
  --scene-dataset-config "$SUPERNAV_SCENE_DATASET_CONFIG" \
  --sensor-height 1.25 \
  --template-config configs/experiments/habitat-geo-based-executor.json \
  --instructions-dir "$SUPERNAV_WORK_DIR/private/$SUPERNAV_TASK_SET" \
  --config-dir "$SUPERNAV_WORK_DIR/configs/$SUPERNAV_TASK_SET" \
  --output-dir "$SUPERNAV_WORK_DIR/runs/$SUPERNAV_TASK_SET-batch"

export SUPERNAV_INSTRUCTIONS="$SUPERNAV_WORK_DIR/private/$SUPERNAV_TASK_SET/ovon_${SUPERNAV_SPLIT}_mtu3d120.instructions.json"
export SUPERNAV_PREPARED_CONFIG="$SUPERNAV_WORK_DIR/configs/$SUPERNAV_TASK_SET/ovon-${SUPERNAV_SPLIT}-mtu3d120-localnav.json"
```

生成配置使用 `--template-config` 所选的执行器。

## 先运行单任务，再批量运行

完成任意一种任务转换后，执行以下步骤。几何执行器模板已提供 `prompts_dir`、原生导航
Skills、`default` arm 和模型/客户端设置，使用 oracle depth 与 NavMesh 几何。
转换过程设置 `HAB_DEFAULT_AGENT_NAVMESH=0`，保留数据集原始 NavMesh。

创建本地覆盖配置，显式指定输出位置。`visuals_root` 将图像、审计日志与轨迹 sidecar
放在数据盘上，下文评分也使用相同目录。

```bash
export SUPERNAV_CONFIG="$SUPERNAV_WORK_DIR/configs/$SUPERNAV_TASK_SET/run.json"
export SUPERNAV_VISUALS_ROOT="$SUPERNAV_WORK_DIR/runs/$SUPERNAV_TASK_SET-visuals"
export SUPERNAV_RUNS_ROOT="$SUPERNAV_WORK_DIR/runs/$SUPERNAV_TASK_SET-one"
export SUPERNAV_SWEEP_ID=one

python - <<'PY'
import json
import os
from pathlib import Path

config = {
    "extends": os.environ["SUPERNAV_PREPARED_CONFIG"],
    "output_dir": os.environ["SUPERNAV_RUNS_ROOT"],
    "visuals_root": os.environ["SUPERNAV_VISUALS_ROOT"],
}
Path(os.environ["SUPERNAV_CONFIG"]).write_text(json.dumps(config, indent=2) + "\n")
PY

export SUPERNAV_TASK_ID=$(python - <<'PY'
import json
import os
from pathlib import Path

rows = json.loads(Path(os.environ["SUPERNAV_INSTRUCTIONS"]).read_text())["instructions"]
print(rows[0]["task_id"])
PY
)

supernav run --config "$SUPERNAV_CONFIG" \
  --sweep-id "$SUPERNAV_SWEEP_ID" \
  --task-ids "$SUPERNAV_TASK_ID" --arms default \
  --dry-run --output-dir "$SUPERNAV_WORK_DIR/runs/$SUPERNAV_TASK_SET-dry"

supernav run --config "$SUPERNAV_CONFIG" \
  --sweep-id "$SUPERNAV_SWEEP_ID" \
  --task-ids "$SUPERNAV_TASK_ID" --arms default \
  --output-dir "$SUPERNAV_RUNS_ROOT"
```

脚本默认选择生成清单中的第一条任务；也可将 `SUPERNAV_TASK_ID` 设为清单内其他 ID。
dry-run 只准备配置、提示词与 Skills，不加载场景或执行导航。扩大批量前，先检查真实
运行的日志与 RGB 观测。若有其他任务正在运行，可用 `--bridge-port` 指定空闲端口。

省略 `--task-ids` 即可顺序运行已准备的全部任务：

```bash
export SUPERNAV_RUNS_ROOT="$SUPERNAV_WORK_DIR/runs/$SUPERNAV_TASK_SET-batch"
export SUPERNAV_SWEEP_ID=batch
supernav run --config "$SUPERNAV_CONFIG" --arms default \
  --sweep-id "$SUPERNAV_SWEEP_ID" \
  --output-dir "$SUPERNAV_RUNS_ROOT"
```

使用 `--max-episodes 1` 生成的清单仍然只有一条任务。运行器默认跳过已存在的 episode
目录；重新运行请使用新输出目录或 `--run-tag`，并保留已有证据。
复用输出目录时，还应设置新的 `SUPERNAV_SWEEP_ID`，保留之前保存的批次配置与任务快照。

要使用 Learned Executor，请先完成[权重与策略服务配置](habitat.md#learned-executor-weights)，
将 `--template-config` 改为 `configs/experiments/habitat-learned-executor.json`，
生成单独配置后执行相同步骤，并使用独立输出目录。

## 对已保存运行评分

使用模拟器解释器运行评分器，以读取原始场景 NavMesh。
`SUPERNAV_RUNS_ROOT` 应指向待评分的真实单任务或批量运行目录。输入使用该次运行保存的
已解析清单，`visuals_root` 与运行时保持一致：

```bash
"$SUPERNAV_HABITAT_PYTHON" -m supernav score-objectnav \
  --runs-root "$SUPERNAV_RUNS_ROOT" \
  --instructions "$SUPERNAV_RUNS_ROOT/_sweeps/$SUPERNAV_SWEEP_ID/instructions.resolved.json" \
  --scenes-root "$SUPERNAV_HM3D_SCENES_ROOT" \
  --visuals-root "$SUPERNAV_VISUALS_ROOT" \
  --criterion viewpoint_geodesic \
  --success-distance 0.2 \
  --l-source viewpoint_geodesic \
  --out-json "$SUPERNAV_RUNS_ROOT/score-geodesic.json" \
  --out-md "$SUPERNAV_RUNS_ROOT/score-geodesic.md"
```

评分参数设置如下：

| 参数 | 示例值 | 含义 |
| --- | --- | --- |
| `--criterion` | `viewpoint_geodesic` | 使用轨迹终点到最近目标视点的 NavMesh 测地距离 |
| `--success-distance` | `0.2` | 距离阈值，单位为米，按严格小于判定 |
| `--l-source` | `viewpoint_geodesic` | 重新计算起点到目标视点的最短距离，用于 SPL |

SPL 为 `S × L / max(P, L)`，其中 `S` 是独立评分的成功标记，`P` 是累计轨迹长度，
`L` 是起点到目标视点的测地距离；若无法重新计算 `L`，评分器会回退到 episode
存储的距离。SR、SPL 对已评分 episodes 取均值。输出中的 `claim_success` 记录 agent
的完成声明，`success` 和 `spl` 来自独立评分。

评分器按 `task_id` 去重，以 metrics 文件修改时间选择最新一次运行。不同方法、arm
和重复次数使用独立评分目录，并通过 `summary.episodes` 核对已评分的任务数量。

保留原始 episode 文件、任务快照、原生日志和 `<session_id>.trajectory.json`。
评分器从任务快照记录的 source episode 路径读取目标视点，需保持这些文件可读。

遇到轨迹缺失或目标不可达时，检查 `visuals_root`、匹配的 `.basis.navmesh`
与 source episode 路径。可通过
[Web Viewer 指南](web-viewer.md)查看保存的观测与轨迹。
