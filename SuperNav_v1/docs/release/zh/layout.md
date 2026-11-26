[English](../en/layout.md) | [简体中文](layout.md)

# 目录布局

从 [README](../../../README.zh-CN.md) 进入项目。运行职责见[架构说明](architecture.md)，
实验组合规则见[配置指南](configuration.md)。

```text
SuperNav/
├── src/
│   ├── supernav/             入口、实验、执行、方法、后端、评测、展示
│   └── habitat_contract/     独立于仿真器的共享协议
├── configs/
│   ├── experiments/          命名实验方案
│   ├── agents/               客户端与超时设置
│   ├── models/               模型名称与推理设置
│   ├── backends/             仿真后端设置
│   ├── methods/              导航方法
│   ├── benchmarks/           通用提示模板
│   ├── runtime/              运行策略
│   ├── evaluation/           评测策略
│   ├── providers/            Provider catalog 与凭证变量示例
│   ├── viewers/              Rerun 等展示配置
│   └── local/                仅本机使用的配置覆盖
├── robots/
│   └── go2/                  独立 map-backed PixelGoal 部署包
│       ├── dimos/            固定 commit 的公开 DimOS submodule
│       ├── supernav/         组装工具、manifest、overlay/ 与 patches/
│       ├── deploy/           安装器、启动器、配置模板与依赖锁
│       ├── tests/            部署与资源测试
│       └── runtime/          生成源码与本机运行产物（忽略）
├── skills/                   导航 Skills 源，运行时冻结快照
├── scripts/                  开发、验证与审计辅助脚本
├── tests/                    自动化测试及 fixtures
├── docs/
│   ├── release/
│   │   ├── en/               英文指南，含 go2*.md 部署教程
│   │   └── zh/               对应的简体中文指南
├── data/
│   ├── embodiments/          小型机体参数快照
│   ├── test_assets/          小型输入资产
│   └── runs/                 运行证据，可链接至外部存储
├── .cache/                   构建与测试缓存
├── README.md                 英文项目介绍
└── README.zh-CN.md            简体中文项目介绍
```

仿真运行时位于 `src/`，通过 `supernav` 命令树操作。
[Go2 部署包](go2.md) 以 `robots/go2/` 为独立安装根目录，
使用自己的 `.venv` 和依赖锁。组装命令需在该目录执行；其中的 `supernav/`
存放组装工具和 DimOS 改动，与仿真 Python 包独立。公开 submodule 加上 overlay
和 patches 组装为 `robots/go2/runtime/dimos/`。

Rerun 实现在 `src/supernav/web/rerun_viewer.py`，配置在 `configs/viewers/rerun.yaml`。

## 输入与生成产物

| 位置 | 用途 | 是否提交 |
| --- | --- | --- |
| `robots/go2/dimos/` | 公开 DimOS submodule | 固定 gitlink |
| `robots/go2/supernav/`、`robots/go2/deploy/`、`robots/go2/tests/` | Go2 集成源码、安装器、锁文件、模板与测试 | 是 |
| `docs/release/en/go2*.md`、`docs/release/zh/go2*.md` | Go2 双语部署教程 | 是 |
| `robots/go2/runtime/`、`robots/go2/.venv/`、`robots/go2/deploy/config/*.env` | 生成运行目录、环境、日志与真实机器人配置 | 否 |
| `configs/benchmarks/` | 通用提示模板 | 是 |
| `data/embodiments/` | 机体参数快照 | 是 |
| `data/test_assets/` | 小型 warmup 图片等输入 | 是 |
| `tests/fixtures/` | 合成测试输入与夹具说明 | 是 |
| `data/runs/` | 实验运行证据 | 否 |
| `data/nav_artifacts*/` | 本地导航日志、观测与测量产物 | 否 |
| `configs/local/` | 本机部署设置与外部任务清单路径 | 否 |
| `.cache/` | 可再生成的构建、测试和打包缓存 | 否 |
| `src/*.egg-info/`、`.venv/` | 包元数据与本机 Python 环境 | 否 |

当前运行输出的默认位置应在 `data/runs/` 下。若该位置是外部存储链接，保留链接及已有
运行证据。场景、模型权重、任务清单和评测 GT 单独准备，再通过环境变量或显式本机配置定位。
真实基准任务和 GT 从公开源码导出及安装包中排除。

## 新文件放在哪里

- Go2 改动维护在 `robots/go2/supernav/overlay/` 和 `patches/`，更新 manifest 后重建 runtime。导航规则维护在部署包内的 Markdown skills 中。
- 功能实现放在 `src/supernav/` 的对应职责模块，CLI 保持薄路由。
- 命名实验放 `configs/experiments/`，复用设置放对应片段，通过本机覆盖选择外部任务清单。
- 通用提示模板放 `configs/benchmarks/<suite>/`；真实任务、标注和 GT 保存在私有存储中，并从发行产物中排除。
- 开发与验证脚本放 `scripts/`；可复用逻辑进入规范包。
- `tests/fixtures/` 使用合成测试输入，也可在测试临时目录中生成输入；真实任务数据保持私有。
- 公开指南在 `docs/release/en/` 和 `docs/release/zh/` 中维护对应页面。
- 运行证据放 `data/runs/` 或显式选择的输出目录，保留每次运行的原始内容和记录路径。

根目录只保留项目级文件。`pyproject.toml` 定义依赖、包、分发资源与 CLI，`setup.cfg`
指定构建位置。构建与测试缓存统一放在 `.cache/` 下。

## 导出发布快照

完成发布修改的审阅与提交后，导出当前提交：

```bash
python scripts/export_release_source.py \
  --output .cache/release/supernav-source.tar.gz
```

该导出与根 Python wheel/sdist 覆盖仿真运行时，排除 `robots/go2/` 及其
submodule 配置。Go2 需通过完整 Git 克隆获取，再按其
[安装指南](go2.md)初始化 `robots/go2/dimos`。

归档包含仿真源码和双语指南。解压到新目录后，按安装教程配置仿真运行时。

凭据保存在本机客户端文件或环境变量中。忽略与打包规则覆盖环境文件、认证文件
和客户端配置目录；空的 `.env.example` 模板随项目发布。
