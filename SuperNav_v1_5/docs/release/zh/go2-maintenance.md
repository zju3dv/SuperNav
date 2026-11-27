[English](../en/go2-maintenance.md) | [简体中文](go2-maintenance.md)

# Go2 源码维护

本文命令在 SuperNav 的 `robots/go2/` 目录执行。Go2 使用该目录下的独立 Python 环境。

## 目录

| 路径 | 用途 |
| --- | --- |
| `dimos/` | 公开 DimOS submodule |
| `supernav/overlay/` | 经审计的 SuperNav-1.5 源码、release 适配、PixelGoal、操作端、空间记忆、SLAM、相机与 GoPro bridge |
| `supernav/patches/` | 对上游已有文件的补丁 |
| `supernav/manifest.json` | 上游版本、批准的实现来源、文件清单、经审计的替换项、补丁顺序和 SHA256 |
| `supernav/audit_snapshot.py` | 与批准实现快照独立比较源码、skill 资源和行为 |
| `deploy/` | 安装与启动脚本、配置模板、依赖锁和系统配置 |
| `runtime/dimos/` | 组装后的运行源码 |
| `.venv/` | 部署环境，以 editable 方式安装运行源码 |

新增文件维护在 `overlay/`；普通上游修改维护在对应 patch。只有 `update_manifest.py` 的 `AUDITED_REPLACEMENTS` 明确列出的源码保真替换才可覆盖上游路径。组装时先应用 patch，再应用经审计的替换，最终生成 `runtime/dimos/`。

## 组装源码

仓库根 `.gitmodules` 和 manifest 固定公开 [dimensionalOS/dimos](https://github.com/dimensionalOS/dimos) commit `79c9ed163ec6b1abd278a9161f0c74f4562f1231`。

```bash
python3 -m supernav.prepare
python3 -m supernav.prepare --check
```

组装工具检查 submodule 版本、工作区状态及输入哈希，导出上游源码，依次应用补丁、新增 overlay 和显式审计的替换，再生成 blueprint 注册表。结果写入 `runtime/dimos/`，上游 submodule 保持原样。

重复执行 `prepare` 会检查已有输出。修改输入后使用 `--rebuild` 重建；构建与启动共用锁，重建前需停止栈。

## 修改代码

先停止栈，保存 `runtime/dimos/` 中需要保留的日志及本地修改。编辑 overlay 或 patch 后执行：

```bash
python3 -m supernav.update_manifest
python3 -m supernav.prepare --rebuild
./deploy/bin/check.sh --static
python3 -m unittest discover -s tests -v
uv pip install --python .venv/bin/python --reinstall-package dimos \
  --no-deps --no-build-isolation -e ./runtime/dimos
```

`update_manifest` 更新输入文件的清单和哈希。重建会替换生成目录，随后重新安装 DimOS 及其原生扩展。功能测试见 [本地测试](go2-testing.md)。

补丁按以下层次组织，可用 `git apply --stat supernav/patches/*.patch` 查看文件改动：

| 补丁 | 内容 |
| --- | --- |
| `01-runtime-compatibility.patch` | 核心运行时、消息和配置兼容性 |
| `02-agent-and-operator.patch` | CLI、MCP 和操作端入口 |
| `03-go2-and-cameras.patch` | Go2 连接、运动和相机支持 |
| `04-mapping-and-navigation.patch` | 地图、局部规划和导航行为 |
| `05-visualization.patch` | 可选 Rerun 图像与目标显示 |
| `06-dependencies.patch` | Unitree WebRTC 和运行依赖 |
| `07-skill-assets.patch` | wheel 和 sdist 的 Markdown 资源 |

## 修改 prompt 与导航规则

Go2 prompt 由 `dimos/agents/mcp/external_harness_codex.py` 组装。空间记忆、2D SLAM 和路线融合的 canonical skill 文本及引用文件，从运行源码顶层的 `skills/` 加载。`supernav/overlay/skills/` 原样提供这些资源；遗漏它们会静默启用不同的内嵌 fallback 文本。PixelGoal 和全局导航 skill 使用模块内嵌文本。PixelGoal 任务会写入全部五份运行时 skill 快照及引用文件。

canonical 文件与加载器必须一起维护，不要用重新编写的 release 说明替换快照的引用文件读取规则或记忆节点创建规则。支持的部署方式是文档中的 editable runtime 安装。修改后更新 manifest、重建、重新安装，并按[保真审计](go2-testing.md#与批准快照比较)比较实际生成的 prompt 与 skill。输入哈希只能检测打包改动，本身不能证明源码保真。

## 更新依赖与上游

修改依赖声明后，先更新 manifest 并重建源码，再使用 uv 0.11.11 生成锁文件：

```bash
./deploy/bin/lock-python.sh --update
```

该命令分别生成构建工具锁、Jetson ARM64 CPU 运行锁和 x86_64 CPU 测试锁。检查版本差异后，按 [安装说明](go2.md) 在新部署目录安装，并运行[本地测试](go2-testing.md)。

升级 DimOS 时，同时更新 submodule gitlink 和 manifest 的 commit，检查补丁应用、blueprint 组成及局部规划器接口。
