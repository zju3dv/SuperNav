<div align="center">

<h1>SuperNav</h1>

<h3>智能体导航系统系列</h3>

<p><a href="README.md">English</a> | <strong>简体中文</strong></p>

</div>

本仓库收录 SuperNav 系列工作。每项工作位于独立、可单独安装的源码目录中，并提供各自的安装与使用文档。

| | [SuperNav-1.5](SuperNav_v1_5/README.zh-CN.md) | [SuperNav](SuperNav_v1/README.zh-CN.md) |
| --- | --- | --- |
| 重点 | 面向智能体导航的结构化空间记忆 | 基于工具调用的通用智能体导航 |
| 持久空间状态 | 智能体编写、由统一度量地图定位的记忆节点 | 有界空间事实账本 |
| 建图与定位 | 支持仿真器里程计和权威位姿的定制 Cartographer 2D SLAM | 仿真器原生导航状态 |
| 视觉证据 | 可选 VLM 描述服务，将描述附加到记忆节点 | 实时观测与工具结果 |
| 推荐目录 | `SuperNav_v1_5/` | `SuperNav_v1/` |

<div align="center">

<h2>🧭 SuperNav-1.5: Structured Spatial Memory For Agentic Navigation Systems</h2>

<hr>

<p>
  <a href="https://zju3dv.github.io/SuperNav-1.5/"><img src="https://img.shields.io/badge/Project-SuperNav--1.5-e34b4b" alt="SuperNav-1.5 项目主页"></a>
  <img src="https://img.shields.io/badge/Paper-Coming_Soon-lightgrey" alt="论文即将发布">
  <a href="SuperNav_v1_5/README.zh-CN.md"><img src="https://img.shields.io/badge/Code-SuperNav--1.5-black?logo=github" alt="SuperNav-1.5 代码"></a>
  <a href="https://raw.githubusercontent.com/bugByteMe/SuperNav-1.5-assets/main/assets/overview/supernav-1.5-overview.mp4"><img src="https://img.shields.io/badge/Demo-Overview_Video-c79a00" alt="SuperNav-1.5 概览视频"></a>
</p>

<p>
  Jiarui Guo<sup>1</sup> ·
  <a href="https://the0xka1.cc/">Jinkai Zhang</a><sup>1</sup> ·
  <a href="https://yuanhongyu.xyz/">Yuanhong Yu</a><sup>1</sup> ·
  Yongchen Zai<sup>2</sup> ·
  Futang Zhu<sup>2</sup> ·
  <a href="https://csse.szu.edu.cn/pages/user/index?id=531">Ruizhen Hu</a><sup>3</sup> ·<br>
  <a href="https://pengsida.net/">Sida Peng</a><sup>1</sup> ·
  <a href="https://xzhou.me/">Xiaowei Zhou</a><sup>1</sup>
</p>

<p>
  <sup>1</sup> Zhejiang University &nbsp;·&nbsp;
  <sup>2</sup> BYD New Technology Research Institute &nbsp;·&nbsp;
  <sup>3</sup> Shenzhen University
</p>

<video src="https://raw.githubusercontent.com/bugByteMe/SuperNav-1.5-assets/main/assets/overview/supernav-1.5-overview.mp4" poster="SuperNav_v1_5/docs/release/assets/supernav-1.5-overview.webp" controls muted loop width="920"></video>

</div>

SuperNav-1.5 为导航智能体提供结构化空间记忆：由分区占据地图与智能体维护的
**记忆节点**组成。它将场景回忆连接到可执行路线，帮助智能体高效探索、连续到达
多个目标，并跨会话复用记忆。选择该方法后，空间记忆将独占实验的空间状态，原有
的有界空间事实账本会被停用。

[代码与使用](SuperNav_v1_5/README.zh-CN.md) ·
[安装](SuperNav_v1_5/docs/release/zh/spatial-memory.md#安装与构建) ·
[项目主页](https://zju3dv.github.io/SuperNav-1.5/) ·
[概览视频](https://raw.githubusercontent.com/bugByteMe/SuperNav-1.5-assets/main/assets/overview/supernav-1.5-overview.mp4) ·
[方法与结果](https://zju3dv.github.io/SuperNav-1.5/#method) ·
[空间记忆指南](SuperNav_v1_5/docs/release/zh/spatial-memory.md)

<br>

<div align="center">

<h2>🌐 SuperNav: An Agentic Navigation System for Any Task in Any Scene</h2>

<hr>

<p>
  <a href="https://zju3dv.github.io/SuperNav/"><img src="https://img.shields.io/badge/Project-SuperNav-e34b4b" alt="项目主页"></a>
  <a href="https://arxiv.org/abs/2610.12126v1"><img src="https://img.shields.io/badge/Paper-arXiv%3A2610.12126-2f6feb" alt="SuperNav 论文"></a>
  <a href="SuperNav_v1/README.zh-CN.md"><img src="https://img.shields.io/badge/Code-SuperNav-black?logo=github" alt="SuperNav 代码"></a>
  <a href="https://github.com/user-attachments/assets/d415a2ec-89de-4f6d-a2e5-eeec8993d05d"><img src="https://img.shields.io/badge/Demo-Overview_Video-c79a00" alt="概览视频"></a>
</p>

<p>
  <a href="https://the0xka1.cc/">Jinkai Zhang</a><sup>1</sup> ·
  <a href="https://echo636.github.io/">Jingyi Xu</a><sup>1</sup> ·
  <a href="https://yuanhongyu.xyz/">Yuanhong Yu</a><sup>1</sup> ·
  Jiarui Guo<sup>1</sup> ·
  <a href="https://csse.szu.edu.cn/pages/user/index?id=531">Ruizhen Hu</a><sup>2</sup> ·<br>
  <a href="https://person.zju.edu.cn/0093140">Hujun Bao</a><sup>1</sup> ·
  <a href="https://xzhou.me/">Xiaowei Zhou</a><sup>1</sup> ·
  <a href="https://pengsida.net/">Sida Peng</a><sup>1,3†</sup>
</p>

<p>
  <sup>1</sup> Zhejiang University &nbsp;·&nbsp;
  <sup>2</sup> Shenzhen University &nbsp;·&nbsp;
  <sup>3</sup> Causa Robotics<br>
  <sup>†</sup> 通讯作者
</p>

<video src="https://github.com/user-attachments/assets/d415a2ec-89de-4f6d-a2e5-eeec8993d05d" controls width="920"></video>

</div>

原始 SuperNav 使用预训练多模态模型、导航工具、任务进度跟踪和上下文管理完成多种
导航任务，无需针对导航任务微调多模态模型。任务覆盖寻找单个物体、依次访问多个目标，
以及在仿真和真机环境中完成高层需求。

[代码与使用](SuperNav_v1/README.zh-CN.md) ·
[安装](SuperNav_v1/README.zh-CN.md#installation) ·
[项目主页](https://zju3dv.github.io/SuperNav/) ·
[论文](https://arxiv.org/abs/2610.12126v1) ·
[演示](https://github.com/user-attachments/assets/d415a2ec-89de-4f6d-a2e5-eeec8993d05d) ·
[引用](SuperNav_v1/README.zh-CN.md#citation)

## 快速开始

克隆所有版本依赖，然后进入所需工作目录：

```bash
git clone --recurse-submodules https://github.com/zju3dv/SuperNav.git
cd SuperNav/SuperNav_v1_5  # 或 SuperNav_v1
```

请为每项工作创建独立环境。两个目录都提供名为 `supernav` 的 Python 包和命令，
不应安装到同一环境中。SuperNav-1.5 使用 `environment-cartographer.yml` 提供原生
Cartographer 工具链，并使用文档中的 pip 命令安装 Python 运行时；原始 SuperNav
使用其 README 中的 pip 安装方式。

## 许可证

请参阅 [LICENSE](LICENSE) 以及各源码目录中的版本专用声明。
