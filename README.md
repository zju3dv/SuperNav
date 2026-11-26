<div align="center">

<h1>SuperNav</h1>

<h3>A Series of Agentic Navigation Systems</h3>

<p><strong>English</strong> | <a href="README.zh-CN.md">简体中文</a></p>

</div>

This repository collects the SuperNav series. Each work lives in its own
self-contained source directory with separate installation and usage guidance.

| | [SuperNav-1.5](SuperNav_v1_5/README.md) | [SuperNav](SuperNav_v1/README.md) |
| --- | --- | --- |
| Focus | Structured spatial memory for agentic navigation | General agentic navigation through tool use |
| Persistent spatial state | Agent-authored memory nodes grounded in a shared metric map | Bounded spatial-fact ledger |
| Mapping and localization | Patched Cartographer 2D SLAM with simulator odometry and authoritative-pose support | Simulator-native navigation state |
| Visual evidence | Optional VLM captions attached to memory nodes | Live observations and tool results |
| Recommended directory | `SuperNav_v1_5/` | `SuperNav_v1/` |

<div align="center">

<h2>🧭 SuperNav-1.5: Structured Spatial Memory For Agentic Navigation Systems</h2>

<hr>

<p>
  <a href="https://zju3dv.github.io/SuperNav-1.5/"><img src="https://img.shields.io/badge/Project-SuperNav--1.5-e34b4b" alt="SuperNav-1.5 project page"></a>
  <img src="https://img.shields.io/badge/Paper-Coming_Soon-lightgrey" alt="Paper coming soon">
  <a href="SuperNav_v1_5/README.md"><img src="https://img.shields.io/badge/Code-SuperNav--1.5-black?logo=github" alt="SuperNav-1.5 code"></a>
  <a href="https://raw.githubusercontent.com/bugByteMe/SuperNav-1.5-assets/main/assets/overview/supernav-1.5-overview.mp4"><img src="https://img.shields.io/badge/Demo-Overview_Video-c79a00" alt="SuperNav-1.5 overview video"></a>
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

SuperNav-1.5 gives navigation agents structured spatial memory: an
area-segmented occupancy map with agent-maintained **memory nodes**. It connects
scene recall to executable routes, helping the agent explore efficiently, reach
successive goals, and reuse memory across sessions. When this method is selected,
it owns spatial state for the experiment and the original bounded spatial-fact
ledger is disabled.

[Code and usage](SuperNav_v1_5/README.md) ·
[Installation](SuperNav_v1_5/docs/release/en/spatial-memory.md#install-and-build) ·
[Project](https://zju3dv.github.io/SuperNav-1.5/) ·
[Overview video](https://raw.githubusercontent.com/bugByteMe/SuperNav-1.5-assets/main/assets/overview/supernav-1.5-overview.mp4) ·
[Method and results](https://zju3dv.github.io/SuperNav-1.5/#method) ·
[Spatial memory guide](SuperNav_v1_5/docs/release/en/spatial-memory.md)

<br>

<div align="center">

<h2>🌐 SuperNav: An Agentic Navigation System for Any Task in Any Scene</h2>

<hr>

<p>
  <a href="https://zju3dv.github.io/SuperNav/"><img src="https://img.shields.io/badge/Project-SuperNav-e34b4b" alt="Project page"></a>
  <a href="https://arxiv.org/abs/2610.12126v1"><img src="https://img.shields.io/badge/Paper-arXiv%3A2610.12126-2f6feb" alt="SuperNav paper"></a>
  <a href="SuperNav_v1/README.md"><img src="https://img.shields.io/badge/Code-SuperNav-black?logo=github" alt="SuperNav code"></a>
  <a href="https://github.com/user-attachments/assets/d415a2ec-89de-4f6d-a2e5-eeec8993d05d"><img src="https://img.shields.io/badge/Demo-Overview_Video-c79a00" alt="Overview video"></a>
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
  <sup>†</sup> Corresponding author
</p>

<video src="https://github.com/user-attachments/assets/d415a2ec-89de-4f6d-a2e5-eeec8993d05d" controls width="920"></video>

</div>

SuperNav uses a pretrained multimodal model, navigation tools, task-progress
tracking, and context management to solve diverse navigation tasks without
navigation-specific fine-tuning of the model. It supports tasks ranging from
finding a single object and visiting multiple targets to fulfilling high-level
requests across simulation and real-robot settings.

[Code and usage](SuperNav_v1/README.md) ·
[Installation](SuperNav_v1/README.md#installation) ·
[Project](https://zju3dv.github.io/SuperNav/) ·
[Paper](https://arxiv.org/abs/2610.12126v1) ·
[Demo](https://github.com/user-attachments/assets/d415a2ec-89de-4f6d-a2e5-eeec8993d05d) ·
[Citation](SuperNav_v1/README.md#citation)

## Getting Started

Clone all versioned dependencies, then enter the work you want to use:

```bash
git clone --recurse-submodules https://github.com/zju3dv/SuperNav.git
cd SuperNav/SuperNav_v1_5  # or SuperNav_v1
```

Create a separate environment for each work. Both directories publish the
Python package and command name `supernav`, so they should not be installed into
the same environment. SuperNav-1.5 uses `environment-cartographer.yml` for the
native Cartographer toolchain and the documented pip installation for the
Python runtime; original SuperNav uses the pip installation in its README.

## License

See [LICENSE](LICENSE) and the version-specific notices in each source directory.
