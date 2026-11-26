[English](go2-maintenance.md) | [简体中文](../zh/go2-maintenance.md)

# Go2 source maintenance

Run these commands from `SuperNav/robots/go2/`. The Go2 deployment uses its own Python environment in that directory.

## Layout

| Path | Purpose |
| --- | --- |
| `dimos/` | Public DimOS submodule |
| `supernav/overlay/` | Added PixelGoal, operator, surround-camera, and GoPro bridge source |
| `supernav/patches/` | Changes to existing upstream files |
| `supernav/manifest.json` | Upstream version, file list, patch order, and SHA256 hashes |
| `deploy/` | Installers, launchers, configuration templates, dependency locks, and system configuration |
| `runtime/dimos/` | Assembled runtime source |
| `.venv/` | Deployment environment with an editable runtime installation |

Maintain new files in `overlay/` and changes to existing upstream files in the corresponding patch. The assembly tool generates `runtime/dimos/`.

## Assemble the source

The repository's `.gitmodules` and manifest pin public [dimensionalOS/dimos](https://github.com/dimensionalOS/dimos) commit `79c9ed163ec6b1abd278a9161f0c74f4562f1231`.

```bash
python3 -m supernav.prepare
python3 -m supernav.prepare --check
```

The assembly tool checks the submodule version, working tree, and input hashes. It exports the upstream source, applies patches and overlays, and generates the blueprint registry in `runtime/dimos/`. The upstream checkout stays unchanged. Overlay paths must be new files; edit existing paths through patches.

Running `prepare` again checks the existing output. After changing inputs, use `--rebuild`. Assembly and launch share a lock, so stop the stack before rebuilding.

## Change the code

Stop the stack and save any logs or local edits you need from `runtime/dimos/`. Edit the overlay or patch, then run:

```bash
python3 -m supernav.update_manifest
python3 -m supernav.prepare --rebuild
./deploy/bin/check.sh --static
python3 -m unittest discover -s tests -v
uv pip install --python .venv/bin/python --reinstall-package dimos \
  --no-deps --no-build-isolation -e ./runtime/dimos
```

`update_manifest` updates the input file list and hashes. Rebuilding replaces the generated directory; reinstalling then builds DimOS and its native extensions. See [local tests](go2-testing.md) for functional checks.

Use `git apply --stat supernav/patches/*.patch` to inspect changes by layer:

| Patch | Contents |
| --- | --- |
| `01-runtime-compatibility.patch` | Core runtime, messages, and configuration compatibility |
| `02-agent-and-operator.patch` | CLI, MCP, and operator entry points |
| `03-go2-and-cameras.patch` | Go2 connection, motion, and cameras |
| `04-mapping-and-navigation.patch` | Mapping, local planning, and navigation |
| `05-visualization.patch` | Optional Rerun image and goal displays |
| `06-dependencies.patch` | Unitree WebRTC and runtime dependencies |
| `07-skill-assets.patch` | Markdown resources in wheels and source distributions |

## Change navigation rules

Maintain navigation rules in `supernav/overlay/dimos/agents/skills/`:

- [go2-pixelgoal/SKILL.md](../../../robots/go2/supernav/overlay/dimos/agents/skills/go2-pixelgoal/SKILL.md): selecting points from four camera views, executing a hop, and handling its result.
- [global-navigation-pixelgoal/SKILL.md](../../../robots/go2/supernav/overlay/dimos/agents/skills/global-navigation-pixelgoal/SKILL.md): target search, route selection, and completion. Its `references/` directory covers entrances, exploration and return, recovery, and stopping.

The PixelGoal skill defines point-selection rules. After editing Markdown, update the manifest, rebuild, and reinstall as above. `mcp/skill_assets.py` copies the skills and references into each task's `workspace/.agents/skills/` directory for that task.

## Update dependencies and upstream

After changing dependency declarations, update the manifest and rebuild the source, then generate locks with uv 0.11.11:

```bash
./deploy/bin/lock-python.sh --update
```

This generates separate locks for build tools, Jetson ARM64 CPU runtime dependencies, and x86_64 CPU tests. Review the version changes, follow the [installation guide](go2.md) in a new deployment directory, and run the [local tests](go2-testing.md).

When upgrading DimOS, update both the submodule gitlink and the manifest commit. Check patch application, blueprint composition, and local-planner interfaces.
