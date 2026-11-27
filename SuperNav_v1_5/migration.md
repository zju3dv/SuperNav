---
title: SuperNav-1.5 Structured Spatial Memory Migration and Release Plan
form: design
topic: [architecture, implementation, release, spatial-memory]
updated: 2026-11-16
status: active
tags: [supernav-1.5, migration, spatial-memory, memory-nodes, cartographer, captioning]
target_lines: 320
---

# SuperNav-1.5 Structured Spatial Memory Migration and Release Plan

## Summary

Port the committed spatial-memory research implementation from
`habitat-gs/jiarui/memory-slam@042ccf10` into the SuperNav repository in two
stages.

The release is titled **SuperNav-1.5: Structured Spatial Memory For Agentic
Navigation Systems**. It is distinct subsequent work to SuperNav, not a minor
add-on.
Release-facing language calls the complete method **spatial memory** and its
persistent area/door records **memory nodes**. Provenance-oriented internal
identifiers may retain their source names where renaming would reduce transfer
fidelity.

- Stage 1 delivers memory nodes, Cartographer mapping, segmentation,
  occupancy navigation, full non-GT relocalization, and asynchronous Qwen
  captioning.
- Stage 2 adds `supernav stack` orchestration, Dockerfiles, and Docker
  Compose.
- Exclude benchmarks, private GT/NavMesh memory routing, SuperOdom,
  dirty-worktree files, machine-specific paths, tmux launchers, and
  experimental harness code.
- Preserve behavior and artifact formats wherever possible. Record every
  intentional divergence in a provenance manifest.

## Stage 1: Faithful Core Transfer

### Memory nodes and MCP surface

- Port the memory-node store, validation, atomic mutations, SLAM-pose capture, map
  annotations, and agent-facing Skills.
- Introduce sanitized memory-node schema v7:
  - Preserve public area/door nodes, ordering, records, panoramas, and
    `slam_pose`.
  - Read v1-v6 documents and migrate them to v7.
  - Explicitly discard the private `navigation_goal`; never persist or expose
    Habitat/NavMesh coordinates.
- Expose the original MCP contracts:
  - `hab_list_nodes`, `hab_grep_node`, `hab_inspect_node`, and
    `hab_query_node_context`.
  - `hab_add_node`, `hab_upsert_node`, `hab_update_node`,
    `hab_apply_memory_node_mutations`, and `hab_delete_node`.
  - `hab_report_memory_pos` for non-GT relocalization priors.
  - `hab_navigate_to_memory_node_occ` for occupancy-map navigation.
- Do not expose the GT-backed `hab_navigate_to_memory_node`.
- Keep SuperNav's existing bounded spatial-fact ledger unchanged internally.
  SuperNav-1.5 is the method-level spatial-memory design; do not present the
  transferred memory-node system as a minor feature layered onto that ledger.
- SuperNav-1.5 Structured Spatial Memory exclusively owns spatial memory during
  its experiments. Keep the bounded ledger disabled even if a caller
  accidentally enables its legacy flag.

### Cartographer, occupancy, and relocalization

- Port the scan stream, queue/drain lifecycle, native worker, segmentation,
  occupancy planner, memory-node overlays, dual visual/planning maps, and map
  attachments into SuperNav-owned Habitat backend modules.
- Pin upstream Cartographer at
  `877157a0d91788a7700221d87232d412cb3c1ef4` as a submodule and retain the
  authoritative-odometry patch separately for review and reproducible
  application.
- Launch the worker directly as a bridge-owned subprocess. Remove Conda-name,
  tmux, `/nas1`, `/tmp`, and user-home assumptions.
- Support only simulator-evaluator odometry as the source:
  - Default to patched `authoritative` mode.
  - Allow explicit `cartographer_input` fallback using the same simulator
    odometry.
  - Reject SuperOdom and other odometry sources in the released method.
- Add a machine-readable worker capability probe so startup fails when
  `authoritative` is requested from an incompatible build.
- Port the complete non-GT relocalization path: frozen-map delayed matching,
  probability-grid source construction, geometry/conflict gates,
  agent-reported node priors, corrective retries, and diagnostic artifacts.
- Mark relocalization thresholds experimental in configuration, status output,
  and documentation. Retain the evidence required to audit acceptance or
  rejection.
- Preserve the existing scan and artifact protocol: metadata, scans, poses,
  occupancy snapshots, render transforms, pbstreams, relocalization grids,
  candidate metrics, and summaries.

### Captioning and experiment integration

- Port key-frame cadence, asynchronous batching, queue/error records,
  area-grid membership selection, ray-cast fallback, and caption attachment.
- Release the caption service with:
  - `GET /healthz`.
  - `POST /caption`.
  - `POST /caption_batch`.
  - A versioned health payload reporting protocol, model, device, and loaded
    state.
- Default to `Qwen/Qwen3.5-2B`, while accepting a compatible Hugging Face
  identifier or local model path. Never bundle model weights.
- Use inline image bytes for remote/container calls. Retain shared-path
  transport only as an explicit local optimization.
- Treat captions as provenance-bearing visual evidence, never automatic room
  labels or memory-node facts.
- Add a `habitat-spatial-memory` experiment and `spatial-memory` method recipe with
  the memory-node, 2D-SLAM, and spatial-memory-navigation native Skills.
- Add a validated `memory_slam` configuration namespace covering memory-node state,
  Cartographer worker/output, odometry mode, relocalization, segmentation,
  occupancy navigation, and captioning. Resolve default state beneath each
  run instead of global paths.
- Caption-enabled profiles fail before the experiment if service health is
  unavailable. Provide an explicit SLAM-only profile rather than silently
  degrading.
- Update English and Chinese release documentation, package data, dependency
  extras, third-party notices, and release-privacy checks.

## Stage 2: Local and Container Orchestration

### `supernav stack`

Add a stable CLI:

- `supernav stack run`: preflight dependencies, start required managed
  services, wait for health, delegate to the existing experiment runner, and
  tear down only processes it owns.
- `supernav stack up`, `status`, `logs`, and `down`: support long-lived
  development sessions.
- Store PID identity, command, ports, health state, and logs under SuperNav's
  ignored cache directory. Detect stale PIDs and refuse to adopt unrelated
  processes on occupied ports.
- Handle signals and failures with bounded graceful shutdown followed by
  forced cleanup when necessary. Support `--keep-services` for retained
  development services.
- Keep the existing per-episode Habitat bridge lifecycle. Cartographer remains
  a per-session bridge child, not a standalone daemon; the stack preflights
  its binary and capabilities.
- Support service mode `managed` for local subprocesses and `external` for
  Docker Compose or remote endpoints.

### Docker and Compose

- Provide two images:
  - Runtime image: SuperNav, compatible Habitat-GS SDK, patched Cartographer,
    native worker, Open3D/SciPy relocalization dependencies, and experiment
    runner.
  - Caption image: CUDA/PyTorch/Transformers caption service only.
- Provide Compose profiles:
  - `spatial-memory`: runtime plus the required caption service.
  - `slam-only`: runtime with captioning explicitly disabled.
- Mount datasets and model weights externally and read-only; mount run
  artifacts writable. Keep credentials in runtime environment/secrets, never
  image layers or example files.
- Support configurable NVIDIA device assignment through deployment environment
  variables; do not hardcode GPU IDs.
- Add health checks and dependency ordering so the runtime starts only after
  caption readiness.
- Ship Dockerfiles and CI build/smoke validation, but do not publish registry
  images in this stage.

## Test and Release Gates

- Add golden parity fixtures derived from committed `042ccf10` behavior for
  memory-node operations, scan conversion, segmentation, occupancy planning,
  caption selection, and relocalization outputs.
- Cover memory-node v1-v6 migration, v7 atomicity, and proof that `navigation_goal`
  is discarded and never written.
- Test authoritative and fallback odometry, unsupported-source rejection,
  worker queue draining, capability mismatch, map attachment, and subprocess
  cleanup.
- Port synthetic relocalization tests for correct alignment, occupied/free
  conflicts, low support, ambiguous candidates, corrective priors, and
  diagnostic artifacts.
- Test caption health, single/batch requests, inline/path transport, queue
  saturation, timeout, shutdown, cadence, and area-selection behavior.
- Test every released MCP schema, whitelist gate, Skill package,
  occupancy-only node navigation, and absence of the private GT-routing tool.
- Build the patched native worker and run its CTest suite in CI. Verify that
  the patch applies cleanly to the pinned submodule.
- Test `stack` port conflicts, startup failure, signal handling, stale state,
  logs, owned teardown, and external-service mode.
- CI must build both images, validate Compose configuration, run Python/native
  tests, scan for absolute/private paths, and verify wheel/source-export
  exclusions.
- Before release, run one real Habitat smoke covering mapping, memory-node writes,
  captions, frozen-map retrieval, automatic and prior-guided relocalization,
  and occupancy navigation in both authoritative and fallback modes. Audit the
  complete agent-visible payload for GT/NavMesh leakage.

## Implementation TODOs

### Stage 1

- [x] Freeze a source-provenance manifest for every transferred module at
  `042ccf10`; acceptance: source path, commit, hash, destination, and intentional
  deviations are recorded.
- [x] Add memory-node schema v7 and v1-v6 migration; acceptance: all public
  fields survive migration and `navigation_goal` is absent from persisted and
  agent-visible output.
- [x] Register the released memory-node MCP tools and Skill assets;
  acceptance: whitelist/schema tests pass and the GT-backed navigation tool is
  unavailable.
- [x] Add the pinned Cartographer submodule and authoritative patch;
  acceptance: a clean checkout applies the patch and builds the native worker.
- [x] Port the Cartographer stream and session lifecycle; acceptance: scans
  drain before simulator teardown and all expected mapping artifacts are
  written beneath the run directory.
- [x] Implement the worker capability probe and odometry validation;
  acceptance: authoritative and fallback modes pass while unsupported sources
  fail closed.
- [x] Port segmentation and occupancy navigation; acceptance: golden map,
  planner, blocked-path, and replan fixtures match the committed source
  behavior.
- [x] Port frozen-map and prior-guided relocalization; acceptance: synthetic
  positive, conflict, ambiguous, corrective, and rejection cases emit complete
  diagnostics.
- [x] Port the caption worker and Qwen service; acceptance: health, batching,
  queue, timeout, inline-image, and area-selection tests pass.
- [x] Add the `spatial-memory` method and `habitat-spatial-memory` experiment;
  acceptance: configuration resolves without private paths and a dry run
  exposes exactly the intended tools and Skills.
- [x] Make Structured Spatial Memory mutually exclusive with SuperNav's legacy
  bounded spatial-fact ledger in both the release recipe and MCP runtime.
- [x] Complete release-boundary and license review; acceptance: no benchmark,
  GT-routing, SuperOdom, secret, absolute path, or local-only artifact enters
  source/wheel exports.
- [x] Canonicalize the release identity as **SuperNav-1.5: Structured Spatial
  Memory For Agentic Navigation Systems**; acceptance: public recipes, Skills,
  tools, artifacts, and documentation use “spatial memory” and “memory nodes”
  and do not present the work as a minor add-on.
- [x] Match Habitat-GS GPU selection; acceptance: the bridge receives explicit
  `HAB_GPU_DEVICE_ID`, inherits the host CUDA environment, and an
  out-of-sandbox Habitat-GS probe reaches scene loading without an EGL/context
  error.
- [x] Reproduce the native release build from a clean Conda specification;
  acceptance: `environment-cartographer.yml` contains no prebuilt Cartographer,
  the pinned submodule and release patch build into repository-local output,
  the worker links that local archive, and its native test and capability probe
  pass. Verified with the clean `supernav15-release-verify` environment and the
  empty `build/release-smoke-20261118-v9` build tree created from a pristine
  detached Cartographer source worktree.
- [x] Run a source-built Cartographer real-scene smoke; acceptance: the worker
  produced by that clean build consumes real Habitat-GS depth and simulator
  odometry in authoritative mode, drains without drops, and writes both visual
  and planning occupancy maps and PBStreams. Verified on real scene
  `0001_839920/gs_scene`, session
  `712bf793-36bf-457b-aac9-53a421fbdf6a`: 22 simulator steps, 14/14 stream
  records processed, zero drops, clean close, and metadata declaring
  `simulator_evaluator` odometry with `odometry_authority=true`.
- [x] Run the Stage 1 real-scene smoke; acceptance: authoritative and fallback
  runs produce current maps, captions, memory nodes, occupancy routes, and
  auditable relocalization evidence without GT leakage.
  Completed on the `0733_841584` real Habitat-GS scene. The authoritative run
  `stage1-authoritative-v2` (session `eb198026-28d7-4a1f-a871-e78781c53300`)
  achieved with 189/189 records processed, zero drops, two persisted memory
  nodes, dual occupancy maps/PBStreams, 12 ready Qwen captions, and a successful
  occupancy return. The frozen-map fallback run `stage1-fallback-v6` (session
  `ffa0d12c-eae8-4d16-9e50-623ad824fd60`) also achieved, processed 111/111
  records with zero drops, reused both frozen streams and memory nodes, produced
  seven ready captions, completed occupancy navigation, and emitted two
  rejected and one accepted sequence-correlated prior-guided relocalization
  attempts with retained diagnostics. This is execution evidence, not a
  calibrated accuracy claim. An audit of agent-visible tool
  result content found no ground-truth position, goal, path, simulator-state,
  or NavMesh payload fields. The run also confirmed that the earlier EGL failure
  was only a device-restricted sandbox false negative.
- [x] Publish the repository as a versioned SuperNav series; acceptance: the
  root is a bilingual series landing page, `SuperNav_v1/` matches the complete
  pre-migration Git tree byte-for-byte (including modes and submodule gitlink),
  `SuperNav_v1_5/` contains this release, and root CI tests and builds both
  independently installable projects.
- [x] Add the paired-session snapshot workflow; acceptance: `supernav stack
  run-paired` runs exploration and retrieval from distinct benchmark poses,
  hash-audits immutable memory-node and dual-PBStream state, gives retrieval a
  writable memory copy, and supports verified resume. Real-scene validation on
  `interior_0010_840154__hardtask_find_kitchen_range_hood` completed both
  sessions. Retrieval issued `hab_report_memory_pos`, triggered the exact
  source `sparse_node_prior` matcher, and conservatively rejected the proposed
  transform as `insufficient_positive_evidence` instead of applying a false
  localization.
- [x] Re-run single-object, multi-object, and HM3D-OVON release episodes after
  paired-session integration; acceptance: all processes returned zero, all
  simulator sessions closed, all three native Skills were visible with the same
  snapshot hash, and final annotated occupancy maps kept node anchors on mapped
  free space. Single-object and HM3D achieved; the multi-object agent closed
  cleanly as blocked after bounded exploration.

### Stage 2

- [x] Implement `supernav stack run/up/status/logs/down`; acceptance: lifecycle,
  health, signal, port-conflict, stale-state, and owned-cleanup tests pass.
- [x] Add managed/external service modes; acceptance: local execution starts
  the caption process while Compose execution validates the external service
  without spawning a duplicate.
- [ ] Build the runtime container; acceptance: the image contains the pinned
  Habitat/Cartographer stack and passes native/Python preflight without host
  paths.
- [ ] Build the caption container; acceptance: mounted model/Hugging Face cache
  configuration reaches healthy status on a supported NVIDIA host.
- [ ] Add `spatial-memory` and `slam-only` Compose profiles; acceptance: service
  ordering, health checks, volumes, GPU selection, and teardown work from a
  clean checkout.
- [ ] Add container CI; acceptance: both images build, Compose configuration
  validates, and lightweight image smoke checks pass without publishing.
- [ ] Publish local and Compose runbooks in English and Chinese; acceptance:
  each path documents prerequisites, one-command startup, status/log access,
  artifact locations, failure recovery, and shutdown.

### Release gate

- [x] Run the complete focused Python and native test suites with no failures.
- [ ] Build and inspect wheel, source archive, runtime image, and caption image.
- [x] Verify no model weights, datasets, credentials, private paths, or dirty
  source files are included.
- [x] Record the real-scene smoke configuration and evidence without claiming
  calibrated relocalization thresholds.
- [ ] Open a focused SuperNav PR with implementation summary, test results,
  deployment impact, licensing changes, and known experimental boundaries.

## Assumptions

- `habitat-gs/` is evidence and source only; all released SuperNav-1.5 code lands
  in SuperNav.
- Only committed source at `042ccf10` is eligible; existing dirty files remain
  untouched.
- Linux with NVIDIA Container Toolkit is the supported Compose target.
  Datasets, Habitat assets, model weights, and credentials remain external.
- Relocalization is released as experimental, not as a calibrated benchmark
  claim.
- Existing SuperNav experiments and default tool surfaces remain unchanged
  unless the new Structured Spatial Memory recipe is selected.
