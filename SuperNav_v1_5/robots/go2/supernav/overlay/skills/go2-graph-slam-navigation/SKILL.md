---
name: go2-graph-slam-navigation
description: Fuse graph-memory semantics with 2D SLAM geometry into one committed navigation route.
---

# Graph-SLAM Navigation

Use this skill only with `go2-graph-memory-hierarchy` and `go2-2d-slam`. It is the sole
owner of final route choice, route commitment, and route replanning when both
semantic graph memory and 2D SLAM are available.

Also obey the
provider skills: `go2-graph-memory-hierarchy` owns graph facts and mutation rules;
`go2-2d-slam` owns map legend, status meanings, geometry limits, and frontier
definitions. Do not override their evidence limits.

## Ownership

- Graph memory supplies semantic area/door candidates, stable landmarks,
  panoramas, target evidence, and node-level relocalization anchors.
- 2D SLAM supplies current or static geometric context, local free-space
  feasibility, blocked paths, unexplored boundaries, and shortcut candidates.
- RGB/depth confirms current semantic identity and safe traversability.
- When the SLAM map has a current red-arrow agent pose, that pose is the
  primary evidence for the agent's location on the map. Use RGB/graph-memory
  semantic matching as supporting evidence, not as a replacement for the live
  SLAM pose.
- In unknown-target exploration, active SLAM frontiers are navigation goals.
  Numbered graph nodes are anchors for orientation and semantic recall, not
  places to revisit by default.

## Required Post-Movement Observation

State your current status from RGB observation, graph memory, and graph-node
overlayed SLAM geometry after EACH movement tool before deciding the next
navigation action.

Use this format:

```text
front: <RGB semantic evidence> + <explored/unexplored> + <graph-memory node/unknown>
left: ...
right: ...
back: ...
```

Also state whether the current red-arrow pose is in a colored segmented area
that already has a numbered memory node, or in an unmarked colored area.
You need to re-identify the area node you currently in, since area segmentations and markers are subject to change on each movement.
For an unmarked colored area, state whether it is semantically distinct from nearby
marked areas; apply `go2-graph-memory-hierarchy` before the next movement when the
area-node creation rule is satisfied.

If the SLAM map shows a reachable unmarked or unexplored area/open boundary in
any direction, include it in this status even when no graph node or semantic
description names it. When an `A<N>` target includes captions, use them only as optional supporting evidence; captions may be absent in DimOS. Identify `A<N>` areas with the latest SLAM map, not by old `A<N>` labels from previous versions.
Do not omit a frontier because it appears at the bottom side, or edge of the map image. Describe it by direction from the red arrow, not
by screen position.

## Route Workflow

1. Before any navigation action, satisfy graph-memory startup reads and any
   direct-observation write required by `go2-graph-memory-hierarchy`.
2. Obtain a semantic candidate route through graph-memory tools and inspect the
   relevant area nodes as required by `go2-graph-memory-hierarchy`.
3. Read the latest SLAM map. For `static`, infer the current numbered node by
   comparing current RGB with graph-node panoramas, landmarks, names, and
   records. For `current`, use the red-arrow pose as the route origin even if
   RGB seems to match a different graph node.
4. Compare the semantic graph candidate with SLAM geometric candidates and state
   one committed route or one committed exploration frontier.
5. For known target navigation, state one committed route as
   `start ID -> ... -> final ID`. Prefer to navigate directly by
   `go2_navigate_to_memory_node`, if available, to the farthest node,
   usually the final one, along the route. Use
   `go2_navigate_to_memory_node` only as a navmesh-backed comparison/fallback.
6. For unknown target exploration, choose among safe active frontiers, unentered
   doorway approaches, and shortcuts toward unmapped space. Prefer to navigate directly by `go2_navigate_to_memory_node`, if available, to a chosen frontier `A<N>`. Fallback to available native visual/relative navigation tools if your frontier is unmarked or far from the marked positions.
7. After every movement, first satisfy graph-memory direct-observation writes,
   then use the post-movement observation to update local direction. Do not
   create an independent competing route plan unless a replan trigger fires.

## Replan Triggers

This replan rule ONLY APPLIES when you are navigating to a clear target node.
DO NOT use it when you are exploring for an unknown target.

Replan only when the intended semantic transition is blocked or missing, current
RGB contradicts the expected landmarks, the SLAM map marks the next route as
blocked or unsafe, an unexpected area is reached, or a clearer SLAM shortcut is
available. State the trigger, obtain the minimum graph-memory read needed, then
make one new route commitment.

When a live red-arrow pose exists and it disagrees with an RGB-only or
graph-description-based location guess, do not jump to the RGB-matched graph
node. Keep the SLAM pose as the current map origin and treat the semantic match
as uncertain until direct movement or observation resolves it.
