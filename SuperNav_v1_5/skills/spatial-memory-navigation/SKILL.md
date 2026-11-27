---
name: spatial-memory-navigation
description: Fuse memory-node semantics with 2D SLAM geometry into one committed navigation route.
---

# Structured Spatial Memory Navigation

Use this skill only with `memory-nodes` and `2d-slam`. It is the sole
owner of final route choice, route commitment, and route replanning when both
semantic memory nodes and 2D SLAM are available.

Read `references/route-fusion.md` completely before navigation. Also obey the
provider skills: `memory-nodes` owns memory-node facts and mutation rules;
`2d-slam` owns map legend, status meanings, geometry limits, and frontier
definitions. Do not override their evidence limits.

## Ownership

- Memory nodes supply semantic area/door candidates, stable landmarks,
  panoramas, target evidence, and node-level relocalization anchors.
- 2D SLAM supplies current or static geometric context, local free-space
  feasibility, blocked paths, unexplored boundaries, and shortcut candidates.
- RGB/depth confirms current semantic identity and safe traversability.
- When the SLAM map has a current red-arrow agent pose, that pose is the
  primary evidence for the agent's location on the map. Use RGB/memory-node
  semantic matching as supporting evidence, not as a replacement for the live
  SLAM pose.
- In unknown-target exploration, active SLAM frontiers are navigation goals.
  Numbered memory nodes are anchors for orientation and semantic recall, not
  places to revisit by default.

## Required Post-Movement Observation

State your current status from RGB observation, memory nodes, and memory-node
overlayed SLAM geometry after EACH movement tool before deciding the next
navigation action.

Use this format:

```text
front: <RGB semantic evidence> + <explored/unexplored> + <memory node/unknown>
left: ...
right: ...
back: ...
```

Also state whether the current red-arrow pose is in a colored segmented area
that already has a numbered memory node, or in an unmarked colored area.
You need to re-identify the area node you currently in, since area segmentations and markers are subject to change on each movement.
For an unmarked colored area, state whether it is semantically distinct from nearby
marked areas; apply `memory-nodes` before the next movement when the
area-node creation rule is satisfied.

If the SLAM map shows a reachable unmarked or unexplored area/open boundary in
any direction, include it in this status even when no memory node or semantic
description names it. When an `A<N>` `slam_area_node` includes ready async captions, use them as supporting visual evidence for the frontier or unmarked area. Identify `A<N>` areas with the latest SLAM map, not by old `A<N>` labels from previous versions.
Do not omit a frontier because it appears at the bottom side, or edge of the map image. Describe it by direction from the red arrow, not
by screen position.

## Route Workflow

1. Before any navigation action, satisfy memory-node startup reads and any
   direct-observation write required by `memory-nodes`.
2. Obtain a semantic candidate route through memory-node tools and inspect the
   relevant area nodes as required by `memory-nodes`.
3. Read the latest SLAM map. For `static`, infer the current numbered node by
   comparing current RGB with memory-node panoramas, landmarks, names, and
   records. For `current`, use the red-arrow pose as the route origin even if
   RGB seems to match a different memory node.
4. Compare the semantic memory-node candidate with SLAM geometric candidates and state
   one committed route or one committed exploration frontier.
5. For known target navigation, state one committed route as
   `start ID -> ... -> final ID`. Prefer to navigate directly by
   `hab_navigate_to_memory_node_occ`, if available, to the farthest node,
   usually the final one, along the route. The SuperNav-1.5 method intentionally
   has no GT/navmesh-backed memory-node navigation fallback.
6. For unknown target exploration, choose among safe active frontiers, unentered
   doorway approaches, and shortcuts toward unmapped space. Prefer to navigate directly by `hab_navigate_to_memory_node_occ`, if available, to a chosen frontier `A<N>`. Fallback to `hab_visual_point_navigate` if your frontier is unmarked or far from the marked positions.
7. After every movement, first satisfy memory-node direct-observation writes,
   then use the post-movement observation to update local direction. Do not
   create an independent competing route plan unless a replan trigger fires.

## Replan Triggers

This replan rule ONLY APPLIES when you are navigating to a clear target node.
DO NOT use it when you are exploring for an unknown target.

Replan only when the intended semantic transition is blocked or missing, current
RGB contradicts the expected landmarks, the SLAM map marks the next route as
blocked or unsafe, an unexpected area is reached, or a clearer SLAM shortcut is
available. State the trigger, obtain the minimum memory-node read needed, then
make one new route commitment.

When a live red-arrow pose exists and it disagrees with an RGB-only or
memory-node-description-based location guess, do not jump to the RGB-matched memory
node. Keep the SLAM pose as the current map origin and treat the semantic match
as uncertain until direct movement or observation resolves it.
