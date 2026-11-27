---
name: 2d-slam
description: Read and use the live Cartographer 2D occupancy map returned after movement.
---

# Live 2D SLAM Map

When this skill is active, movement responses include `2d_slam_map` and inline
one image titled **Latest 2D SLAM occupancy map** or **Latest 2D SLAM area
segmentation map**. It is a scan-derived local map for orientation and
short-horizon exploration, not a semantic object map.

This skill owns map interpretation only. When memory nodes is active, apply
`memory-nodes` for memory reads/writes and
`spatial-memory-navigation` for final route choice, route commitment, and replans.

## Reading the Map

- The red arrow with a yellow outline is the current odometry-projected robot
  pose rendered into the latest Cartographer map frame. Its tip is the
  estimated forward heading; its base is the current estimated position. When
  this pose exists on a `current` map, prioritize it as the agent's location on
  the SLAM map.
- Map-up is an arbitrary SLAM coordinate direction, not the direction you are
  facing. Determine forward, left, right, and behind from the red arrow,
  geometry, and memory nodes; do not treat the top edge of the image as forward.
- In memory-node/SLAM runs, the attached map may be a colored
  area-segmentation overlay. Solid colored regions are SLAM-segmented
  traversability/evidence areas. They may be well explored or only scanned from
  a distance with incomplete shape. Treat one colored region as a geometric area
  candidate for exploration bookkeeping, not as proof that the agent has
  inspected it in detail. A color is not a semantic room label by itself; use
  RGB/panorama evidence to name or describe it.
- Large black numbers with a white outline are memory-node IDs. The text
  response's `2d_slam_map.memory_nodes` field maps each displayed ID to its full
  node name. Only nodes that have both saved panorama evidence and a
  scan-matched SLAM pose are numbered on the map.
- White `A<N>` labels with a black outline are live unmarked SLAM area targets.
  The text response's `2d_slam_map.slam_area_nodes` field maps each displayed
  label to an ephemeral `slam_area_<N>` node id. These are geometric
  free-space targets regenerated from the latest segmentation, not semantic
  memory nodes. If the area center is blocked in the planning occupancy
  map, the listed target is snapped to a nearby clear planning cell.
  `A<N>` labels are temporary and are subject to change after every movement.
  Do not assume that an `A<N>` label is the same area as a previously seen `A<N>` label.
- `slam_area_nodes` may include async `captions` selected from past observations,
  which serves as visual hint for what area `A<N>` looks like.
  The caption can help you to decide which area to explore next.
- A colored region that already contains a numbered memory-node area node is
  represented in memory. Do not create another area node for that same colored
  region merely because the viewpoint changed. A colored region with an `A<N>`
  label is only a possible navigation/exploration target until RGB/panorama
  evidence supports creating or updating semantic memory. Segmentation can be
  too fragmented; an unmarked colored region is only a possible new area.
- Each label is sized for a map footprint. Labels can overlap local geometry,
  so use the surrounding map structure to identify the stored anchor point or
  generated area center.
- Black cells are occupied evidence: walls, furniture, or other depth returns.
  Do not plan to move through them.
- Light gray cells are unknown or unclassified space. They are not evidence of
  traversable space and should be explored from a visible safe approach.
- Frontiers are unmarked or unexplored geometric areas/open boundaries adjacent
  to observed traversable space. They may have no special color marker.
- White lines are segmentation boundaries between colored regions. They are not
  frontiers by themselves.
- Use the red arrow and nearby geometry; never infer absolute coordinates from
  pixels.

## Map Status

- `current`: use the red arrow, segmented nearby geometry, unmarked or
  unexplored frontier candidates, numbered anchors, and `A<N>` area targets.
  Treat the red-arrow pose as the map-route origin.
- `static`: trust the persisted numbered geometry, but no real-time robot pose
  marker is available. If memory nodes is active, infer the current numbered
  node from current RGB/panorama evidence plus memory-node panoramas, stable
  landmarks, names, and records.
- `pending`: no map image is attached. Use current RGB/depth and wait for a
  later movement response instead of inventing map geometry.
- `collecting`, `matching`, or `anchoring`: relocation has not finished. Use
  the static-map workflow and do not wait solely for a live pose marker. If
  memory nodes is active and current RGB/panorama evidence identifies the stored
  node you are in, `memory-nodes` owns the
  `hab_report_memory_pos(id)` relocation-prior rule.

## Frontiers

A frontier is a reachable unmarked or unexplored area/open boundary where
observed traversable space leads into unknown, unclassified, or not-yet-audited
space. Examples include map-edge openings, unlabeled corridors, unentered
colored regions, and visible open passages without memory-node markers. It may
have no special color marker. Black wall-like lines, gray unknown space by
itself, and white segmentation boundaries are not frontiers.

Treat a frontier as active exploration debt until it is entered, seen to be
blocked/unsafe in RGB/depth, or explicitly superseded by a closer frontier. If
you skip a visible frontier, state the concrete reason: `blocked`, `unsafe`,
`not reachable from current pose`, `already audited`, or `lower priority than
<specific frontier>`.

## Geometry Limits

- Do not invent semantic destinations from occupancy pixels. The map has no
  object labels, room names, or door semantics.
- Confirm all semantic targets, doorways, and final-task evidence in RGB
  panorama images. Use depth/current observation for immediate traversability.
- Treat black/white boundaries and the red arrow as estimates. If the map and
  current RGB/depth observation disagree, trust the current observation and
  proceed cautiously.
- If the red-arrow pose conflicts with a location guess from RGB or
  memory-node descriptions, trust the red-arrow map position for route
  geometry and treat the semantic match as uncertain. Do not override a live
  SLAM pose with an RGB-only or graph-description-only match unless direct
  movement evidence confirms the map pose is wrong.
