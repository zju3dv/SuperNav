# Route Fusion Workflow

This reference is the detailed integration workflow for `go2-graph-slam-navigation`.
It does not redefine graph-memory schema or SLAM map legend. Use
`go2-graph-memory-hierarchy` as the canonical source for memory writes and
`go2-2d-slam` as the canonical source for map status, geometry limits, and frontier
definitions.

## Evidence Priority

- Semantic area and door identity: current RGB first, then graph-memory
  panoramas, landmarks, and records.
- Current map location when a live pose exists: the SLAM red-arrow pose first,
  then RGB and graph-memory matching only as supporting evidence.
- Semantic route anchors: graph-memory nodes and directly observed semantic
  transitions.
- Local route geometry and shortcut candidates: 2D SLAM occupancy evidence.
- Immediate collision and traversability: current RGB/depth over stale map
  geometry.
- Unknown-target exploration priority: safe active SLAM frontiers first,
  unentered doorway approaches second, shortcuts toward unmapped space third,
  known numbered-node routes last unless they are waypoints to a frontier.

## Map Status Use

Use `go2-2d-slam` for the canonical status meanings. Route fusion applies those
statuses this way:

- `current`: treat the red-arrow pose as the route origin. If RGB appears to
  match a different graph node, mark that semantic match uncertain instead of
  relocating the agent by language/appearance alone.
- `static`: infer the current node from current RGB plus graph memory before
  using it as the route origin.
- `pending`: proceed from current RGB/depth and graph memory until a map
  arrives.
- `collecting`, `matching`, or `anchoring`: use the static-map workflow; do not
  wait solely for a live pose marker. If status is `collecting` and RGB plus
  graph memory identifies the current stored node, apply
  `go2-graph-memory-hierarchy`'s `go2_report_graph_position(id)` relocation-prior rule.

## Route Decision Record

Before movement, state one concise record:

```text
semantic candidate: dining room -> doorway -> kitchen
SLAM candidate: 1 -> 2 -> 4
next transition: opening beside the window bench, forward-right
decision: commit to the kitchen route
```

For unknown-target exploration, the `SLAM candidate` line must include the
nearest active SLAM frontier from the red-arrow pose. If a visible frontier is
not chosen, the `decision` line must name the concrete blocker or the
higher-priority frontier. Do not select a route through already explored
numbered nodes merely because graph memory has names for them.

Do not invent semantic destinations from occupancy pixels. A shortcut can be
geometrically selected without graph-memory connectivity, but create or update
graph area/door facts only after direct observation.

Avoid RGB-only map relocalization when a live SLAM pose is present. Graph memory
descriptions and panoramas can be too sparse to distinguish similar areas, so an
appearance match that conflicts with the red-arrow map position is a warning
about aliasing, not permission to switch the current node.

## Post-Movement Fusion Loop

At each movement response:

1. Classify direct graph-memory facts before routing. If you entered a new
   functional area, reached a semantic door threshold, observed a new semantic
   door, or verified target evidence, write the corresponding graph update
   before the next navigation action.
2. Apply `go2-graph-memory-hierarchy`'s segmentation-gated area creation rule when
   the latest SLAM map is colored by area segmentation. Never pre-create a node
   for a colored frontier or adjacent colored region before entering it.
3. Perform the `go2-graph-slam-navigation` post-movement observation in front, left,
   right, and back directions.
4. Compare current RGB and SLAM geometry against the expected next transition.
5. Continue the committed route when evidence is consistent. Replan only when a
   listed trigger fires.

## Known Target Commitment

For a known target, commit to one semantic route from the current node or live
SLAM pose toward the target evidence. Prefer `go2_navigate_to_memory_node`,
if available, to the farthest useful node along the committed route, usually
the final target-side node. Use `go2_navigate_to_memory_node` only as a
navmesh-backed comparison/fallback. Use SLAM geometry to reject blocked edges,
choose safe shortcuts, and select the next local transition.

Do not repeatedly read memory at intended intermediate nodes while following
the route correctly. Reads during committed execution are reserved for an
accidental or unexpected encounter, a visual route mismatch, or a blocked or
missing intended transition.

## Unknown Target Exploration

When there is no committed target-node route and the agent is exploring, commit
to the selected frontier until it is entered, blocked by current RGB/depth, or
superseded by a closer safe frontier discovered from the current pose.

Known graph nodes are semantic anchors and possible waypoints to the selected
frontier. They are not default exploration goals.

## Replan Triggers

For known-target routes, replan only when one of these occurs:

- the intended semantic transition is blocked or missing;
- current RGB contradicts the expected landmarks;
- the SLAM map marks the next route as blocked or unsafe;
- an unexpected area is reached;
- a materially clearer SLAM shortcut is available.

For unknown-target exploration, do not treat every new frontier as a global
replan. Continue the selected frontier unless it is entered, blocked/unsafe, no
longer reachable from the current pose, already audited, or superseded by a
closer safe frontier.
