---
name: go2-graph-memory-hierarchy
description: Build and reuse concise node-only area/door memory, with detailed rules in references.
---

# Graph Memory Navigation

Graph memory is your own fallible node-only exploration record. It contains only
facts you explicitly write through MCP tools, not a ground-truth map or stored
topology.

## Ownership

This skill owns semantic memory facts, area/door node creation and update rules,
node schema, graph-memory tool usage, mutation batching, startup memory reads,
target-query memory reads, door panorama thresholds, segmentation-gated
area-node creation, and memory-node relocation priors.

When `go2-2d-slam` is active, it owns map legend, map status, geometry limits, and
frontier definitions. When `go2-graph-slam-navigation` is active, provide semantic
anchors to its route-fusion workflow; it owns final route choice, route
commitment, and replans.

## Core Rules

- ALWAYS be conscious of which area or door node you are currently in. MUST
  state your current node after every movement.
- Record only directly observed facts. Never invent room identity, adjacency,
  routes, unseen contents, or door connections.
- Use only two node types: `area` and `door`. Every node has an explicit `kind`.
- An area is a semantically distinct place with a different observed function,
  such as a kitchen, dining room, bedroom, office, bathroom, or corridor. Open
  pathways and viewpoint changes do not create a node, but an open-plan space
  with directly observed distinct functions may contain separate area nodes.
- A door is an observed semantic door, open or closed, that separates two areas.
  Do not create door nodes for open passages, hallway openings, or transitions
  without a semantic door.
- Store objects, target evidence, and search observations in area `records`, not
  as graph nodes.
- Store doorway visibility, destination clues, and explored/unexplored state in
  node `records`. Do not write adjacency lists or remembered paths.
- Existing memory may contain ambiguous descriptions or wrong notes. Use the
  memory as guidance, and update memory when you have better evidence.

## Direct Observation

Explore using target-oriented visual heuristics and inference, not a goal of
exhaustively capturing panoramas or completing the graph. If a room is visible
from its doorway or another viewpoint and the observation gives only a weak
possibility that the target is inside, do not enter solely to capture panorama
evidence or create a graph node. Record the visible doorway clue, then
prioritize routes with stronger target evidence. Enter when direct evidence, a
useful unexplored frontier, or a materially plausible target hypothesis
justifies the movement.

## Door Panoramas

A door node may contain `pano_images` ONLY WHEN the robot is physically centered
in that doorway/threshold and has a clear threshold view. Seeing a door from an
area, corridor, room center, or another doorway is not enough, even when the
door is large, clear, or visually identifiable.

When a door is only visible from the current area, create/update its node
without `pano_images`. After you later reach and center in that exact threshold,
update the same node with door panorama evidence.

## Segmentation-Gated Area Creation

When `go2-2d-slam` / `go2-graph-slam-navigation` provides a colored segmentation map,
use colored regions as geometric gate reference for area-node creation, not as final
semantic rooms. Create an area node ONLY when current RGB/panorama directly supports a useful area description that is functionally different from nearby marked areas, or it is a meaningful sub-space in a room worthwhile for further navigation. Area nodes helps you to navigate back to previously visited areas. Do not create duplicate area nodes inside a region that already has a similar area marker. Do not create area nodes for colored regions, frontiers, or rooms seen only from a distance before entering them.

Create or update a node with pano images passed will update the node's slam position. If you do not want to modify an existing node's slam position, do not update the node with pano images. If you want current slam position to be marked while keeping the existing slam position for further navigation, you can create a new node according to the sub-space rule above.

## Relocation Prior

In graph-memory/SLAM relocation runs, if you can identify the current stored
area or door node at episode start, call `go2_report_graph_position(id)` before the
first navigation action. This report gives DimOS SLAM an agentic relocation
prior; it is not merely a memory read.

If relocation status remains `collecting` after later movement and the current
RGB/panorama evidence newly identifies a stored node, call
`go2_report_graph_position(id)` again before the next navigation action.

## Required Workflow

1. Immediately inspect the current surround at the start of an episode. If graph
   memory is empty and the current place is functionally distinct, create its
   area node before the next navigation action.
2. At the start of each navigation task, if memory is not empty, use
   `go2_query_route_context()` to search target/context terms and inspect
   candidate nodes from one snapshot. This read is mandatory before any graph
   mutation or navigation action.
3. Match the initial surround to an existing area before creating anything;
   never create a replacement start-area or doorway node merely because the
   current session has a new panorama ID.
4. Inspect each candidate node with `go2_inspect_graph_node(id, return_images=true)`
   and use its stable landmarks, panoramas, and records to identify semantic
   anchors.
5. If the current stored node is identifiable in a graph-memory/SLAM relocation
   run, call `go2_report_graph_position(id)` before the first navigation action.
6. While executing a committed path, do not additionally query graph memory at
   intended intermediate nodes if you are following the path correctly.
   Navigate and identify doors/areas from current visual evidence.
7. Graph-memory reads are allowed only after an accidental or unexpected
   encounter, a visual route mismatch, or a blocked/missing intended door.
8. After every movement, create/update directly observed area, semantic door,
   visible-door, and target-evidence facts before the next navigation action.
   Apply the segmentation-gated area creation rule when a colored SLAM
   segmentation map is available.
9. Before close, update target evidence and correct wrong memory.
10. Prefer one `go2_apply_graph_mutations(mutations=[...])` call for independent
    graph creates, upserts, and updates that are already fully specified. Do not
    batch mutations whose facts or IDs depend on an earlier mutation result.

## Clear Viewpoint Rules

- Keep clear of walls, corners, furniture, and door frames when observing.
- Update area `pano_images` when you have a better centered view of the area.
- Movement results automatically save fresh labeled panorama images and return
  an exact `pano_images` mapping. Copy that mapping into the node mutation; do
  not invent or rename references. Later graph reads with `return_images=true`
  hydrate the references back into image blocks.
- For door `pano_images`, stand centered in the exact doorway/threshold with
  both sides visible. Never copy the current area panorama into a door node.
- In long narrow hallways, stand near the centerline, inspect both sides, and
  choose the most promising semantic doors to explore.
- If many semantic doors are close together, add each door node separately with
  a note that identifies it by stable visual anchors, destination clue, and
  exploration state.

## References

- `node-schema.md`: exact node-only schema rules.
- `references/workflows.md`: startup, exploration, hallway, dense-doorway, and
  target-query workflows.
- `references/examples.md`: compact area/door node examples.
- `references/tool-interface.md`: graph-memory MCP tool usage.

Do not navigate, create nodes, or update nodes until all four reference files
have been read in the current episode. Do not read files outside this skill for
scene perception.
