---
name: memory-nodes
description: Build and reuse concise node-only area/door memory, with detailed rules in references.
---

# Memory Nodes

Memory nodes are your own fallible node-only exploration record. They contain only
facts you explicitly write through MCP tools, not a ground-truth map or stored
topology. Use it after `hab_init_scene`.

MUST DO: At skill startup, before any navigation action or memory-node write,
read every file listed in `References`. These references are part of the skill,
not optional background. After reading them, follow both this concise file and
the referenced details for the rest of the episode.

## Ownership

This skill owns semantic memory facts, area/door node creation and update rules,
node schema, memory-node tool usage, mutation batching, startup memory reads,
target-query memory reads, door panorama thresholds, segmentation-gated
area-node creation, and memory-node relocation priors.

When `2d-slam` is active, it owns map legend, map status, geometry limits, and
frontier definitions. When `spatial-memory-navigation` is active, provide semantic
anchors to its route-fusion workflow; it owns final route choice, route
commitment, and replans.

## Core Rules

- ALWAYS be conscious of which area or door node you are currently in. MUST
  state your current node after every movement. Do not associate your current
  position with a node only by node name, which is misleading. Identify the node
  by its description. Check its panoramas if you are unsure.
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
  as memory nodes.
- Store doorway visibility, destination clues, and explored/unexplored state in
  node `records`. Do not write adjacency lists or remembered paths.
- Existing memory may contain ambiguous descriptions or wrong notes. Use the
  memory as guidance, and update memory when you have better evidence.

## Direct Observation

Explore using target-oriented visual heuristics and inference, not a goal of
exhaustively capturing panoramas or completing the graph. If a room is visible
from its doorway or another viewpoint and the observation gives only a weak
possibility that the target is inside, do not enter solely to capture panorama
evidence or create a memory node. Record the visible doorway clue, then
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

When `2d-slam` / `spatial-memory-navigation` provides a colored segmentation map,
use colored regions as the geometric gate for area-node creation, not as final
semantic rooms. Create an area node ONLY after the robot has entered a colored
region that has no numbered memory-node area marker on it, and ONLY when
current RGB/panorama directly supports a useful area description that is
functionally different from nearby marked areas. Because segmentation may be too
fragmented, do not create a node for an unmarked colored fragment when a nearby
marked node describes the same function, landmarks, and semantic place. Do not
create duplicate area nodes inside a colored region that already has a numbered
area marker. Do not create area nodes for colored regions, frontiers, or rooms
seen only from a distance before entering them.

## Relocation Prior

In memory-node/SLAM relocation runs, if you can identify the current stored
area or door node at episode start, call `hab_report_memory_pos(id)` before the
first navigation action. This report gives Cartographer an agentic relocation
prior; it is not merely a memory read.

If relocation status remains `collecting` after later movement and the current
RGB/panorama evidence newly identifies a stored node, call
`hab_report_memory_pos(id)` again before the next navigation action.

After relocation reaches `localized`, issue a corrective report only when new
direct visual evidence shows that the previously reported position was wrong.
Do not call `hab_report_memory_pos` while relocation status is `matching` or
`anchoring`: localization is already in progress, and another report would be
redundant. Wait for that attempt to settle before reporting a correction.

## Required Workflow

1. Immediately after `hab_init_scene`, inspect the current surround. If the
   memory-node store is empty and the current place is functionally distinct, create its
   area node before the next navigation action.
2. At the start of each navigation task, if memory is not empty, use
   `hab_query_node_context()` to search target/context terms and inspect
   candidate nodes from one snapshot. This read is mandatory before any memory-node
   mutation or navigation action.
3. Match the initial surround to an existing area before creating anything;
   never create a replacement start-area or doorway node merely because the
   current session has a new panorama ID.
4. Inspect each candidate node with `hab_inspect_node(id, return_images=true)`
   and use its stable landmarks, panoramas, and records to identify semantic
   anchors.
5. If the current stored node is identifiable in a memory-node/SLAM relocation
   run, call `hab_report_memory_pos(id)` before the first navigation action.
6. While executing a committed path, do not additionally query memory nodes at
   intended intermediate nodes if you are following the path correctly.
   Navigate and identify doors/areas from current visual evidence.
7. Memory-node reads are allowed only after an accidental or unexpected
   encounter, a visual route mismatch, or a blocked/missing intended door.
8. After every movement, create/update directly observed area, semantic door,
   visible-door, and target-evidence facts before the next navigation action.
   Apply the segmentation-gated area creation rule when a colored SLAM
   segmentation map is available.
9. Before close, update target evidence and correct wrong memory.
10. Prefer one `hab_apply_memory_node_mutations(mutations=[...])` call for independent
    memory-node creates, upserts, and updates that are already fully specified. Do not
    batch mutations whose facts or IDs depend on an earlier mutation result.

## Clear Viewpoint Rules

- Keep clear of walls, corners, furniture, and door frames when observing.
- Update area `pano_images` when you have a better centered view of the area.
- For door `pano_images`, stand centered in the exact doorway/threshold with
  both sides visible. Never copy the current area panorama into a door node.
- In long narrow hallways, stand near the centerline, inspect both sides, and
  choose the most promising semantic doors to explore.
- If many semantic doors are close together, add each door node separately with
  a note that identifies it by stable visual anchors, destination clue, and
  exploration state.

## References

- MUST READ `references/node-schema.md`: exact node-only schema rules.
- MUST READ `references/workflows.md`: startup, exploration, hallway, dense-doorway, and
  target-query workflows.
- MUST READ `references/examples.md`: compact area/door node examples.
- MUST READ `references/tool-interface.md`: memory-node MCP tool usage.

Do not navigate, create nodes, or update nodes until all four reference files
have been read in the current episode. Do not read files outside this skill for
scene perception.
