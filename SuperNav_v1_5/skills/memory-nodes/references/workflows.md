# Memory-node Workflows

## Startup

1. If memory is empty, bootstrap it from the initial surround: create the current
   functionally distinct area before the next navigation action and preserve any
   directly visible semantic-door clue in `records`. If the current place cannot
   yet be classified, take one move to a clearer viewpoint, then create/update it.
2. If memory has nodes, call `hab_query_node_context()` before any graph write or
   navigation action. Use the initial surround to match the current area or door
   to an existing node; a new session's panorama references do not make a new
   area. Create a node only after this read establishes that no existing node
   matches the directly observed place.
3. If current RGB/panorama evidence identifies the current stored node in a
   memory-node/SLAM relocation run, call `hab_report_memory_pos(id)` before the
   first navigation action. This gives Cartographer a node prior.
4. When candidate IDs are already known, call `hab_query_node_context()` once with
   target/landmark `terms` and `inspect_ids`. It returns matching node summaries
   and requested node details from one memory snapshot.
5. Use full landmarks, records, and panoramas from candidate nodes to identify
   semantic anchors. Inspect a door node only when an area's details cannot
   distinguish the intended semantic doorway.
6. State the relevant semantic anchors and visual clues. When
   `spatial-memory-navigation` is active, provide those anchors and wait for SLAM/RGB
   route fusion before entering committed-path mode.

## Committed-Path Execution

While following the route chosen from current RGB/depth and SLAM geometry, do not
make memory-node reads at intended intermediate areas or doors. Follow live
visual landmarks and the committed route. Writes for direct new observations
remain allowed.

Read memory only when an accidental/unexpected encounter, route mismatch, missing
or blocked intended door requires a replan or relocalization check. State the
exception, make the minimum node read needed, decide a new movement route from
current visual/SLAM evidence, and resume committed-path execution.

## Exploration

After each movement, decide whether you are in an existing area, existing door,
new area, new door, or uncertain location.

Before the next navigation action, write memory whenever direct observation
revealed a new functional area, a semantic door threshold, a visible semantic
door missing from the current area's records, or newly verified target evidence.
This write obligation is not optional route replanning and remains active with
`spatial-memory-navigation`. If none of those facts were observed, state that no
graph write is due and continue.

Choose exploration moves from visual heuristics and target-oriented inference.
Do not treat graph coverage or panorama collection as an objective. When a room
can already be observed from its doorway or another viewpoint, do not enter it
solely to capture panoramas or create a node because the target has only a tiny
visual possibility of being there. Preserve the directly visible doorway clue in
the current area's records, and spend movement on stronger target evidence, a
useful unexplored frontier, or a materially plausible target hypothesis instead.

- If the place is existing, update it instead of creating a duplicate.
- If the place is new, create the appropriate `area` or `door` node.
- When a newly observed door is still viewed from an area, create the door node
  without `pano_images`. Do not attach the area's panorama to the door node.
- Add door `pano_images` only after reaching and centering in that exact
  threshold; then update the existing door node.
- If an existing area reveals new doorways missing from `records`, add those
  observations before leaving.
- Do not write connectivity fields. Routes come from current RGB/depth and SLAM
  geometry, not memory-node paths.
- State whether movement remained in the same area, crossed a semantic door,
  entered a new area, or produced an unexpected encounter. Only the unexpected
  case permits an extra memory-node read during committed-path execution.

## Long Narrow Hallways

The following rule applies only when there is not a remembered semantic anchor or
direction you are intentionally reaching.

Treat long narrow hallways as area nodes, but move carefully:

1. Stand near the hallway centerline, clear of both side walls.
2. Create or update the hallway node from the centered view.
3. Record every visible semantic side door as a stable record or door node.
   Keep open side passages in the same area.
4. Stop after each advance and re-check both sides.
   Record any newly visible semantic doors before moving again.
   Choose the most promising semantic doors to explore.
5. Distinguish repeated similar doors with stable visual anchors and features of
   the door itself. You can use SLAM maps to help locate your position if
   provided.

## Dense Doorway Areas

When several openings are close together:

- Add each semantic door as its own door node when directly observed. Do not
  create door nodes for open pathways.
- For each visible door, write records that include distinct visual features
  with stable landmarks, what it appears to lead to, and explored/unexplored
  state.
- Include what is visible beyond the threshold and whether it is explored.
- If you later notice a missing close doorway, update the area immediately.

## Target Query

1. Use `hab_query_node_context()` with target/context terms and candidate IDs to
   select relevant semantic anchors.
2. Inspect candidate nodes with `hab_inspect_node(..., return_images=true)`
   before committing. Use full details to choose stable landmarks and doorway
   clues.
3. Commit to a route from current RGB/depth and SLAM geometry, then execute it
   without memory-node reads at intended intermediate nodes.
4. At each intended semantic anchor, compare the current view with the node's
   stable landmarks, doorway/transition clues, and recorded evidence. Similar
   furniture, doors, or layouts are not identity proof.
5. If the view does not match the intended node, stop following that route. This
   is an exception: read the minimum memory needed to identify the place or
   semantic anchor, then plan a fresh route.
6. Navigate using current visual evidence.
7. If remembered evidence is wrong, update memory.
8. Close achieved only from a current, clear, close visual verification.
