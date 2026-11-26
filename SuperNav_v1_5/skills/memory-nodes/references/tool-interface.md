# Memory-node Tool Interface

Use these MCP tools after `hab_init_scene`.

## `hab_list_nodes()`

List concise summaries of all known nodes. Use before choosing initial semantic
anchors, or after an accidental encounter or route mismatch requires
relocalization. Do not use during normal committed-path execution.

## `hab_grep_node(term)`

Search node names, descriptions, and records. Use it only to choose initial
semantic anchors or replan after an exception.

## `hab_inspect_node(id, return_images=true)`

Inspect a node. Use `return_images=true` when comparing current observation to a
remembered area/door, choosing between similar doors, or validating target
evidence while planning. Do not read it at intended intermediate nodes on a
committed path.

## `hab_report_memory_pos(id)`

Report the memory node ID you believe you are physically at or inside. The tool
returns the full public node details only; it does not return neighboring nodes,
paths, or route suggestions. In memory-node/SLAM relocation runs, this call also
records an agentic Cartographer relocation prior when the node has a stored SLAM
pose.

At episode start, if memory is not empty and current RGB/panorama evidence
matches one stored node, call this before the first navigation action. If the
latest SLAM map reports relocation status `collecting`, call it again after any
new observation or movement that lets you identify the current stored node; do
not wait passively for relocation to accumulate enough points when you can
provide this prior.

After an accepted localization, call it again only if later direct visual
evidence is more confident or contradicts the accepted position, that you want to correct the localization.
Reaching an expected stored node is not by itself a reason to request corrective relocalization.

Outside startup/relocation, use it after an accidental encounter, visual route
mismatch, or another exception requires relocalization. Never call it merely
because you reached an intended known node on a committed path. If the report
does not match current visual evidence, say the position is uncertain and gather
more evidence before relying on that memory.

## `hab_query_node_context(...)`

Read several related memory facts from one consistent node-only snapshot. Prefer
it when choosing initial semantic anchors or replanning after an exception.
Arguments:

```json
{
  "terms": ["laptop", "white blinds", "pink bedroom"],
  "inspect_ids": ["known-target-area"],
  "return_images": true
}
```

- `terms` returns matches for every target or landmark term.
- `inspect_ids` returns full details for known IDs.
- `return_images=false` omits raw `pano_images` from inspected nodes.
- Unknown `inspect_ids` fail the whole call with a memory-node error.
- The response never contains `current_node_id`, `depth`, `paths`,
  `current_context`, or `adjacent_nodes`.

Do not use this tool to guess your current node from a visual frame. First use
current visual evidence and known memory IDs. If an ID or term must be chosen
from a prior response, make that dependency explicit and use the follow-up read.

## `hab_upsert_node(node)`

Add a new node or partially update an existing node. New nodes require `id`,
`kind`, `name`, and `description`; `records` and `pano_images` default to empty
containers.

## `hab_update_node(id, node)`

Partially update mutable attributes of an existing node. Use it to add target
evidence, add doorway observations, correct wrong notes, or refresh better
panorama evidence. Mutable fields are `name`, `description`, `records`, and
`pano_images`. When updating list fields, first inspect the node and preserve
items that should remain.

## `hab_apply_memory_node_mutations(mutations)`

Atomically apply a non-empty ordered array of node mutations. Use it to avoid
separate calls when several independent graph writes are already fully known.
Every mutation is one of:

```json
{"op": "create", "node": {"id": "...", "kind": "area", "name": "...", "description": "...", "records": [], "pano_images": {}}}
{"op": "create", "node": {"id": "...", "kind": "door", "name": "...", "description": "...", "records": [], "pano_images": {}}}
{"op": "upsert", "node": {"id": "...", "records": ["..."]}}
{"op": "update", "id": "...", "node": {"records": ["..."]}}
```

The server validates the final node set and saves nothing if any mutation is
invalid. Do not batch mutations whose facts or IDs depend on a prior tool result.
On success, `affected_nodes` returns the final complete state of every distinct
node created, updated, or upserted by the batch; raw `pano_images` are omitted.
Use these snapshots immediately to verify semantic area boundaries before
continuing.

## `hab_add_node(node)`

Add one complete node and fail if the ID already exists. Prefer
`hab_upsert_node(node)` unless duplicate failure is specifically useful.

## `hab_delete_node(id)`

Delete only accidental duplicates or clearly invalid observations.
