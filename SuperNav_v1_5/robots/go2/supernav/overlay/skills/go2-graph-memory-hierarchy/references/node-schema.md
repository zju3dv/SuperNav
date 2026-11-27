# Node Schema

Every graph-memory node has this public shape:

```json
{
  "id": "stable-node-id",
  "kind": "area",
  "name": "human readable exact name",
  "description": "Area node. ... or Door node. ...",
  "records": ["Directly observed facts, target evidence, and transition clues."],
  "pano_images": {"front": "pano:...:front"}
}
```

Rules:

- `id` is stable, lowercase, descriptive, and never changed after creation.
- `kind` is exactly `area` or `door`. Use `area` for a functionally distinct
  observed place and `door` only for observed semantic doors that divide areas.
- Do not write `adjacents`, routes, paths, edge lists, or connectivity fields.
- Put target evidence, object observations, search results, doorway visibility,
  destination clues, and explored/unexplored state in `records`.
- Do not create object/furniture/target nodes.
- Do not put episode- or task-specific facts in `records`, such as "I start
  from this area." Prefer durable navigation evidence, such as "Kitchen doorway
  visible beside the marble island; unexplored."

Panorama evidence:

- Area panoramas should come from near the area center when reachable.
- Door panoramas should come from the center of the exact traversable threshold.
- If the doorway is visible only from an area, hall, or another doorway, omit
  `pano_images` from that door write. Add it only after physically reaching and
  centering in the exact threshold.
- Do not replace a useful centered panorama with a worse wall-hugging or corner
  view.
