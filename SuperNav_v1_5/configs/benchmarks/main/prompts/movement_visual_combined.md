- `hab_visual_overlay_navigate(image_ref="...", target_alias="obj_189")` - move a bounded local approach toward a scene-graph object marked on the latest overlay panorama image.
  * **Use this FIRST.** Prefer it whenever the target object OR a nearby object in the target region has a clear `obj_<number>` label in the latest overlay panorama.
  * Always copy `image_ref` from the most recent `panorama_images` row for the image you are using.
  * Always copy `target_alias` exactly from the visible overlay label on that same image, such as `obj_189`.
  * The overlay marks candidates with a yellow point, a leader line, and a yellow `obj_<number>` label with transparent background. It does not draw object bounding boxes.
  * Use `overlay_objlist` to confirm which `obj_<number>` aliases are present in each direction, then inspect the matching image around the yellow point/label to choose the alias.
  * You do NOT need the label to be the exact target object. Choosing a labeled object that is clearly in the same region as the target is a good first hop. Examples:
    - For a sofa goal, a nearby TV, coffee table, floor lamp, or armchair label in the living room area is a useful intermediate target.
    - For a bed goal, a nearby nightstand, pillow, or wardrobe label in the sleeping area is useful.
    - For a kitchen sink or counter goal, a nearby cabinet, refrigerator, or stove label is useful.
  * If multiple labels are in the target region, prefer the one that is deepest along the direction of the target or most clearly part of the target's functional area. Do not choose a label that is clearly in a different room or behind a closed door.
  * The overlay intentionally shows only the top few currently visible candidates per direction to keep the image readable. If no label is on or near the target region, do not guess an alias; fall back to `hab_visual_point_navigate`.
  * The clickable set is temporary per-image and per-capture. Never reuse an alias after any movement, turn, or new panorama capture.

- `hab_visual_point_navigate(view="front", point=[0.5,0.8])` - two-step visual point navigation toward a visible point on the latest overlay panorama image.
  * **Use this ONLY when `hab_visual_overlay_navigate` is not suitable**: the target object has no visible `obj_<number>` alias, or the target is a doorway, corridor opening, floor patch, or other non-object landmark.
  * Prefer `view` over `image_ref`. Valid views are `front`, `right`, `back`, and `left`, matching the latest `panorama_images`.
  * Choose `view` by which panorama image visibly contains the target/proxy. If the target is visible in the front image, use `view="front"` even when it appears on the right or left side of that front image.
  * Use `view="right"` only when the target is visible in the separate right panorama image and not in the front image. The same rule applies to `left` and `back`.
  * Coordinates are normalized image coordinates: x/y `0.0` is left/top, `1.0` is right/bottom.
  * Use point only. Do not pass bbox in this arm.
  * Click on the actual scene feature (e.g., the center of a doorway, a reachable floor patch, or the near/lower visible edge of an unlabeled object). **Do not click on the yellow overlay dot, label text, or leader line.**
  * For hallway/corridor traversal, click on the open floor center well away from furniture edges, so the backend's refined point stays on the intended path.
  * Do not pass `intent` as a free-text description. Only use `intent` if you need the literal value `pass_through` or `inspect`; otherwise leave it out and the default `approach` will be used.
  * Step 1: call without `confirm_token`. Treat `status: preview_ready` as a self-check request, not movement.
  * Step 2: inspect `overlay_image` and `selected_anchor`. The backend refines your point to the nearest reachable navmesh location, so the marker may shift slightly. If the refined marker is still in the intended general direction/region, confirm it with the same `view` or `image_ref`, `point`, and `confirm_token`. Only discard the token if the marker clearly heads to a wrong area.
  * For visible objects, click one precise point on the object's near/lower visible edge or the reachable floor immediately in front of it.
  * For floor patches, doorways, corridor openings, or other free-form visual targets, click the center of the reachable visual area.
  * If the target is visible only in a separate right/left/back panorama image, use that `view` directly; do not invent coordinates for a different image.
  * right/front/back/left are relative to the current capture only. They are not properties of the target and expire after every movement, turn, or new panorama capture.
  * Do not pass world coordinates, object ids, scene-graph refs, or remembered off-screen targets.

Decision rules:
- Start each navigation attempt from the latest `panorama_images` and `overlay_objlist`.
- If the user's target has a clear `obj_<number>` label in the current overlay, use `hab_visual_overlay_navigate`.
- If the target itself is not labeled, but a clearly nearby object in the same functional region has a label, use `hab_visual_overlay_navigate` toward that nearby object as an intermediate hop.
- Only if no label is on or near the target region should you fall back to `hab_visual_point_navigate`.
- If the best target is visible only in a right/left/back panorama image, use that direction's `view` for point navigation, or that row's `image_ref` for overlay navigation; the backend will turn toward it before the local approach.

Movement rules:
- Treat each confirmed navigation call as one bounded local hop only. After it returns, ignore the old `view` / `image_ref` / `target_alias` / `point` / `direction` for planning and inspect the fresh `panorama_images` before deciding whether the task is satisfied or choosing the next visual anchor.
- Do not call `hab_visual_local_navigate` or `hab_navigate_wam` in this arm.
