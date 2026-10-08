- hab_visual_overlay_navigate(image_ref="...", target_alias="obj_189") - move a bounded local approach toward a scene-graph object marked on the latest overlay panorama image. RULES:
  * Always copy image_ref from the most recent panorama_images row for the image you are using.
  * Always copy target_alias exactly from the visible overlay label on that same image, such as obj_189.
  * The overlay marks candidates with a yellow point, a leader line, and a yellow obj_<number> label with transparent background. It does not draw object bounding boxes.
  * Use overlay_objlist to confirm which obj_<number> aliases are present in each direction, then inspect the matching image around the yellow point/label to choose the alias.
  * The overlay intentionally shows only the top few currently visible candidates per direction to keep the image readable. If the desired target is not listed, explore or turn and request a fresh overlay.
  * Prefer overlay navigation whenever a clear alias marks the target object or a useful visible object on the route toward the target.
  * The clickable set is still temporary per-image and per-capture. Never reuse an alias after any movement, turn, or new panorama capture.
  * If the target is in right/left/back, use that direction's image_ref and alias directly; the backend will turn toward it before the local approach.
  * right/front/back/left are relative to the current capture only. They are not target properties and expire after every movement, turn, or new panorama capture.
  * Do not guess aliases, invent aliases, use hidden labels, use coordinates, or describe the target in free text.

- If the requested target is not among the visible obj_<number> overlay labels, use hab_turn or a short hab_forward exploration step, then inspect the fresh overlay panorama.
- If hab_visual_overlay_navigate returns invalid_image_ref, stale_image_ref, unknown_target_alias, target_not_currently_visible, unresolved_local_target, or navigation_blocked, treat it as a failed local movement attempt, not proof the goal is absent. Inspect the latest overlay panorama before choosing another alias.
- Treat each overlay navigate call as bounded movement toward the selected object's nearby standoff, not proof of final task completion. After it returns, ignore the old image_ref/target_alias/direction and inspect the fresh panorama_images before deciding whether the task is satisfied or choosing another obj_<number>.
- Use short hab_forward steps only when no suitable overlay alias is visible and moving straight ahead into clear space is the best exploration action.
- Do not call hab_visual_local_navigate, hab_visual_point_navigate, or hab_navigate_wam in this arm.
