- hab_visual_point_navigate(view="front", point=[0.5,0.8]) - two-step visual point navigation toward a visible point in a latest panorama image. First call previews the selected point and returns confirm_token without moving. Second call repeats the same view or image_ref and point with confirm_token to execute one bounded local hop. The default local-hop geodesic cap is 40m. RULES:
  * Prefer view over image_ref. Valid views are front, right, back, and left, matching the latest panorama_images.
  * Choose view by which panorama image visibly contains the target/proxy. If the target is visible in the front image, use view="front" even when it appears on the right or left side of that front image.
  * Use view="right" only when the target is visible in the separate right panorama image and not in the front image. The same rule applies to left and back.
  * Coordinates are normalized image coordinates: x/y 0.0 is left/top, 1.0 is right/bottom.
  * Use point only. Do not pass bbox in this arm.
  * Step 1: call without confirm_token. Treat status preview_ready as a self-check request, not movement.
  * Step 2: inspect overlay_image and selected_anchor. If the marker is correct and reachable, call again with the returned confirm_token and the exact same view or image_ref and point.
  * If the overlay marker is wrong, discard the token and make a new preview with a better point.
  * For visible objects, click one precise point on the object's near/lower visible edge or the reachable floor immediately in front of it.
  * For floor patches, doorways, corridor openings, or other free-form visual targets, click the center of the reachable visual area.
  * If the target is visible only in a separate right/left/back panorama image, use that view directly; do not invent coordinates for a different image.
  * right/front/back/left are relative to the current capture only. They are not properties of the target and expire after every movement, turn, or new panorama capture.
  * Do not pass world coordinates, object ids, scene-graph refs, or remembered off-screen targets.
  * Gated deployments (bridge env HAB_VISUAL_POINT_GEO_BASED_EXECUTOR=1) behave differently: every call executes immediately with no preview/confirm round-trip and no confirm_token, max_steps defaults to 200 with no hard step cap, and the geodesic horizon is effectively unlimited. In that mode treat every call as movement and skip the two-step rules above.

- If hab_visual_point_navigate returns bbox_not_allowed, retry with point=[x,y] only.
- If hab_visual_point_navigate returns preview_ready, do not count that as movement. Inspect overlay_image and either confirm with confirm_token or choose a better point. (Not returned in gated deployments.)
- If hab_visual_point_navigate returns self_check_required with reason near_zero_motion or target_too_close, the visual target was accepted but the refined reachable point produced no meaningful local movement. Inspect overlay_image plus the latest panorama_images, then choose a farther point on reachable floor or on the object's near/lower visible edge.
- If hab_visual_point_navigate returns stale_image_ref, invalid_image_ref, no_depth_near_anchor, no_reachable_point_near_anchor, snap_too_far, or path_exceeds_horizon, treat it as a failed local movement attempt, not proof the target is absent. Inspect the returned overlay_image and latest panorama_images, then choose a better point or use hab_turn / short hab_forward / depth tools.
- Treat each confirmed visual point navigate call as one bounded local hop only. After it returns, ignore the old view/image_ref/direction/point for planning and inspect the fresh panorama_images before deciding whether the task is satisfied or choosing another visual anchor.
- Use short hab_forward steps only when moving straight ahead into clear space is more direct than marking a target region.
- Do not call hab_visual_local_navigate or hab_navigate_wam in this arm.
