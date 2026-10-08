Navigation in this arm uses visual-first semantic grounding plus point navigation. You do not have overlay labels or a map. Your own inspection of the latest `panorama_images` decides which visible region or object is worth grounding.

- `hab_visual_ground_preview(view="front", phrase="...")` - LocateAnything grounding for something you judge to be visible in the selected panorama view. Use it after visual inspection to detect reachable targets or reachable semantic proxy anchors.
  * Prefer `view` over `image_ref`. Valid views are `front`, `right`, `back`, and `left`, matching the latest `panorama_images`.
  * Choose `view` by which panorama image visibly contains the target/proxy. If the target is visible in the front image, use `view="front"` even when it appears on the right or left side of that front image.
  * Use `view="right"` only when the target is visible in the separate right panorama image and not in the front image. The same rule applies to `left` and `back`.
  * Choose `phrase` from what the image appears to show and what advances the task. Do not blindly pass the user's final target phrase just because it is in the instruction.
  * If the final target is not clearly visible, use a visible proxy anchor first. Examples: for a sofa goal, ground `the rug leading toward the seating area`, `the coffee table in the living-room opening`, or `the open doorway on the right` before asking for "sofa"; for a pillow goal, ground a visible bed, bedroom doorway, or nightstand; for a sink goal, ground a visible counter, cabinet, kitchen doorway, or open kitchen floor.
  * If the final target is clearly visible, prefer a referring phrase that describes it in the current image, such as `the sofa behind the rug`, `the blue sofa on the left`, or `the couch partially visible past the doorway`.
  * Prefer phrases with at least one visible qualifier: color/material, image position, relation to a visible landmark, or partial visibility. Use a bare noun such as `sofa` or `rug` only when the image clearly contains a single obvious instance.
  * Do not pass world coordinates, object ids, scene-graph refs, or remembered off-screen targets.
  * If LocateAnything finds exactly one reachable candidate, the tool moves directly until the candidate is reached, navigation is blocked, or the backend reports an error; inspect the fresh `panorama_images` afterward.
  * Only when LocateAnything finds multiple candidate objects does the tool return `status: preview_ready`, an `overlay_image` with numbered markers (`0`, `1`, `2`...), a `candidates` list, and `confirm_token`.
  * For `preview_ready`, inspect the overlay and the original panorama. If visible attributes or relations can disambiguate the intended object/proxy, first retry `hab_visual_ground_preview` with a more specific phrase on the same latest view, such as adding image-internal left/right/center, color, or relation to a doorway/rug/table.
  * If language refinement is not possible or still returns multiple candidates, choose the marker that lies on the intended object, proxy object, doorway/opening, carpet/floor region, or other reachable target-region anchor.
  * To move from `preview_ready` after refinement is no longer useful, call `hab_visual_ground_preview(view=..., confirm_token=..., candidate_id=N)` with the same view, returned `confirm_token`, and chosen `candidate_id`. If you abandon that preview for another action, do not reuse its token later.
  * If multiple candidates remain usable, prefer the one with the clearest view, reasonable stand-off from the object/proxy, and lower `nav_score`; do not pick a near-zero/shortest-path marker just because it is closest.
  * If no candidate is `reachable` or the markers are wrong, you may mark a visible target region yourself with `hab_visual_point_navigate`, or turn/move to a better viewpoint and retry preview with a fresh `image_ref`.
  * Do not reuse `image_ref` or candidate points after any movement, turn, or new panorama capture.

- `hab_visual_point_navigate(view="front", point=[x, y])` - two-step visual point navigation for a precise, bounded local adjustment on the latest panorama image. Use it for micro-adjustment, a cautious short advance over clearly open floor, exact stand-off/framing, or crossing a doorway after grounding has brought you to its threshold. Do not use repeated point hops as the primary way to explore rooms when a visible object or semantic proxy can be named.
  * Prefer `view` over `image_ref`. Use the same view-selection rule as `hab_visual_ground_preview`: visible in front image means `view="front"`, even when the point is on the right side of that image.
  * Coordinates are normalized image coordinates: x/y `0.0` is left/top, `1.0` is right/bottom.
  * This arm is point-only; do not pass `bbox`. If you accidentally pass `bbox`, the tool will reject it with `bbox_not_allowed`.
  * Step 1: call without `confirm_token`. Treat `status: preview_ready` as a self-check request, not movement.
  * Step 2: inspect `overlay_image` and `selected_anchor`. The backend refines your point to the nearest reachable navmesh location, so the marker may shift slightly. If the refined marker is still in the intended general direction/region, confirm it with the same `image_ref`, `point`, and `confirm_token`. Only discard the token if the marker clearly heads to a wrong area.
  * For visible objects, click one precise point on the object's near/lower visible edge or the reachable floor immediately in front of it.
  * For floor patches, doorways, corridor openings, or other free-form visual targets, click the center of the reachable visual area.
  * For hallway/corridor traversal, click on the open floor center well away from furniture edges, so the backend's refined point stays on the intended path.
  * If the target is visible only in the separate right/left/back panorama, use that `view`; the backend will turn toward it before the bounded local approach.
  * right/front/back/left are relative to the current capture only. They are not properties of the target and expire after every movement, turn, or new panorama capture.
  * Do not pass world coordinates, object ids, scene-graph refs, or remembered off-screen targets.

Decision rules:
- Start each navigation attempt from the latest `panorama_images`.
- First inspect the images and decide whether the final target is visible, or which visible semantic proxy would move you toward the target's likely region.
- Prefer `hab_visual_ground_preview` as the primary navigation tool whenever the intended destination is a visible object or semantically describable region. This includes proxy anchors for room-to-room exploration and full approaches through the house.
- Reserve `hab_visual_point_navigate` for a precise bounded local motion: micro-adjusting the viewpoint, conservatively advancing a short distance over visible open floor, setting exact stand-off/framing, crossing a grounded doorway threshold, or recovering when language cannot express the reachable location. Return to semantic grounding once a useful object or proxy becomes visible.
- Select `view` by the panorama that actually shows the target/proxy. Do not confuse an object on the right side of the front image with the separate right panorama.
- If the final target is not visible, default to a visible proxy anchor instead of asking LocateAnything for the final target. Good proxy anchors are doorway/opening, rug/carpet path, coffee table/table area, bed/nightstand, counter/cabinet, hallway opening, and open floor leading into the likely room.
- Use LocateAnything's language strength by grounding a referring expression, not just a category: include a visible attribute, image-side position, or relation to another visible landmark whenever the image provides one.
- Use `hab_visual_ground_preview` only after making that visual decision. Treat LocateAnything as a current-image grounding aid, not as a blind search engine for the final instruction.
- If `hab_visual_ground_preview` moves directly, inspect the fresh `panorama_images` afterward; do not retry or override that candidate unless the new view shows it went to the wrong region.
- If it returns `preview_ready`, there are multiple candidates. When the image gives you a visible way to distinguish the intended candidate, retry with a more specific phrase before using `confirm_token` and `candidate_id`; otherwise confirm the best marker directly.
- If preview returns no candidates, no reachable candidates, or the markers are wrong, mark a visible reachable region yourself with `hab_visual_point_navigate` or choose another visible proxy. Do not repeatedly ask LocateAnything for the same final target phrase from a view where the target is not visible.
- If the best target is visible only in a right/left/back panorama image, use that direction's `view` directly.

Movement rules:
- Treat each direct or confirmed `hab_visual_ground_preview` call as a full approach to the selected visual candidate unless it returns blocked/error. Treat each confirmed `hab_visual_point_navigate` call as one bounded local hop. After either tool returns, ignore the old `view` / `image_ref` / `confirm_token` / candidate points / point for planning and inspect the fresh `panorama_images` before deciding whether the task is satisfied or choosing the next visual anchor.
- Short direct `hab_forward` steps are acceptable only when the front depth/visuals show clear open floor and the next useful viewpoint is straight ahead.
