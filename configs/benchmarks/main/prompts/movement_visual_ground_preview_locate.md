Navigation in this arm uses LocateAnything phrase grounding as the primary navigation tool, with direct point navigation for bounded local adjustment. You do not have overlay labels or a map. Inspect the latest `panorama_images` first and prefer a visible object or semantic proxy that LocateAnything can approach.

- `hab_visual_ground_preview(view="front", phrase="...")` - LocateAnything grounding for something visible in the selected panorama view.
  * Use this after visual inspection, not as a blind search over the scene.
  * Follow the LocateAnything skill below for phrase selection.
  * Prefer `view` over `image_ref`. Valid views are `front`, `right`, `back`, and `left`.
  * Choose `view` by which panorama image actually contains the target or proxy. If the target is in the front image, use `view="front"` even when it appears on that image's right or left side.
  * Do not pass world coordinates, object ids, scene-graph refs, hidden labels, or remembered off-screen targets.
  * If LocateAnything finds exactly one reachable candidate, the tool moves directly; inspect the fresh `panorama_images` afterward.
  * If it returns `status: preview_ready`, inspect `overlay_image` and the original panorama, then choose the numbered candidate that best matches the intended target or proxy.
  * To move from `preview_ready`, call `hab_visual_ground_preview(view=..., confirm_token=..., candidate_id=N)` with the same view, returned `confirm_token`, and chosen `candidate_id`.
  * Prefer a reachable candidate with a clear view, reasonable stand-off, and lower `nav_score`; do not pick a near-zero/shortest-path marker just because it is closest.
  * A returned LocateAnything preview is pending state tied to that capture. Reject it by not confirming it; there is no cancel/abandon mode or cancel candidate ID. Never pass `mode="abandon"`, invent an out-of-range `candidate_id`, or send the old token merely to clear state.
  * To replace a rejected preview, call `hab_visual_ground_preview` again without `confirm_token` or `candidate_id`, using a refined/new phrase on the same latest view; the new preview supersedes the pending one. To choose a justified movement or turn instead, call that action normally without the preview token; its fresh panorama invalidates the old token. Never reuse a rejected token.

- `hab_visual_point_navigate(view="front", point=[x, y])` - two-step visual point navigation for a precise, bounded local adjustment on the latest panorama image.
  * Use it for micro-adjustment, a cautious short advance over clearly open floor, exact stand-off/framing, or crossing a doorway after grounding has brought you to its threshold. Do not chain point hops as the primary exploration method when a visible object or semantic proxy can be named.
  * Pick a visible reachable point yourself from the latest panorama image.
  * Use the same view rule: if the target is in the front image, use `view="front"` even when it is on that image's right side.
  * Coordinates are normalized image coordinates: x/y `0.0` is left/top, `1.0` is right/bottom.
  * This arm is point-only; do not pass `bbox`.
  * Step 1: call without `confirm_token`. Treat `status: preview_ready` as a self-check request, not movement.
  * Step 2: inspect `overlay_image` and `selected_anchor`. If the marker is still in the intended region, repeat the same `view` or `image_ref` and `point` with `confirm_token` to move.
  * For objects, click the near/lower visible edge or reachable floor immediately in front of it.
  * For doorways, corridors, floor patches, or proxy regions, click the center of reachable open space.

Decision rules:
- Start each navigation attempt from the latest `panorama_images`.
- First decide whether the final target is visible. If not, choose a visible proxy that moves toward the likely target region.
- Prefer `hab_visual_ground_preview` whenever you can name a visible object or semantically describable region to approach, including proxy anchors used for whole-house exploration.
- Reserve `hab_visual_point_navigate` for micro-adjustment, conservative short advance, exact local framing or stand-off, doorway-threshold crossing, or a reachable location that language cannot express. Return to LocateAnything grounding when a useful visible object or proxy is available.
- Treat LocateAnything as current-image grounding, not as proof that the target exists or is absent from the scene.
- If the best target is visible only in a right/left/back panorama image, use that direction's `view` directly.
- After any movement, turn, or fresh panorama, old `view`, `image_ref`, `confirm_token`, candidate markers, and points expire.

Movement rules:
- Treat each direct or confirmed `hab_visual_ground_preview` call as a full approach to the selected visual candidate unless it returns blocked/error.
- Treat each confirmed `hab_visual_point_navigate` call as one bounded local hop.
- After either movement tool returns, inspect the fresh `panorama_images` before deciding whether the task is satisfied or choosing another visual anchor.
