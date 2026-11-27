---
name: localnav-pointnav-geo-based-executor
description: Drive the RGBD-backprojected navmesh local navigator — mark a point on anything you can see and the bridge snaps it to the navmesh and walks there with a greedy follower. Covers how to pick the point, how to read what comes back, and what to do on every status.
---

# Local Point Navigation (Geo-Based Executor)

One tool does the walking: **`hab_visual_point_navigate(view="front", point=[x, y])`**.

You mark a point on the picture you are looking at; the bridge back-projects that
pixel through the depth channel into a 3D hint, snaps it to the nearest reachable
navmesh point, and a greedy navmesh follower walks there. No learned policy, no
map shown to you, no coordinates accepted. **One call per hop: the call executes
immediately — there is no preview/confirm round-trip and no confirm_token.**

## How to pick the point

- Pass `view` ("front", "right", "back", or "left") plus the point. Prefer `view` over
  `image_ref`, and choose the view by which of the four latest surround images visibly
  contains your target — if it shows in the front image's right half, that is still
  `view="front"`. Never invent coordinates for a different image than the one you mean.
- Mark **anything you can currently see** in the view. The point is **not restricted
  to floor**: an object body, a wall, a doorframe, or a doorway opening all work,
  because the pixel is back-projected through depth and the bridge then searches a
  small neighborhood around the anchor for a reachable navmesh point. If you can see
  it and depth exists there, it is a legal mark.
- `[x, y]` are normalized to the image: `x` 0 = left edge, 1 = right edge; `y` 0 = top,
  1 = bottom. `[0.5, 0.5]` is dead centre.
- What the bridge does with your mark, in order: sample depth in a small cross around
  the pixel → unproject to a 3D hint → generate candidate standoff points around it
  (default `standoff_m` 0.7) → snap each candidate to the navmesh (candidates whose
  snap is worse than ~0.75 m are rejected) → walk the best reachable candidate with
  the follower, stopping within `goal_radius` (default 0.3 m) of the snapped point.
  Consequences:
  - Marking the **target object itself** (its body or base) is the reliable final
    approach: the follower stops at the standoff point facing it.
  - Marking the **floor past a doorway threshold**, centred in the opening, is the
    reliable way to cross it — the snap puts the goal inside the next room.
  - Marking a wall or doorframe is fine as a proxy — the follower walks to the
    reachable floor nearest that surface.
  - Only sky/ceiling above the horizon line is pointless: no depth, no hint.
- There is **no per-hop step cap and no geodesic distance cap** in this deployment:
  a single call walks as far as the navmesh path requires (default budget 200
  follower steps, effectively unlimited horizon). A long mark is legal, but every
  hop still ends with exactly one fresh surround — prefer chained bounded hops
  through visible proxies over one long shot when you need the intermediate
  observation.
- The camera sits at 1.25 m. Floor closer than about 1.7 m is below the bottom
  edge of the frame — you cannot mark the floor at your own feet or at the base
  of a target you are already standing next to; for that final approach, mark
  the target's body instead. Judge distances from the image itself: floor far
  away bunches just below the horizon, so near the middle line a small vertical
  difference is several meters on the floor — something far away sits close to
  the horizon line, not at the bottom of the frame.
- The point belongs to the **latest** capture. After any movement, turn, or new
  observation, old points and image references are stale — mark a fresh one.

## What comes back

`status` plus a FRESH four-view surround. Every next hop MUST be marked on that
latest surround — never reuse a previous call's point, and never aim from memory.

| status | meaning | what to do |
|---|---|---|
| `reached_visual_point` | the follower arrived at the snapped point near your mark — **not** necessarily at your target; a fresh surround is attached | look at the attached surround: if the target is still metres away, the hop simply moved you closer, so mark the next point |
| `en_route_visual_point` | the hop moved but did not report arrival | inspect the fresh surround and re-mark from where you now stand |
| `navigation_blocked` | the follower reported blocked/unreachable/error mid-hop | inspect the fresh surround; back off by marking the back view, then mark a clearly more open route |
| `self_check_required` (`near_zero_motion`) | the mark was valid but produced near-zero movement — usually the snap landed on your current patch | inspect `overlay_image`; mark a farther point, or accept that you are already on the spot |
| `self_check_required` (`target_too_close`) | the snapped goal is already within reach | you are effectively there; verify on the surround and proceed to the next decision |
| `no_depth_near_anchor` | no valid depth around the marked pixel (sky, glass, sensor hole) | mark a slightly different pixel on the same visible structure, or a nearby surface with clean depth |
| `no_reachable_point_near_anchor` / `snap_too_far` | nothing near the mark snaps onto the navmesh (beyond a wall, through glass, outside the walkable mesh) | the mark is not walkable from here — pick a visible proxy that is (open floor, doorway threshold) |
| `path_exceeds_horizon` | the geodesic path to the snap exceeds the horizon | effectively unreachable in this deployment (horizon is practically unbounded); if you passed an explicit small `horizon_m`, drop it or mark an intermediate proxy |
| `stale_image_ref` / `invalid_image_ref` | the point referred to an old capture | mark a fresh point on the latest surround |
| `missing_visual_anchor` / `point_required` / `bbox_not_allowed` | malformed call: this arm takes `point=[x,y]` only | retry with a normalized `point` on the latest view |

Every movement result also carries `planned_path_m` (geodesic length of the hop),
`displacement_m` (how far you actually moved), `steps_executed`,
`distance_to_visual_target`, and an `overlay_image` showing where your mark
landed. Use them to sanity-check the hop before deciding the next one.

## Working with the rest of the loop

1. You always hold the full surround: `hab_init_scene`, every
   `hab_visual_point_navigate` hop, and every `hab_turn` all return fresh
   front/right/back/left views. Inspect the surround you already have before
   spending any action — a `hab_panorama` or `hab_turn` called just to look
   around is almost always redundant.
2. Choose the direction that plausibly holds what you were asked for and mark
   the point directly on that view — front, right, back, and left are all
   legal. The follower turns itself toward the path before moving off, so do
   **not** spend a `hab_turn` just to face the target first; turning first is
   pure overhead.
3. Mark a point and call `hab_visual_point_navigate`. The call executes
   immediately — the returned surround is the post-move observation.
4. On `reached_visual_point`, inspect the attached surround; if the target is
   not actually there, keep going — mark the next point.
5. Only close the session (`hab_close_session`) at arm's length from the target:
   near enough to touch it (within ~1 m of its surface). Before any achieved
   close you MUST take one final approach hop marked on the target's body or
   base (never the floor well in front of it), even when you believe you are
   already close enough — the follower stops at a standoff point up to ~0.7 m
   from the marked surface, so a hop that merely entered the target's room
   does not count. On the surround that hop returns, verify: the target's base
   is cut by the frame's bottom edge — you can no longer see the floor patch it
   stands on — and its body dominates the view. Seeing it across the room is
   not arriving, and seeing it clearly with its whole base and floor visible in
   front of it is not arriving either: repeat the final approach hop. If that
   hop returns `navigation_blocked` or `self_check_required` with the target
   directly in front of you, you are already at touch distance — that result
   completes the final approach; verify the frame test and close. Measured on
   this benchmark, sessions closed while the target was still 2 m away scored
   as failures more often than not, and stops 0.3–1 m short of the target
   failed the strict arrival check too.
   Close with a not-found report instead if two further hops bring you no closer, or if
   you are confident the target is not in this space.


## Reference

- `references/failure-playbook.md` — worked examples of each status, and how the
  hop numbers look in each case.
