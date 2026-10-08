# Failure playbook (what to do for each status)

Each entry is written as "what you see → what to do → what not to do". The hop
numbers come from the `planned_path_m` / `displacement_m` / `steps_executed` /
`distance_to_visual_target` fields and the `overlay_image` in the call result —
there is no separate telemetry tool in this setup.

Keep one fact in mind throughout: every `hab_visual_point_navigate` result
already carries a fresh front/right/back/left surround, and the follower turns
itself toward the path before moving off. You almost never need `hab_panorama`
or `hab_turn` just to look or to face something — act on the surround you
already hold.

## `reached_visual_point`

**What you see**: a fresh surround is attached; `displacement_m` matches the
hop you asked for.
**Do**: look at the surround first and confirm the thing you want is really
nearby (clearly visible, occupying a solid share of the frame). Only then
`hab_close_session(outcome="achieved")`.
**Do not**: announce completion the moment you see `reached_visual_point` —
the follower stops at a standoff point near your mark, and the attached view
can still show the target a standoff away. If you are not there, mark a fresh
point from where you stand and keep walking.

## `en_route_visual_point`

**What you see**: the hop executed steps but did not report arrival;
`displacement_m` is smaller than `planned_path_m`.
**Do**: inspect the fresh surround — you are somewhere along the path. Re-mark
the same target from where you now stand (closer, so the snap is easier).
**Do not**: repeat the old point from the old capture; it is stale.

## `navigation_blocked`

**What you see**: the follower gave up mid-path — wedged against furniture or
unable to complete the route.
**Do**: back off by marking a point on the back view (the follower turns
itself around), then from the returned surround mark a new point in a
**clearly more open** direction.
**Do not**: keep marking nearly the same point from the same spot — the
follower walks the same navmesh path and will wedge the same way.

## `self_check_required` + `near_zero_motion`

**What you see**: `steps_executed` is 0 or `displacement_m` is under ~0.15 m,
although the mark was accepted.
**Do**: open the `overlay_image` and check where the marker landed. Most often
the snap put the goal on the patch you already occupy — mark a point genuinely
farther away, on the other side of the room or past the doorway. If the target
you meant is already right in front of you, this result *is* your arrival
signal — run the frame verification and proceed.
**Do not**: re-issue the identical mark expecting a different snap.

## `self_check_required` + `target_too_close`

**What you see**: the snapped goal is within ~0.3 m of where you stand.
**Do**: treat it as "already there": verify on the surround you already hold
and move to the next decision (final-approach check, census, or close).
**Do not**: spend another hop on the same spot.

## `no_depth_near_anchor`

**What you see**: an error saying no valid depth exists around the marked
pixel — typical for sky, windows, mirrors, or thin glossy structures.
**Do**: shift the mark a few pixels onto solid-looking structure next to the
hole (the doorframe instead of the glass, the wall edge instead of the mirror),
or mark the floor in that direction instead.
**Do not**: mark through glass and expect a route — even with depth, the snap
cannot put a goal inside a room the navmesh does not reach from here.

## `no_reachable_point_near_anchor` / `snap_too_far`

**What you see**: candidates near your mark all failed the navmesh snap —
the visible surface is beyond a wall, outside the walkable mesh, or across a
gap the mesh does not cover.
**Do**: pick a visible proxy that is walkable from here — open floor, a
doorway threshold, the near edge of the obstacle — and chain from there.
**Do not**: treat the failure as proof the target is absent; it only means
*this mark* is not walkable from *this spot*.

## `path_exceeds_horizon`

**What you see**: the geodesic path to the snapped point exceeds the hop
horizon. In this deployment the horizon is practically unbounded, so this
should not appear unless an explicit small `horizon_m` was passed.
**Do**: drop any explicit `horizon_m`, or mark an intermediate visible proxy
and continue in two hops.
**Do not**: read it as "target unreachable" — it is a per-hop budget, not a
connectivity verdict.

## `stale_image_ref` / `invalid_image_ref`

**What you see**: an error saying the image reference has expired.
**Do**: mark the point on the freshest surround you hold — the last movement
result usually already carries one. Call `hab_panorama` only if you genuinely
hold no fresh views.
**Do not**: reuse coordinates from the expired capture.

## Judging a bad mark from the overlay

Every result carries an `overlay_image` with the marker where the bridge
interpreted your point. When a hop behaved oddly — wrong direction, short
move, surprising snap — open it before the next call: a marker sitting on the
wrong surface tells you to re-mark; a marker on the right surface with a tiny
`displacement_m` tells you the navmesh route, not the mark, is the problem, so
change the route instead of the pixel.
