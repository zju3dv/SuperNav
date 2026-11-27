# Exploration, Return, and Movement

Read this file completely before leaving the local area, taking a long return, or acting on
a frontier warning.

## Exploration Order

- **GN-MOVE-010 — Relevant novelty.** Prefer, in order: current complete target evidence; a
  target-relevant current route; a safe unfamiliar room or side opening; a current
  observed-but-unentered sibling; a bounded frontier-directed return to an older unresolved
  branch; familiar space only as a route to one specific unresolved branch.
- **GN-MOVE-011 — Inspect enough.** Crossing a doorway or touching a threshold does not by
  itself inspect a room. Current pixels decide whether main contents and side exits are
  already exposed. If they are not, continue to an interior viewpoint; if they are, do not
  approach an irrelevant interior object merely to clear follow-up. A threshold touch is not
  an exhaustion test.
- **GN-MOVE-012 — Guesses are revisable.** Do not reject a passable unfamiliar bathroom,
  kitchen, public area, dark doorway, or other opening only because its guessed function
  seems low value. When current pixels clearly show a mismatched room and a stronger route
  exists, defer it in working state, but keep it in the frontier. If no stronger choice
  remains or blocked close is being considered, inspect it; audit only when independent
  current evidence supports an accepted disposition.
## Directed Return

- **GN-MOVE-020 — Nearest unresolved junction.** After a wrong room, return only far enough
  to reacquire the nearest unresolved junction. Use visible proxies and verify after each
  bounded move.
- **GN-MOVE-021 — Familiarity brake.** When familiar public or starting-area context
  appears, state that recognition and stop advancing through it. Reorient toward the nearest
  recent view that exposed an unfamiliar side opening. Familiar space is not exhaustion.
- **GN-MOVE-022 — Named destination.** An aimless return is forbidden. Retain the exact
  destination branch ID and literal phrase in working state, but omit `branch_id` from
  visible return proxies until the original opening and sibling context are re-observed.
  Distance and an off-screen branch are not reasons to stop by themselves.

## Warnings and Long Moves

- **GN-MOVE-030 — `frontier_reminder`.** This is a pre-move warning: the proposed preview
  has a path of at least 3 m while local unentered siblings remain. Treat it as a
  confirmation gate. Compare the proposed route with the named siblings and current census;
  discard the token if a more relevant local sibling, shorter reacquisition, or unregistered
  side opening exists.
- **GN-MOVE-031 — `long_move_with_local_frontier`.** This is a post-move audit fact: a move
  of at least 3 m occurred while an unselected local frontier remained. It does not label a
  room or prove failure; inspect the returned surround and reconsider the remaining local
  alternatives.
- **GN-MOVE-032 — Warning-independent gate.** Apply the same long-move check even without a
  warning because the harness cannot warn about an opening that was never registered.

## Visual Movement

- **GN-MOVE-040 — Panorama directions.** The latest front/right/back/left surround covers
  360 degrees. Do not turn 90 or 180 degrees merely to relabel a side/back view. Ground or
  point directly from that view. A small 10–40 degree turn is allowed only to reveal a seam
  or improve final framing; if one turn reveals nothing new, translate or change proxy.
- **GN-MOVE-041 — Grounding.** Prefer `hab_visual_ground_preview` for visible objects,
  furniture, doorways, corridors, and semantic proxies. Inspect overlays and choose the
  semantically correct reachable candidate with clean depth and the greatest useful progress
  toward the selected visible opening. Do not choose a shorter lower-edge candidate merely
  to preserve siblings already recorded in frontier. Ground-preview candidates remain on
  the agent-facing side, so inspect the returned surround before any threshold crossing.
- **GN-MOVE-042 — Point movement.** Use `hab_visual_point_navigate` only for bounded
  threshold crossing, open-floor adjustment, final framing, or recovery when language
  cannot express the point. After a failed threshold point, use fresh pixels to change the
  point, reveal one seam, or change recovery; do not repeat the same phrase, candidate, or
  point without new visual information. Do not chain point hops as primary room-to-room
  exploration.
- **GN-MOVE-043 — Information-gain brake.** After any action that yields no new visual
  information, MUST change the phrase, candidate, point, viewing seam, route, or recovery
  method. A nominally successful micro-move does not justify repeating the same action.
