---
name: global-navigation-geo-based-executor
description: Mapless long-horizon Habitat-GS navigation for global tasks driven by the RGBD-backprojected navmesh local walker (hab_visual_point_navigate). Use when an episode requires searching across rooms, maintaining target evidence, choosing visible proxies, tracking unexplored openings, recovering from stalls, and stopping only on final visual evidence.
---

# Global Navigation (Geo-Based Executor)

Use MUST, MUST NOT, SHOULD, and MAY literally. Rule IDs are stable audit labels.

## Load the Needed Rules

- Before leaving the local area, taking a long return, or committing to a long hop,
  read [references/exploration-and-return.md](references/exploration-and-return.md)
  completely.
- When the target or its room is visible only through glass, or no passable entrance is
  visible near the spawn, read [references/entrance-search.md](references/entrance-search.md)
  completely before searching the building perimeter or closing blocked.
- At the first stall, or before any blocked close, read
  [references/recovery-and-stop.md](references/recovery-and-stop.md) completely.

These references are part of this skill. Read them from the same run-local skill bundle;
never substitute a user-level copy.

## Episode Contract

- **NM-LIFE-001 — One initialization.** MUST read this file before scene action and call
  the prompt's exact `hab_init_scene` once. MUST NOT use init as observation refresh or
  blindly retry a failed/timed-out init.
- **NM-LIFE-002 — Walker rules.** MUST read the complete run-local
  `localnav-pointnav-geo-based-executor/SKILL.md` before the first `hab_visual_point_navigate`.
- **NM-LIFE-003 — Mapless evidence boundary.** MUST perceive and act only through the
  whitelisted MCP tools and their latest returned images. MUST NOT use shell, Python,
  files, artifact directories, maps, coordinates, goal metadata, trajectories, evaluator
  data, or hidden simulator state.
- **NM-LIFE-004 — Formal finish.** MUST obtain exactly one close result with
  `closed=true`, then answer. A close call without `closed=true` is not a finish and
  MUST be resolved as specified under NM-TERM-002.

## Target and Observation Gate

- **NM-OBS-001 — Complete target hypothesis.** MUST keep a compact hypothesis containing
  the requested object/region, distinguishing appearance or material, required relations,
  room context, and final visual evidence. A category match alone is insufficient.
- **NM-OBS-002 — Observe–decide–act.** After init and every successful hop or turn,
  MUST inspect front, right, back, and left, including both horizontal edges; test the
  full target hypothesis; census every visible transition; choose one bounded action;
  and repeat on its returned surround. Init, every hop, and every turn already return
  the full surround — a `hab_panorama` or `hab_turn` called just to look around is
  almost always redundant.
- **NM-OBS-003 — Census before decision.** MUST classify each doorway, side opening,
  corridor continuation, and room interior as traversed, meaningfully inspected, or
  new/uncertain. MUST list new/uncertain choices before selecting a proxy. Use this
  compact trace form:

  `census: front=...; right=...; back=...; left=... | new=... | frontier=... | next=... | why=...`

- **NM-OBS-004 — HM3D v2 plant label.** When the requested category is `plant`,
  MUST treat indoor decorative plants, potted plants, flower arrangements, and
  vase/planter displays as target candidates, including cases where the vase is
  more visually prominent than the foliage. MUST NOT pursue trees, shrubs, or
  landscaping seen outdoors or through windows/glass; the target is an indoor,
  reachable instance.

Current pixels outrank what you noted earlier. An empty list of noted openings is not
house exhaustion. A hop that empties the list may expose the next junction. MUST NOT
close blocked or start a return in the same decision as a successful hop; first report
the fresh surround with NM-OBS-003.

## Core Navigation Decisions

- **NM-MOVE-001 — Relevance before novelty.** SHOULD prefer current target evidence or a
  target-relevant current route, then a safe unfamiliar opening, a current unentered
  sibling, and finally a bounded return to one exact older noted opening. A clearly
  mismatched room MAY be deferred while a stronger route exists, but MUST stay on the
  noted list. MUST NOT call an unfamiliar opening exhausted only because its guessed
  room type seems unlikely.
- **NM-MOVE-002 — Visible bounded movement.** MUST mark a point only on a target or
  proxy visible in the selected latest view. The mark is not restricted to floor: the
  tool back-projects the pixel through depth and snaps to the nearest reachable
  navmesh point, so the object itself, a wall, a doorframe, or a doorway opening are
  all legal marks. All locomotion is `hab_visual_point_navigate`; each call executes
  immediately — there is no preview/confirm round-trip and no map. Reaching a proxy is
  not completion. Any hop, turn, or fresh panorama expires old image refs and points.
- **NM-MOVE-003 — Hop-length gate.** The follower walks the navmesh shortest path and
  arrives whenever a path exists, but a hop yields exactly one surround at its end: a
  long hop skips every junction observation in between and can carry you past an
  unrecorded opening. SHOULD keep each hop within 4 m. Before any planned hop of at
  least 4 m, MUST compare it with the current census and local unentered siblings. It
  is allowed when it approaches the target itself or a target-relevant current route,
  provided no more relevant local choice or unrecorded opening is skipped. Otherwise
  hop to an intermediate visible proxy instead. Note lower-relevance siblings for
  later. Read the exploration reference before committing.
- **NM-MOVE-004 — Information gain when nothing matches.** When no current view shows
  target evidence, MUST place the point where the hop buys the most new information:
  through the most promising new/uncertain census opening, far enough past the
  threshold that the next surround reveals genuinely new area. MUST NOT spend a hop on
  a spot that re-shows the current room (a nearby wall, the middle of the same floor)
  while an unentered opening is visible.

## Stop Gates

- **NM-TERM-001 — Achieved.** The episode scores success only when you stop
  within 1.0 m of the target instance; a stop that "looks close" at 1.5–3 m
  scores as a failure. MUST close achieved only after completing the final
  approach procedure below — believing you are already close is not a
  substitute for it:
  1. From your current standoff, MUST spend one `hab_visual_point_navigate` hop
     marked on the target itself — its body or its base, not the floor well
     in front of it — even when you believe you are already within ~1 m. The
     follower stops at a standoff point up to ~0.7 m from the marked surface,
     and a hop that merely reached the target's room is not this hop.
  2. Verify on the surround that hop returned: the target's base is cut by
     the frame's bottom edge (you can no longer see the floor patch it stands
     on, nor its whole underside) and its body dominates the view. For a
     target on a raised surface, the surface's near edge under the target is
     at or below the frame's bottom area and the target looms large; if you
     can still see the target's entire base with floor or surface-front
     visible in front of it, you are too far — MUST repeat step 1.
  3. If the final-approach hop returns `navigation_blocked` or
     `self_check_required` with the target directly in front of you, you are
     already at touch distance: that result satisfies step 1 — run the frame
     verification and close. If the hop overshoots into an unreadable
     close-up, apply NM-TERM-012 standoff and re-verify.
  The complete target, distinguishing attributes, required relations, and
  matching context MUST all be verifiable from that final close-range
  surround. Once that gate is met, close promptly — do not keep hunting for
  an even better angle; over-searching after the gate is met risks losing
  the target entirely. The first `outcome="achieved"` close returns
  `achieved_close_audit_required` with a token: immediately retry with that
  `close_audit_token` and an `arrival_confirmation` whose
  `final_approach_done` / `target_base_cut` are true and
  `estimated_distance_m` ≤ 1.0 — any tool call in between invalidates the
  token, and confirmation values that admit the criteria are not met reject
  the close, so keep approaching instead of closing early.
- **NM-TERM-002 — Blocked.** MUST read the recovery reference and exhaust its bounded
  gate before blocked close. Before calling `hab_close_session(outcome="blocked")`,
  MUST write the blocked audit from the recovery reference with every clause true.
  MUST NOT answer until a close result has `closed=true`.
