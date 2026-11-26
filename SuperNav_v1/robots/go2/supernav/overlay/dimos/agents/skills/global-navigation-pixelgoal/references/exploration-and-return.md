# Exploration, Return, and Movement

Read this file completely before leaving the local area, taking a long return, or
committing to a long hop.

## Exploration Order

- **NM-MOVE-010 — Relevant novelty.** Prefer, in order: current complete target evidence;
  a target-relevant current route; a safe unfamiliar room or side opening; a current
  observed-but-unentered sibling; a bounded return to an older noted-but-unresolved
  opening; familiar space only as a route to one specific unresolved opening.
- **NM-MOVE-011 — Inspect enough.** Crossing a doorway or touching a threshold does not
  inspect a room. Current pixels decide whether main contents and side exits are exposed.
  If not, hop to an interior viewpoint; if they are, do not approach an irrelevant object
  merely to clear follow-up. A threshold touch is not an exhaustion test.
- **NM-MOVE-012 — Guesses are revisable.** Do not reject a passable unfamiliar room or
  opening only because its guessed function seems low value. When pixels clearly show a
  mismatch and a stronger route exists, defer it but keep it noted. If no stronger choice
  remains or blocked is being considered, inspect it.

## Directed Return

- **NM-MOVE-020 — Nearest unresolved opening.** After a wrong room, return only far
  enough to reacquire the nearest noted-but-unresolved opening. Choose visible proxies
  according to `go2-pixelgoal` and verify against the fresh surround after each hop.
- **NM-MOVE-021 — Familiarity brake.** When familiar public or starting-area context
  appears, state that recognition and stop advancing through it. Reorient toward the
  nearest recent view that exposed an unfamiliar side opening. Familiar space is not
  exhaustion.
- **NM-MOVE-022 — Named destination.** An aimless return is forbidden. Keep the exact
  destination phrase in mind. A long return is a chain of visible bounded proxy hops;
  re-verify the direction against noted openings after each hop.

## Long Hops

- **NM-MOVE-030 — Current-census gate.** Before a long hop, compare it with the named
  siblings and current census. Take a shorter safe proxy instead whenever a more
  relevant local sibling, shorter reacquisition, or unrecorded side opening exists.
- **NM-MOVE-031 — Passed-opening audit.** If a returned surround shows an unrecorded or
  unentered opening was passed, note it before moving on. A long hop does not label a
  room or prove failure.

## Visual Movement

- **NM-MOVE-040 — Surround directions.** The latest front/right/back/left surround covers
  360 degrees, and every view is legal for `pixelgoal_navigate`. Do not spend an action
  turning to relabel a side/back view; pass its direction directly.
- **NM-MOVE-041 — Point choice.** Follow the complete point-selection rules in
  `go2-pixelgoal`, including its permitted stable vertical surfaces, exclusions,
  and safety margins. Choose the farthest safe point those rules permit in the
  selected travel direction. Use a shorter proxy when farther choices are occluded,
  ambiguous, unsafe, or would skip a more relevant local opening.
- **NM-MOVE-042 — Chained hops.** Room-to-room travel is a chain of bounded hops through
  visible proxies. After a failed threshold hop, use fresh pixels to change the point or
  route; do not repeat the same point without new visual information.
- **NM-MOVE-043 — Information-gain brake.** After any action that yields no new visual
  information, MUST change the point, viewing seam, route, or recovery method. A nominally
  successful micro-hop does not justify repetition.
- **NM-MOVE-050 — Loop brake.** If the fresh surround after a hop shows a place already
  visited from a similar heading, MUST NOT reverse direction on the next decision.
  Choose one recorded unentered opening and chain two to three hops through it before
  re-evaluating. A turnaround needs fresh visual evidence that the forward route is
  blocked or exhausted; a familiar-looking view is not evidence of exhaustion.
