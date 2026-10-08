# Exploration, Return, and Movement

Read this file completely before leaving the local area, taking a long return, or
committing to a hop of at least 4 m.

## Exploration Order

- **NM-MOVE-010 — Relevant novelty.** Prefer, in order: current complete target
  evidence; a target-relevant current route; a safe unfamiliar room or side opening; a
  current observed-but-unentered sibling; a bounded return to an older noted-but-
  unresolved opening; familiar space only as a route to one specific unresolved
  opening.
- **NM-MOVE-011 — Inspect enough.** Crossing a doorway or touching a threshold does not
  by itself inspect a room. Current pixels decide whether main contents and side exits
  are already exposed. If they are not, hop to an interior viewpoint; if they are, do
  not approach an irrelevant interior object merely to clear follow-up. A threshold
  touch is not an exhaustion test.
- **NM-MOVE-012 — Guesses are revisable.** Do not reject a passable unfamiliar
  bathroom, kitchen, public area, dark doorway, or other opening only because its
  guessed function seems low value. When current pixels clearly show a mismatched room
  and a stronger route exists, defer it, but keep it on the noted list. If no stronger
  choice remains or blocked close is being considered, inspect it; call it exhausted
  only when independent current evidence supports that.

## Directed Return

- **NM-MOVE-020 — Nearest unresolved opening.** After a wrong room, return only far
  enough to reacquire the nearest noted-but-unresolved opening. Hop between visible
  proxies and verify against the fresh front RGB observation after each hop.
- **NM-MOVE-021 — Familiarity brake.** When familiar public or starting-area context
  appears, state that recognition and stop advancing through it. Reorient toward the
  nearest recent view that exposed an unfamiliar side opening. Familiar space is not
  exhaustion.
- **NM-MOVE-022 — Named destination.** An aimless return is forbidden. Keep the exact
  destination phrase in mind. The follower needs a visible point, so a long return is a
  chain of bounded proxy hops — re-verify the direction against your noted openings
  after each hop. Distance and an off-screen opening are not reasons to stop by
  themselves.

## Long Hops

- **NM-MOVE-030 — Warning-independent gate.** Apply the NM-MOVE-003 check on every
  planned hop of at least 4 m: compare it against the named siblings and the current
  census, and take a shorter proxy hop instead whenever a more relevant local sibling,
  shorter reacquisition, or unrecorded side opening exists.
- **NM-MOVE-031 — Passed-opening audit.** After any hop whose returned front RGB observation shows
  you passed an unrecorded or unentered opening, note it in your census before moving
  on. A long hop does not label a room or prove failure; inspect the returned front RGB observation
  and reconsider the remaining local alternatives.

## Visual Movement

- **NM-MOVE-040 — Front-only marks.** Only the latest front image is a legal
  marking view. Use `hab_turn` to face an off-screen target before marking it.
  Choose observation directions and timing yourself.
- **NM-MOVE-041 — Point choice.** Mark the floor in front of the target for
  reliability, or the target itself when its floor point is occluded. Choose the point
  with a clean approach and the greatest useful progress toward the selected visible
  opening. Do not choose a shorter point merely to preserve siblings you already
  noted. Inspect the returned front RGB observation before any threshold crossing.
- **NM-MOVE-042 — Chained hops.** Room-to-room travel is a chain of bounded hops
  through visible proxies, not one long shot. After a failed threshold hop, use fresh
  pixels to change the point, reveal one seam, or change recovery; do not repeat the
  same point without new visual information.
- **NM-MOVE-043 — Information-gain brake.** After any action that yields no new visual
  information, MUST change the point, phrase, viewing seam, route, or recovery method.
  A nominally successful micro-hop does not justify repeating the same action.
