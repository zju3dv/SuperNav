# Entrance Search

Read this file completely when the target or its room is visible only through glass, or
when no passable entrance is visible near the spawn.

## Detect the Facade Trap

- **NM-ENTRY-010 — Glass is evidence, not a route.** Treat a target, room, or furniture
  visible through a window or glass wall as useful room evidence but not as a passable
  opening. After one failed or visually impassable approach, MUST stop approaching that
  pane and search for an entrance elsewhere on the same structure.
- **NM-ENTRY-011 — Stable facade anchor.** Retain one distinctive visible part of the
  facade as the return anchor. Choose a visible wall end, corner, walkway continuation,
  fence gap, or other open-floor proxy that advances laterally; do not mark a point on
  the hidden entrance or an off-screen destination.

## Two-Sided Perimeter Search

- **NM-ENTRY-020 — Audit both directions.** From the facade anchor, choose one safe
  lateral direction and follow visible bounded proxies to the next corner or meaningful
  change in the facade. Inspect all four views after every hop. If that side yields no
  passable opening, reacquire the anchor and inspect the opposite lateral direction.
- **NM-ENTRY-021 — Turn corners deliberately.** At each reached wall end or corner,
  move only enough to expose the adjoining facade, then census doors, archways, gaps,
  walkways, and open interior transitions. A corner reached without seeing around it is
  not audited.
- **NM-ENTRY-022 — Entrance outranks the window target.** Once a plausible passable
  opening appears, prefer it over further target-visible glass. Approach the opening
  with a bounded hop, inspect the returned surround, and cross only with fresh visual
  evidence under the normal movement and census rules.

## Exit Condition

- **NM-ENTRY-030 — Bounded completion.** Leave perimeter-search mode when a passable
  entrance is entered or when both lateral directions from the retained facade anchor
  have been meaningfully inspected. MUST NOT close blocked for `no entrance` while
  either side is unaudited, a reached corner has not been viewed around, or a visible
  exterior continuation remains untested.
