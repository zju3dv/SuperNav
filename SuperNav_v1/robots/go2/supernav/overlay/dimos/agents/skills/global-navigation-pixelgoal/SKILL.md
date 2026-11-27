---
name: global-navigation-pixelgoal
description: Long-horizon Go2 PixelGoal navigation for searching across rooms, tracking openings, recovering from stalls, and stopping on fresh visual evidence.
---

# Global Navigation (Go2 PixelGoal)

These rules guide global navigation through the Go2 PixelGoal tools.
Use MUST, MUST NOT, SHOULD, and MAY literally. Rule IDs remain stable audit labels.

## Load the Needed Rules

- Before leaving the local area, taking a long return, or committing to a long hop,
  read [references/exploration-and-return.md](references/exploration-and-return.md)
  completely.
- When the target or its room is visible only through glass, or no passable entrance is
  visible near the start, read [references/entrance-search.md](references/entrance-search.md)
  completely before searching the perimeter or reporting blocked.
- At the first stall, or before reporting blocked, read
  [references/recovery-and-stop.md](references/recovery-and-stop.md) completely.

## Go2 Contract

- **NM-LIFE-001 — One initialization.** The harness supplies one initial fresh
  front/right/back/left surround. MUST NOT call observation tools just to repeat it.
- **NM-LIFE-002 — PixelGoal rules.** MUST read the complete run-local
  `go2-pixelgoal/SKILL.md` before the first `pixelgoal_navigate`. That skill owns safe
  point selection and result handling; this skill owns global search decisions.
  Its permitted surfaces include traversable floor and stable vertical surfaces
  under the conditions it defines. Apply those rules throughout this skill and its
  references; route selection does not impose a floor-only point restriction.
- **NM-LIFE-003 — Evidence boundary.** MUST perceive and act only through the
  whitelisted DimOS MCP tools and their latest returned images. MUST NOT use shell,
  Python, files, artifacts, maps, coordinates, trajectories, or hidden robot state for
  navigation decisions.
- **NM-LIFE-004 — Formal finish.** There is no session-close tool. MUST answer only
  after the achieved gate holds on the latest surround, or after writing the complete
  blocked audit required by the recovery reference. On operator stop or abnormal
  motion, MUST call `pixelgoal_stop` before answering.

## Target and Observation Gate

- **NM-OBS-001 — Complete target hypothesis.** MUST keep a compact hypothesis containing
  the requested object/region, distinguishing appearance or material, required relations,
  room context, and final visual evidence. A category match alone is insufficient.
- **NM-OBS-002 — Observe-decide-act.** After initialization and every hop, MUST inspect
  front, right, back, and left, including both horizontal edges; test the full target
  hypothesis; census every visible transition; choose one bounded action; and repeat on
  the returned surround. Every completed `pixelgoal_navigate`, including a
  `needs_agent` handoff, returns fresh available views. `exploration_complete`
  calls for a new census, not success or blocked reporting. Use `observe_surround`
  only for missing/stale-image recovery.
- **NM-OBS-003 — Census before decision.** MUST classify each doorway, side opening,
  corridor continuation, and room interior as traversed, meaningfully inspected, or
  new/uncertain. MUST list new/uncertain choices before selecting a proxy using:

  `census: front=...; right=...; back=...; left=... | new=... | frontier=... | next=... | why=...`

Current pixels outrank earlier notes. An empty noted list is not exhaustion. A hop that
empties the list may expose the next junction. MUST NOT report blocked or begin a return
in the same decision as a successful hop; first census its fresh surround.

## Core Navigation Decisions

- **NM-MOVE-001 — Relevance before novelty.** SHOULD prefer current target evidence or a
  target-relevant current route, then a safe unfamiliar opening, a current unentered
  sibling, and finally a bounded return to one exact older noted opening. A clearly
  mismatched room MAY be deferred while a stronger route exists, but MUST stay noted.
- **NM-MOVE-002 — Visible bounded movement.** All locomotion is one
  `pixelgoal_navigate(point=[x, y], direction=...)` call on a safe point selected
  according to `go2-pixelgoal` in the latest view. Reaching a proxy is not completion. Every
  hop or refresh expires old image points.
- **NM-MOVE-003 — Hop gate.** SHOULD choose the farthest safe point allowed by
  `go2-pixelgoal` that advances
  through the selected transition without skipping a more relevant local choice or an
  unrecorded opening. Otherwise use an intermediate visible proxy. Read the exploration
  reference before leaving the local area or committing to a long return.
- **NM-MOVE-004 — Information gain when nothing matches.** When no current view shows
  target evidence, MUST place the point where the hop buys the most new information:
  through the most promising new/uncertain census opening, far enough past the threshold
  that the next surround reveals new area. MUST NOT spend a hop re-showing the same room
  while an unentered opening is visible.

## Real-World Adaptation (0.6 m Go2 rig)

- The cameras sit about 0.6 m above the floor. Chair legs, cables, thresholds,
  and bag straps are real collisions even when they look small; give them
  margin. You cannot see over desks, shelves, or cubicle walls — a blocked
  high view does not mean the floor is blocked; judge floor from the lower
  half of every view.
- Offices have glass partitions and polished floors. Glass shows unreachable
  space (follow entrance-search); floor glare and reflections are not
  obstacles. Dark glass can look like open space — confirm with the other
  three views before marking near it.
- Crossing a doorway: prefer a point clearly past the threshold on the far-side
  floor. Apply `go2-pixelgoal` to any alternative surface; never mark the threshold
  strip itself, and confirm the opening is passable.
- People may walk through the scene; if legs or moving objects cover the
  intended point, wait one observation cycle instead of marking near them.

## Stop Gates

- **NM-TERM-001 — Achieved.** MUST finish achieved only when the latest fresh surround
  shows the complete target, distinguishing attributes, required relations, and matching
  context at human-verifiable distance. Reaching a waypoint is never success by itself.
- **NM-TERM-002 — Blocked.** MUST read the recovery reference and exhaust its bounded gate
  before reporting blocked. MUST write its blocked audit with every clause true before
  answering.
