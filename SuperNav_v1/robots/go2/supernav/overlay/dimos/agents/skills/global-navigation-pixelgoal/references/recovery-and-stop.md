# Recovery and Stop Rules

Read this file completely at the first stall and before reporting blocked.

## PixelGoal handoffs and safety stops

- `exploration_complete` is a normal map-backend handoff when the exploration
  budget or eligible waypoints run out after one or more legs (up to 1.5 m each).
  An unknown frontier with no new nearby free-space evidence also ends exploration.
  Inspect the fresh surround and choose again; do not apply the stall recovery
  ladder or mark the target/task reached solely because exploration completed.
- `deadline_exceeded` ends the whole hop after the configured limit (60 seconds
  by default); it does not restart the timer for each leg. Inspect fresh views
  before choosing another safe point. A timeout alone is not task completion.
- `no_progress` and `exploration_failed` require a new visual assessment before
  movement. Never blindly retry the same point or blocked doorway. Choose a
  different safe visible route only when the returned images justify it.
- `blocked_pixel_target` means a known obstacle blocks the selected approach or
  stopping point. Use fresh images to choose a different safe point or route.
- At glass, an uncertain/closed door, or a person blocking passage, stop if no
  clearly safe alternative is visible. Operator and physical-safety stops take
  precedence over the recovery ladder and any required blocked audit; do not
  execute extra movement to satisfy those rules.

## Bounded Recovery

- **NM-TERM-010 — Escalation ladder.** When progress stalls: stop repeating the same
  point/room; inspect all four returned views and image edges; enter a visible unfamiliar
  room far enough to expose it; inspect a current unentered sibling; use one bounded point
  adjustment if it reveals a new view; return to the strongest exact older noted opening;
  then re-identify the current location and note materially new siblings.
- **NM-TERM-011 — Continue on information gain.** Continue while bounded actions yield
  new visual information. Do not call a corridor cyclic while any view exposes an
  uninspected room. Empty frontier only means no observed-but-unentered opening is noted;
  leave a terminal room and inspect the nearest multi-opening corridor and image edges.
- **NM-TERM-012 — Standoff recovery before rejection.** If a result leaves the target
  wall or feature filling the view, too close or oblique to verify, regain readable
  standoff with one bounded hop in a back or side view, selecting a safe point
  according to `go2-pixelgoal`, then inspect again.
  Never reject a feature from an unreadable close-up.

## Pre-finish Audit

- **NM-TERM-020 — Trace audit.** Before reporting blocked, first write an NM-OBS-003
  census from the latest result, then state:

  `blocked audit: latest hop censused; current openings none; frontier audited; remembered alternatives tested`

  Every clause MUST be true. The audit cannot be the first message after movement. If the
  latest hop just entered a new opening, inspect its returned surround before this audit.
- **NM-TERM-021 — Evidence, not likelihood.** Semantic likelihood cannot mark an opening
  tested. Relevant bounded alternatives must be exhausted, unsafe, visibly impassable,
  freshly verified duplicates, or attempted and unreachable. One arbitrary retreat is
  not an exhaustion test.

## Blocked Finish

- **NM-TERM-030 — Finish only after the audit.** There is no close-session tool. Answer
  blocked only after the audit is written and every clause holds. If an operator stop or
  abnormal motion is involved, call `pixelgoal_stop` first.
- **NM-TERM-031 — No silent give-up.** Do not stop calling tools and answer blocked while
  safe bounded exploration remains. The terminal states are fresh visual achievement,
  a completed blocked audit, or an explicit operator/safety stop.
