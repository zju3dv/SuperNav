# Recovery and Stop Rules

Read this file completely at the first stall and before any blocked close.

## Bounded Recovery

- **NM-TERM-010 — Escalation ladder.** When progress stalls: stop repeating the same
  point/phrase/room; reconsider the available visual evidence; enter a visible unfamiliar
  room far enough to expose it; inspect a current unentered sibling; use one bounded
  point adjustment if it reveals a new view; return to the strongest exact older noted
  opening; then re-identify the current location and note materially new siblings.
- **NM-TERM-011 — Continue on information gain.** Continue while bounded actions yield
  new visual information. Do not call a corridor cyclic while any view exposes an
  uninspected room. Empty frontier only means no observed-but-unentered opening is
  noted; leave a terminal room and inspect the nearest multi-opening corridor and all
  image edges.
- **NM-TERM-012 — Standoff recovery before rejection.** If a hop result leaves the
  target wall or feature filling the view, too close or too oblique to verify, first
  regain a readable standoff with one bounded point move — for example turning toward a retreat route, then marking a visible floor point — then inspect again. Never judge, reject, or close
  on a feature from an unreadable close-up; an unverified near miss is not a mismatch.

## Pre-close Audit

- **NM-TERM-020 — Trace audit.** Before blocked close, first write an NM-OBS-003 census
  from the latest tool result, then state:

  `blocked audit: latest hop censused; current openings none; frontier audited; remembered alternatives tested`

  Every clause MUST be true. The audit cannot be the first message after movement. If
  the latest hop just entered a new opening, take at least one observation-bearing
  action before this audit, or the final clause is false.
- **NM-TERM-021 — Evidence, not likelihood.** Semantic likelihood cannot mark an
  opening tested. Relevant bounded alternatives must be exhausted, unsafe, visibly
  impassable, freshly verified duplicates, or attempted and unreachable. One arbitrary
  retreat is not an exhaustion test.

## Blocked Close

- **NM-TERM-030 — Close only on `closed=true`.** Call
  `hab_close_session(outcome="blocked")` once the audit is written and every clause
  holds. A close call that does not return `closed=true` is not a finish: re-read the
  result, refresh the audit from current state, and either continue
  exploring or close again with a fresh audit. MUST NOT answer until a close result
  has `closed=true`.
- **NM-TERM-031 — No silent give-up.** Do not stop calling tools and answer in prose
  while the session is still open. The only terminal states are a close result with
  `closed=true`, or continued exploration.
