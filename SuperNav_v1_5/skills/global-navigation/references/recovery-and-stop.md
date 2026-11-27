# Recovery and Stop Rules

Read this file completely at the first stall and before any blocked close.

## Bounded Recovery

- **GN-TERM-010 — Escalation ladder.** When progress stalls: stop repeating the same
  phrase/candidate/room; inspect all four views and image edges; enter a visible unfamiliar
  room far enough to expose it; inspect a current unentered sibling; use one bounded point
  adjustment if it reveals a new view; return to the strongest exact older frontier; then
  run place recognition and register materially new siblings.
- **GN-TERM-011 — Continue on information gain.** Continue while bounded actions yield new
  visual information. Do not call a corridor cyclic while any view exposes an uninspected
  room. Empty frontier only means the bounded ledger has no observed-but-unentered branch;
  leave a terminal room and inspect the nearest multi-opening corridor and all image edges.
- **GN-TERM-012 — Standoff recovery before rejection.** If a movement result leaves the
  target wall or feature filling the view, too close or too oblique to verify, first regain
  a readable standoff with a short `hab_backward` step or one bounded point move, then
  inspect again. Never judge, reject, or close on a feature from an unreadable close-up;
  an unverified near miss is not a mismatch.

## Pre-close Audit

- **GN-TERM-020 — Trace audit.** Before blocked close, first write a GN-OBS-003 census from
  the latest tool result, then state:

  `blocked audit: latest move censused; current openings none; frontier audited; remembered alternatives tested`

  Every clause MUST be true. The audit cannot be the first message after movement. A recent
  `branch_entered` makes the final clause false until GN-MEM-022 is satisfied.
- **GN-TERM-021 — Evidence, not likelihood.** Semantic likelihood cannot mark an opening
  tested. Relevant bounded alternatives must be exhausted, unsafe, visibly impassable,
  freshly verified duplicates, or attempted and unreachable. One arbitrary retreat is not
  an exhaustion test.

## Two-step Blocked Close

- **GN-TERM-030 — Challenge means open.** The first `hab_close_session(outcome="blocked")`
  may return `status=blocked_close_audit_required`, `closed=false`, a token, frontier rows,
  and a post-entry flag. This is not completion and must not be followed by a final answer.
- **GN-TERM-031 — Choose one response.** Either continue exploring, which invalidates the
  token, or immediately retry close. The immediate retry MUST pass the token and satisfy
  GN-MEM-030 and GN-MEM-031. Never reuse a token after another tool call.
- **GN-TERM-032 — Successful terminal result.** Stop only after the close result has
  `closed=true` and the intended terminal claim. `blocked_frontier_audited` confirms that
  the close gate accepted the audit; `blocked_not_exhausted` means the terminal claim and
  ledger still conflict.
