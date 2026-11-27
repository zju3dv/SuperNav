# Spatial Memory Rules

Read this file completely before the first junction registration or any `branch_id` reuse.

## Registration

- **GN-MEM-010 — Literal sibling set.** At a fresh surround with two to four distinct
  new/uncertain sibling transitions, call `hab_register_spatial_junction` with every
  candidate's latest `view`, literal `phrase`, and normalized `[x, y]` point. Do not
  register only the intended choice or invent room identities as memory facts.
- **GN-MEM-011 — Material novelty.** Register only a materially new sibling set. Do not
  create another junction after a tiny pose change that shows the same openings. Do not
  register the traversed route or an inspected route merely to manufacture a pair. This
  preserves the three-junction ledger for useful unresolved choices.

## Branch Lifecycle

- **GN-MEM-012 — Exact reuse.** A branch ID is not a LocateAnything `candidate_id`, an
  off-screen waypoint, or a label for similar corridors. Reuse it only when the original
  opening and sibling context are directly visible. Generic return proxies omit
  `branch_id`. Pass the selected ID on both grounding preview and confirmation.
- **GN-MEM-013 — Movement proxy only.** A preview, intention, abandoned/expired token,
  failed move, or short approach does not enter a branch. `entered` records a formally
  reached branch-bound grounding proxy; it does not prove geometric threshold crossing,
  room function, meaningful inspection, or exhaustion. Current pixels decide whether the
  camera is still at the doorway. If the ledger drops such a proxy from frontier before a
  real crossing or meaningful inspection, retain its ID and phrase in compact working state.
- **GN-MEM-014 — Token/history separation.** A fresh image expires action tokens and
  points but does not erase branch history. `selection_failed`, `confirm_abandoned`, and
  `confirm_invalidated` preserve the branch as unentered.

## Reading Results

- **GN-MEM-020 — Evidence order.** For each result, read current pixels first, then
  `spatial_memory.new_events`, then `frontier.unentered_branches`. Consult
  `recent_junctions` only to reacquire one specific older branch. Treat `observed` and
  `approached` as unentered. Ignore revision, source-capture, evidence-kind, and token
  references when deciding where to navigate.
- **GN-MEM-021 — Bounded ledger.** Junction eviction or disappearance is not evidence
  that a branch was entered, duplicated, or exhausted. Keep one or two important evicted
  IDs and literal phrases in compact working state.
- **GN-MEM-022 — Post-entry follow-up.** When `new_events` contains `branch_entered`, inspect
  the returned surround and take at least one subsequent observation-bearing action before
  blocked close. If still at the threshold, cross or reveal the opening. If pixels already
  expose a clearly mismatched room, the action MAY exit, return, or inspect a side exit; do
  not approach an irrelevant interior object merely to clear follow-up. Newly visible doors
  are new facts and outrank a farther old continuation. The only exception is fresh visual
  evidence that no safe passable interior exists; if the target is visible, verify it.

## Audited Blocked Close

- **GN-MEM-030 — Exact frontier audit.** A `blocked_close_audit_required` result lists the
  authoritative active frontier. The immediate retry's `frontier_audit` MUST contain each
  listed `branch_id` exactly once, no unknown IDs, and non-empty evidence. Each disposition
  MUST be one of:
  - `freshly_verified_duplicate`
  - `unsafe`
  - `visibly_impassable`
  - `attempted_unreachable`
  A guessed or visually mismatched room is not an accepted disposition by itself.
- **GN-MEM-031 — Post-entry exception.** If the close gate reports
  `post_entry_followup_required=true`, the immediate retry also requires
  `post_entry_exception={"disposition":"no_safe_passable_interior","evidence":"..."}`.
  Use it only when the latest pixels actually prove that exception; otherwise continue.
- **GN-MEM-032 — Token scope.** The close-audit token is bound to the session, active
  frontier, and pending post-entry state. Any intervening non-close tool invalidates it.
  After further exploration, start a new blocked-close audit from current state.
