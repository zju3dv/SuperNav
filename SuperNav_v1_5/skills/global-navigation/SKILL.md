---
name: global-navigation
description: Mapless long-horizon Habitat-GS navigation for global tasks. Use when an episode requires searching across rooms, maintaining target evidence, choosing visible proxies, auditing local branches, recovering from stalls, and stopping only on final visual evidence.
---

# Global Navigation

Use MUST, MUST NOT, SHOULD, and MAY literally. Rule IDs are stable audit labels.

## Load the Needed Rules

- Before the first junction registration or reuse of a `branch_id`, read
  [references/spatial-memory.md](references/spatial-memory.md) completely.
- Before leaving the local area, taking a long return, or acting on a frontier warning,
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

- **GN-LIFE-001 — One initialization.** MUST read this file before scene action and call
  the prompt's exact `hab_init_scene` once. MUST NOT use init as observation refresh or
  blindly retry a failed/timed-out init.
- **GN-LIFE-002 — Grounding rules.** MUST read the complete run-local
  `locate-anything/SKILL.md` before the first `hab_visual_ground_preview`.
- **GN-LIFE-003 — Mapless evidence boundary.** MUST perceive and act only through the
  whitelisted MCP tools and their latest returned images. MUST NOT use shell, Python,
  files, artifact directories, maps, coordinates, goal metadata, trajectories, evaluator
  data, or hidden simulator state.
- **GN-LIFE-004 — Formal finish.** MUST obtain exactly one successful close result with
  `closed=true`, then answer. An audit challenge is not a close and MUST be resolved as
  specified under GN-TERM-002.

## Target and Observation Gate

- **GN-OBS-001 — Complete target hypothesis.** MUST keep a compact hypothesis containing
  the requested object/region, distinguishing appearance or material, required relations,
  room context, and final visual evidence. A category match or detector result alone is
  insufficient; a detector miss does not prove absence.
- **GN-OBS-002 — Observe–decide–act.** After init and every successful movement or turn,
  MUST inspect front, right, back, and left, including both horizontal edges; test the full
  target hypothesis; census every visible transition; choose one bounded action; and repeat
  on its returned surround.
- **GN-OBS-003 — Census before decision.** MUST classify each doorway, side opening,
  corridor continuation, and room interior as traversed, meaningfully inspected, or
  new/uncertain. MUST list new/uncertain choices before selecting a proxy. Use this compact
  trace form:

  `census: front=...; right=...; back=...; left=... | new=... | frontier=... | next=... | why=...`

Current pixels outrank bounded memory. An empty ledger frontier is not house exhaustion.
A movement that empties frontier may expose the next junction. MUST NOT close blocked or
start a return in the same decision as a successful movement; first report the fresh
surround with GN-OBS-003.

## Core Navigation Decisions

- **GN-MEM-001 — Register real choices.** When one latest surround contains two to four
  distinct new/uncertain sibling transitions, MUST register all of them before selecting
  one. With one new transition, inspect it directly. Read the spatial-memory reference
  before registration or branch reuse.
- **GN-MEM-002 — Branch meaning.** A `branch_id` records one observed opening and an audited
  movement status. It is not a room label, candidate ID, off-screen waypoint, geometric
  threshold proof, or proof of exploration. When a branch-bound grounding reports entered
  but current pixels still show the doorframe or threshold, treat the branch operationally
  as approached and retain its ID and phrase in compact working frontier until pixels prove
  a real crossing or meaningful inspection. MUST reuse an ID only after exact visual
  re-observation of that opening and sibling context, and MUST pass it on both preview and
  confirmation.
- **GN-MOVE-001 — Relevance before novelty.** SHOULD prefer current target evidence or a
  target-relevant current route, then a safe unfamiliar opening, a current unentered sibling,
  and finally a bounded return to one exact older frontier. A clearly mismatched room MAY be
  deferred while a stronger route exists, but MUST remain remembered as frontier. MUST NOT
  call an unfamiliar opening exhausted only because its guessed room type seems unlikely.
- **GN-MOVE-002 — Visible bounded movement.** MUST ground only a target or proxy visible in
  the selected latest view. SHOULD use LocateAnything for named objects, furniture,
  doorways, corridors, and semantic proxies; MAY use visual point navigation for bounded
  threshold crossing, open-floor adjustment, framing, or recovery. Reaching a proxy is not
  completion. Any movement, turn, or fresh panorama expires old image refs, points,
  markers, and action tokens.
- **GN-MOVE-003 — Long-move gate.** Before any planned move of at least 3 m, MUST compare
  it with the current census and local unentered siblings. It is allowed when it approaches
  the target itself or a target-relevant current route, or targets an exact unresolved
  frontier, provided no more relevant local choice or unregistered opening is skipped.
  Record lower-relevance siblings for later. Read the exploration reference before confirming.

## Stop Gates

- **GN-TERM-001 — Achieved.** MUST close achieved only when the latest surround shows the
  complete target, distinguishing attributes, required relations, and matching context at
  human-verifiable distance.
- **GN-TERM-002 — Blocked.** MUST read the recovery reference and exhaust its bounded gate
  before blocked close. If the first blocked close returns
  `blocked_close_audit_required`, the session is still open. MUST either continue exploring
  (which invalidates that token) or immediately retry with the returned token, an exact
  audit for every reported frontier branch, and any required post-entry exception. MUST NOT
  answer until the result has `closed=true`.
