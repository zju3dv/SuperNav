---
name: multi-global-navigation-learned-executor
description: Multi-goal mapless long-horizon Habitat-GS navigation driven by the learned point-conditioned local walker (hab_local_navigate). Use when an episode lists several ordered or unordered find goals and each reached goal must be marked in the hab_nav_goals ledger before the final close. Extends global-navigation-learned-executor with the goal lifecycle; all of its navigation rules still apply.
---

# Multi-Goal Global Navigation (Learned Executor)

Use MUST, MUST NOT, SHOULD, and MAY literally. Rule IDs are stable audit labels.

## Base Skill

- **MG-BASE-001 — Inherited navigation policy.** MUST read the complete run-local
  `global-navigation-learned-executor/SKILL.md` before scene action and follow all NM-* rules
  (lifecycle, observation gate, movement decisions, recovery, stop evidence) for the
  rest of the episode. Read its `references/` files when it tells you to. This skill
  only adds the multi-goal lifecycle; it never relaxes an NM-* rule.
- **MG-BASE-002 — Walker rules.** NM-LIFE-002 still applies: MUST read the complete
  run-local `localnav-pointnav/SKILL.md` before the first `hab_local_navigate`.

Read these from the same run-local skill bundle; never substitute a user-level copy.

## Goal Ledger

- **MG-LIFE-001 — Reconcile after init.** Immediately after the one successful
  `hab_init_scene`, MUST call `hab_nav_goals(action="status")` and confirm the listed
  goal indexes and descriptions match the mission list. The ledger is the only
  authoritative progress record. MUST NOT track goal progress in files, shell, or any
  side channel (NM-LIFE-003 applies to progress notes too).
- **MG-LIFE-002 — Re-reconcile on doubt.** After context loss, a long recovery, or any
  uncertainty about which goals are found, MUST call `hab_nav_goals(action="status")`
  again before deciding the next move.

## Active Goal Discipline

- **MG-GOAL-001 — One active hypothesis.** Work on exactly one pending goal at a time.
  NM-OBS-001's complete target hypothesis applies to the active goal only. When the
  mission states an order (First/Then/Finally), MUST pursue goals in that order.
- **MG-GOAL-002 — Switch only on a successful mark.** MUST NOT change the active goal
  until the current one is marked found. A failed mark leaves the active goal
  unchanged. Incidentally seeing a later goal MAY be noted, but MUST NOT trigger a
  switch while the active goal is pending.

## Mark Gate

- **MG-MARK-001 — Evidence bar.** MUST mark only when the active goal meets the
  NM-TERM-001 standard: the latest surround shows the complete target, its
  distinguishing attributes, required relations, and matching context at
  human-verifiable distance, with the agent already at the floor in front of it.
  Seeing the target across a room is not markable; hop close first.
- **MG-MARK-002 — Reconcile before marking.** MUST call
  `hab_nav_goals(action="status")` in the same decision as the intended mark, confirm
  the target index is still pending and its description matches the current visual
  evidence, then call `hab_nav_goals(action="mark", target_index=k)` with that exact
  index. Marks cannot be revoked; a wrong mark is permanent.
- **MG-MARK-003 — Mark result handling.** A mark without a successful result is not a
  mark. MUST resolve the returned error (unknown index, already found, or missing
  ledger) before continuing; MUST NOT blind-retry the same failing mark.

## Stop Gates

- **MG-TERM-001 — Achieved.** MUST close achieved only when a fresh
  `hab_nav_goals(action="status")` shows every goal found and the latest surround still
  shows the final goal at NM-TERM-001 standard. Then close exactly once with
  `hab_close_session(outcome="achieved")`.
- **MG-TERM-002 — Blocked.** If any goal stays pending after the bounded recovery gate
  (read `global-navigation-learned-executor/references/recovery-and-stop.md` first), MUST close
  blocked instead of marking speculatively or closing achieved with goals pending. A
  blocked close still requires the recovery reference's audit with every clause true.
