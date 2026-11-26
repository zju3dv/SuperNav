---
name: go2-pixelgoal
description: Navigate a Go2 with four-way PixelGoal observations and bounded visual hops.
---

# Go2 PixelGoal Operating Notes

- The harness attaches the initial front/right/back/left views in that order.
  Inspect them, call exactly one `pixelgoal_navigate(point=[x, y],
  direction=...)` hop per observation, then repeat from the returned views until
  the goal is visually satisfied.
- Every navigation point must sit on a clearly visible, unambiguous surface.
  Preferred: clearly traversable floor. Also allowed as a long progress hop: a
  large stable vertical surface in the travel direction (wall face, cabinet
  face, solid door panel, large furniture front) — the robot stops about
  0.35 m in front of the marked surface, so a distant stable surface is a safe
  way to cross open space efficiently.
- When the travel direction is clear, choose the farthest clearly visible safe
  option: the farthest traversable floor, or a stable vertical surface farther
  along the route when floor runs out of view. Use a shorter local hop only
  when farther choices are occluded, ambiguous, or unsafe.
- Never click glass, windows, mirrors/reflections, transparent doors or
  anything seen through them; thin structures (chair/table legs, cables, bags,
  stands); the floor-wall boundary line itself; sky, ceiling, the horizon, or
  empty space. Keep a safety margin from obstacles and floor boundaries.
- For a find/search task, never click the target object itself; stop only when
  it is close enough for human-verifiable identity, not merely visible far away.
- Keep the robot still between observation and navigation so the normalized
  point still matches the latest camera frame used by PixelGoal.
- `pixelgoal_navigate` waits for the hop to finish and returns its terminal
  status, camera freshness metadata, and fresh front/right/back/left views. Do
  not navigate from a missing or stale view. Call `observe_surround` or
  `pixelgoal_status` only to recover from missing/stale images or diagnose an
  abnormal result.
- The map backend can perform successive exploration legs of up to 1.5 m when
  the target is unresolved, rechecking after each successful leg. The default
  limits are 60 seconds for the whole hop and a 10 m exploration budget.
  Exploration stays close to the clicked direction. At an incomplete map edge,
  it may extend at most 0.5 m into unknown space from reachable free space;
  another unknown leg requires new free-space evidence near that frontier.
  This does not permit clicking visually unsafe or ambiguous surfaces.
- `needs_agent / exploration_complete` is an intentional observation handoff
  when the exploration budget or reachable free space runs out, not a failed
  tool call, a request to keep polling, or proof that the point/task was reached.
  Inspect all fresh returned views and choose a new point if movement is needed.
- `timeout / deadline_exceeded` ends the entire hop; inspect its fresh views and
  choose the next safe point if more exploration is needed. Do not resume the
  old hop or automatically retry its point without a new visual assessment.
- `needs_agent / blocked_pixel_target` means a known obstacle blocks the approach
  to the clicked surface or its stopping point. Inspect the new views and choose
  a safe alternative; do not repeat the same point or assume the target was reached.
- `needs_agent / no_progress` or `exploration_failed` means the hop ended without
  sufficient approach or without reaching its exploration waypoint. Do not repeat
  the same image point or try to push through the same doorway. First inspect all
  returned views; move again only if they establish a clearly safe alternative.
  If glass, a closed/uncertain door, or a person prevents a safe route, remain
  stopped and report the physical blockage. An explicit operator/safety stop
  overrides exploration and blocked-audit requirements.
- For a find/search task, finish only when the target is clearly visible in a
  fresh returned view, close enough for human-verifiable identity, and the
  requested location condition is satisfied. If it is only small or distant,
  approach via safe visible floor proxies while keeping a safety margin. Stop
  and cite the direction and visual evidence once close enough.
- In the normal path, do not call `server_status`, `robot_protection_status`, or
  `motion_safety_status`; the authority proxy checks authority and battery on
  every movement call. Use diagnostics only after a refusal or abnormal result.
- Do not start a second hop while one is running. Use `pixelgoal_stop` on
  operator stop or abnormal motion. `reset_pixel_goal_map` is operator-only.

## Tool Invocation Discipline

- Call the navigation tool directly as `pixelgoal_navigate` on the `dimos`
  MCP server. There is no namespace lookup step and no warmup or test call;
  never probe the tool surface with placeholder commands.
- The shell exists only to read skill files. Never use it for no-op or
  placeholder commands (`echo`, `true`, `printf`, redirections to /dev/null),
  and never for navigation, perception, or note-taking. Every action is
  either a real `dimos` MCP tool call or a message to the operator.
