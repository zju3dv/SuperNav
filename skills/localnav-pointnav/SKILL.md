---
name: localnav-pointnav
description: Drive the learned RGB-only local navigator — mark a point you can see and it walks there. Covers how to pick the point, how to read what comes back, and what to do on every failure code.
---

# localnav-pointnav

One tool does the walking: **`hab_local_navigate(view="front", point=[x, y])`**.

You mark a point on the picture you are looking at; a learned policy drives the body
there using the camera only — no map, no coordinates, no depth. One call per hop, no
preview/confirm round-trip.

## How to pick the point

- Pass `view` ("front", "right", "back", or "left") plus the point. Prefer `view` over
  `image_ref`, and choose the view by which of the four latest surround images visibly
  contains your target — if it shows in the front image's right half, that is still
  `view="front"`. Never invent coordinates for a different image than the one you mean.
- Mark **anything you can currently see** in the view. Near or far is fine — if it is
  visible, it is a legal target.
- `[x, y]` are normalized to the image: `x` 0 = left edge, 1 = right edge; `y` 0 = top,
  1 = bottom. `[0.5, 0.5]` is dead centre.
- Aim at whatever you want to reach. **The floor in front of the target is the
  preferred point** — it is the most reliable choice and gives the policy a
  walkable destination. Marking the target itself (its body, its base, even its
  middle) also works — the policy homes on the exact pixel you marked — so use
  the object body when no floor point is available, for example on the final
  close approach where the floor at the target's base is out of frame (see the
  camera-height note below). Only the sky/ceiling above the horizon line is
  pointless: there is nothing to walk toward there.
- Doorways are the exception to "mark what you want to reach": never mark the
  doorframe or jamb itself — the policy walks toward the exact pixel and will
  scrape or wedge on the frame. When you mean to go through a doorway, or to
  peek into the next room while standing side-on to the door (frame filling the
  view edge), mark the open floor just past the threshold, centred in the
  doorway opening. The hop ends inside the doorway with a fresh surround, which
  is exactly the observation you wanted.
- The camera sits at 1.25 m. Floor closer than about 1.7 m is below the bottom
  edge of the frame — you cannot mark the floor at your own feet or at the base
  of a target you are already standing next to; for that final approach, mark
  the target's body instead. Judge distances from the image itself: floor far
  away bunches just below the horizon, so near the middle line a small vertical
  difference is several meters on the floor — something far away sits close to
  the horizon line, not at the bottom of the frame.
- Horizontal position does not matter: a target at the left or right edge works as well
  as one in the centre (measured, distance held constant).
- The point belongs to the **latest** capture. After any movement, turn, or new
  observation, old points and image references are stale — mark a fresh one.

## What comes back

`status` plus a FRESH four-view surround. Every next hop MUST be marked on that
latest surround — never reuse a previous call's point or goal, and never aim from
memory.

| status | meaning | what to do |
|---|---|---|
| `reached` | the policy arrived at the pixel you marked — **not** necessarily at your target; a fresh surround view is attached | look at the attached view: if the target is still metres away, that hop simply moved you closer, so mark the next point |
| `needs_agent` (`low_confidence`) | the policy is unsure from this view | inspect the surround you already hold, then mark a fresh point on the best of the four views |
| `needs_agent` (`no_progress`) | it is not making headway | change the situation first — mark a nearby offset point or a different route on the latest surround |
| `blocked` | it stopped making forward progress (wedged, or circling in a small area) | back off by marking a floor point on the back view, then mark a new point along a clearly more open route |
| `timeout` | step budget spent; **partial progress is kept** | mark a fresh point toward the same target from where you now stand (closer, so the success rate is higher) |
| `stale_image_ref` / `invalid_image_ref` | the point referred to an old capture | mark a fresh point on the latest surround |
| `policy_service_unreachable` | the learned policy is down | report the degradation; no translation fallback exists in this setup (`hab_turn` still works for observation only) |

## While a hop is running

- `hab_localnav_status` → live progress: steps, replans, collisions, stop probability.
- `hab_localnav_stop` → cancel at the next replan boundary.

Reading the telemetry: collisions climbing while steps advance slowly means the body is
scraping something — stop, back off, and re-route rather than waiting it out. A stop
probability that stays near zero after many steps means the policy still thinks the goal
is far; that is normal on a long hop, not a reason to cancel.

## Working with the rest of the loop

1. You always hold the full surround: `hab_init_scene`, every
   `hab_local_navigate` hop, and every `hab_turn` all return fresh
   front/right/back/left views. Inspect the surround you already have before
   spending any action — a `hab_turn` called just to look
   around is almost always redundant.
2. Choose the direction that plausibly holds what you were asked for and mark
   the point directly on that view — front, right, back, and left are all
   legal. The walker turns itself to face the marked view before moving off,
   so do **not** spend a `hab_turn` just to face the target first; turning
   first is pure overhead.
3. Mark a point and call `hab_local_navigate`.
4. On `reached`, inspect the attached surround view; if the target is not actually there,
   keep going — mark the next point.
5. Only close the session (`hab_close_session`) at arm's length from the target:
   near enough to touch it (within ~1 m of its surface). Before any achieved
   close you MUST take one final approach hop marked on the target's body or
   base (never the floor well in front of it), even when you believe you are
   already close enough — the policy's stop head can end a hop up to ~0.5 m
   short of your marked point, so a hop that merely entered the target's room
   does not count. On the surround that hop returns, verify: the target's base
   is cut by the frame's bottom edge — you can no longer see the floor patch it
   stands on — and its body dominates the view. Seeing it across the room is
   not arriving, and seeing it clearly with its whole base and floor visible in
   front of it is not arriving either: repeat the final approach hop. If that
   hop returns `blocked` or `no_progress` with the target directly in front of
   you, you are already at touch distance — that result completes the final
   approach; verify the frame test and close. Measured on this benchmark,
   sessions closed while the target was still 2 m away scored as failures more
   often than not, and stops 0.3–1 m short of the target failed the strict
   arrival check too.
   Close with a not-found report instead if two further hops bring you no closer, or if
   you are confident the target is not in this space.


## Reference

- `references/failure-playbook.md` — worked examples of each failure code, and how the
  telemetry looks in each case.
