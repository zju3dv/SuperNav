# Failure playbook (what to do for each return code)

Each entry is written as "what you see → what to do → what not to do". Telemetry comes
from `hab_localnav_status` or the `steps` / `replans` / `collisions` /
`done_probability` fields in the call result.

Keep one fact in mind throughout: every `hab_local_navigate` result already carries a
fresh front/right/back/left surround, and the walker turns itself to face the marked
view before moving off. You almost never need `hab_turn` just to
look or to face something — act on the surround you already hold.

## `reached`

**What you see**: a fresh surround view is attached.
**Do**: look at it first and confirm the thing you want is really nearby (clearly
visible, occupying a solid share of the frame). Only then
`hab_close_session(outcome="achieved")`.
**Do not**: announce completion the moment you see `reached` — the stop judgement
often fires early and the attached view still shows the target some distance away. If
you are not there, mark a fresh point from where you stand and keep walking.

## `needs_agent` + `low_confidence`

**What you see**: the policy has no confidence in the current view; it handed control
back after zero or a few steps.
**Do**: inspect the surround you already hold. If the goal sits better in another of
the four views, mark the point on that view directly — the walker re-faces it on its
own. Then retry by marking that fresh point on the latest surround.
**Do not**: call `hab_turn` just to look or to re-face the goal —
you already have all four views, and turning first is pure overhead. Do not switch
goals immediately, and do not retry with an old `image_ref`.

## `needs_agent` + `no_progress`

**What you see**: `steps` increasing while the position barely changes, or `replans`
increasing while `steps` does not move.
**Do**: change the situation before retrying. Mark a nearby offset point on any of the
four surround views — stepping a metre sideways gives the policy a genuinely fresh
view — or pick a different route to the same goal on the latest surround.
**Do not**: the policy only sees the current frame and the goal image; with an
unchanged view a retry produces essentially the same actions, so do not retry
repeatedly from the same spot with the exact same view.

## `blocked`

**What you see**: the robot is pacing inside a very small area or is stuck against
something; `collisions` is usually well above 0.
**Do**: back off by marking a floor point behind you on the back view (the walker
turns itself around), then from the returned surround mark a new point in a **clearly
more open** direction.
**Do not**: keep marking nearly the same point from the same spot.

## `timeout`

**What you see**: the step budget is spent, but **the distance already covered is
kept**.
**Do**: re-mark the same target from where you now stand on the latest surround
(closer, so the success rate is higher).
**Do not**: treat it as a failure and give up — most of the time it simply means the
route was long.

## `stale_image_ref`

**What you see**: an error saying the image reference has expired.
**Do**: mark the point on the freshest surround you hold — the last movement result
usually already carries one. Only if you hold no fresh views, call
`hab_turn(direction="right", degrees=10)` once to turn 10° and obtain a fresh
surround. Mark a new point on that returned surround before retrying.
**Do not**: reuse coordinates from the expired capture.

## `policy_service_unreachable`

**What you see**: the call fails outright, reporting the policy service unreachable.
**Do**: report the degradation explicitly in the final report ("learned policy
unavailable"). No translation fallback exists in this setup — `hab_turn` still works,
but only for observation.
**Do not**: continue silently and let anyone believe the run measured the normal
setup.

## Judging rising collisions

`collisions` climbing on every call while the target feels no closer → the body is
grinding against an obstacle. Interrupt immediately with `hab_localnav_stop`, back
off, and change direction — cheaper than letting it burn the whole budget.
