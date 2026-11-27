---
name: locate-anything
description: Guidance for choosing LocateAnything phrases for Habitat-GS visual grounding. Use when Codex needs to call or prompt around `hab_visual_ground_preview`, LocateAnything, visual ground preview, current-image phrase grounding, proxy visual anchors, candidate disambiguation, or view/phrase selection from latest `panorama_images`.
---

# LocateAnything Phrase Grounding

## Overview

Use LocateAnything as a current-image grounding aid: it answers "what should be found in this image?", not "where should the robot go?". Choose short, visible referring expressions from the selected panorama image.

## Phrase Rules

- Use noun phrases, not task sentences.
  - Good: `bed`
  - Better: `the white bed in the bedroom`
  - Avoid: `go to the bed`, `walk into the bedroom and find the bed`

- Add one visible qualifier when the target is clear:
  - `the white bed`
  - `the bed against the wall`
  - `the bed on the right side of the image`
  - `the couch partially visible through the doorway`
  - `the wooden cabinet beside the sink`

- Disambiguate repeated categories. Never use a bare category when several instances are visible.
  - Avoid: `chair`
  - Use: `the chair near the table`
  - Use: `the left chair`
  - Use: `the dark chair in the corner`

- Prefer visible image evidence over inferred task intent. If the final target is not visible, ground a visible proxy that moves toward the likely target region.
  - Bed target, only doorway visible: `the doorway to the bedroom`, `the open bedroom door`
  - Sink target, only kitchen area visible: `the kitchen counter`, `the cabinet below the sink area`
  - Sofa target, seating area not fully visible: `the rug leading toward the seating area`, `the coffee table in the living room opening`

## View Selection

- Choose `view` by the panorama image that actually contains the target or proxy: `front`, `right`, `back`, or `left`.
- Use image-coordinate language inside the phrase only when it is visually clear.
- Do not switch `view` just because the phrase says "right" or "left".
  - If a bed is in the right half of the front image, call `hab_visual_ground_preview(view="front", phrase="the bed on the right side of the image")`.
  - Use `view="right"` only when the target is in the separate right panorama image.

## Tool Use Pattern

- Inspect the latest `panorama_images` first.
- Decide whether the final target is visible. If not, choose a visible proxy anchor.
- Call LocateAnything with a concise phrase for the selected image.
- If multiple candidates appear, inspect the numbered overlay and choose the best candidate with `confirm_token` and `candidate_id`. A high-confidence door/opening candidate also requires this confirmation even when it is the only candidate.
- Reject an unsuitable preview by not confirming it. There is no cancel/abandon mode or
  cancel candidate ID. MUST NOT pass `mode="abandon"`, invent an out-of-range
  `candidate_id`, or send the old token merely to clear state.
- To refine or replace a rejected preview, call `hab_visual_ground_preview` again without
  `confirm_token` or `candidate_id`, using a more specific or different phrase on the same
  latest view. The new preview supersedes the pending one. To take a justified movement or
  turn instead, call that action normally without the preview token; the fresh panorama
  invalidates the old token. Never reuse a rejected token.
- Candidate metadata may include `anchor_strategy`; `portal_normal` means the stop point was generated from a high-confidence doorway normal. Prefer a semantically correct reachable marker when the overlay shows a bad anchor.
- High-confidence door/opening previews sample stand-off points along the doorway's perpendicular bisector instead of the line from the agent to the box center. Confirmed movement stops on the agent-facing side and automatically turns to face the opening after arrival. Pass through only with a later move after inspecting the doorway again.
- Match door candidates to the immediate intent; do not blindly choose either an in-box marker or the nearest floor marker. Candidate lists are ordered by increasing `rank_score`, so lower is better; treat `ranking_penalties` as reasons to avoid a candidate, not as confidence bonuses.
- Prefer `portal_normal` when present: use it to stage safely at the threshold, inspect again, then cross with a later move. If portal geometry is unavailable and the immediate intent is to enter or cross, prefer a reachable `bbox_center` or semantically correct `lower_band` marker visibly inside the opening when it has a low rank, no depth-contamination penalty, a small snap distance, and a reasonable path.
- Use `below_bbox_floor` when the immediate intent is only to approach or inspect the near side of the doorway, or when every in-opening marker is visibly unsuitable. Do not choose it merely because it is on clear floor or has the shortest path.
- Reject any marker on a jamb, wall, closed door leaf, foreground obstacle, or unrelated far surface even when it lies inside the detection box.
- Use LocateAnything as the primary navigation choice whenever a visible object or semantic proxy can express the destination, including proxies for long or room-to-room approaches.
- If candidates are wrong or unreachable, a visual point on a reachable region is one valid bounded recovery. Otherwise reserve visual-point navigation for micro-adjustment, a cautious short advance over clearly open floor, exact stand-off/framing, or crossing a doorway after grounding reaches its threshold. Do not chain point hops when a useful visible object or proxy can be named.
- Treat every movement, turn, or fresh panorama as invalidating old `view`, `image_ref`, `confirm_token`, candidate markers, and points.

## Avoid

- Do not use full navigation instructions as phrases.
- Do not ask LocateAnything for objects that are not visible in the selected image.
- Do not encode world coordinates, scene graph ids, `obj_<number>` aliases, hidden labels, or remembered off-screen targets in the phrase.
- Do not treat a failed detection as proof that the target is absent from the scene.
