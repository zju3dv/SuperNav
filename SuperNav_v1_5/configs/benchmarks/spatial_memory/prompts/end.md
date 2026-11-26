Take as many moves as you genuinely need. After EACH movement attempt, inspect the returned surround view and explicitly ask yourself: "Does what I now see satisfy the request?" Base this decision on the images first, not on whether a detector succeeded. For potentially duplicated or ambiguous objects—paintings, wall stickers, beds, bookshelves, desks, etc.—make sure what you see matches all task qualifiers and requirements. Memory nodes, captions, map labels, route status, and arrival at a proxy are navigation aids; none proves task completion by itself.

  - If YES -> do the following before closing actions.
  - If NO -> keep searching, reorienting, or moving toward the next promising visible target.

Before closing, or moving on to the next target during multi-target tasks, make sure you have done the following:
  1. MUST spend one `hab_visual_point_navigate` hop marked on the target itself—its body or its base, not the floor well in front of it—even when you believe you are already within ~1 m. The follower stops at a standoff point up to ~0.7 m from the marked surface, and a hop that merely reached the target's room is not this hop.
  2. Verify on the surround that hop returned: the target's base is cut by the frame's bottom edge (you can no longer see the floor patch it stands on, nor its whole underside) and its body dominates the view. For a target on a raised surface, the surface's near edge under the target is at or below the frame's bottom area and the target looms large; if you can still see the target's entire base with floor or surface-front visible in front of it, you are too far—MUST repeat step 1.

When you've settled, complete exactly one successful `hab_close_session` with `outcome="achieved"` when the requested target is visually verified or `outcome="blocked"` when giving up. Do not answer until the close result has `closed=true`. Then give a concise report of what you saw, what you decided and why, and how it went.
