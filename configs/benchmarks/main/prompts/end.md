Take as many moves as you genuinely need. After EACH movement attempt, inspect the returned surround view and explicitly ask yourself: "Does what I now see satisfy the request?" Base this decision on the images first, not on whether a detector succeeded.
Before closing, make sure the target is reasonably close and clear for the final viewpoint: if the target is visible but still far away, move closer when there is a safe reachable approach;
  - If YES -> call hab_close_session immediately and stop.
  - If NO -> keep searching, reorienting, or moving toward the next promising visible target.

When you've settled, call hab_close_session and give a short report: what you saw, what you decided and why, and how it went.
