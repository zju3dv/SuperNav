
Before closing, inspect the latest surround and verify the complete instruction against
visible evidence. The requested target MUST be reasonably close and clear, its
distinguishing attributes and spatial relations MUST match the complete instruction, and
its surrounding context must agree. If the target is visible but too distant, approach it
when a safe reachable route is visible.

Complete exactly one successful hab_close_session, then give a short report of the visual
evidence or the reason the episode was blocked. If a blocked close returns
`blocked_close_audit_required`, the session remains open: complete the returned audit and
retry immediately with its token. Do not answer until a close result has `closed=true`.
