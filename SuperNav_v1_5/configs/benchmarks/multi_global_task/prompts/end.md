
Before closing, call hab_nav_goals(action="status") and verify every goal shows found.
Then inspect the latest surround and verify the final goal against visible evidence: the
requested target MUST be reasonably close and clear, its distinguishing attributes and
spatial relations MUST match its description, and its surrounding context must agree. If
any goal is still pending, do not close with outcome="achieved" — either keep searching
or close blocked.

Complete exactly one successful hab_close_session, then give a short report of the visual
evidence or the reason the episode was blocked. If a blocked close returns
`blocked_close_audit_required`, the session remains open: complete the returned audit and
retry immediately with its token. Do not answer until a close result has `closed=true`.
