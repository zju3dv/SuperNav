"""Variable-length history schedule (GCA-style): anchor + adaptive-stride
memory + current frame.

Single source of truth for BOTH the training dataset and the serving loop —
train/serve consistency is a hard invariant here (the schedule defines the
input distribution the model learns).

Reference behaviour (stride 2, like the design sketch):
    schedule(0, 8, 12)  -> [0, 2, 4, 6, 8]        # short: no compression
When the stride-2 chain exceeds the budget, the OLDER half is repeatedly
thinned (far-sparse / near-dense), always keeping the anchor and the
current index:
    schedule(0, 60, 12) keeps index 0 and 60, dense near 60, sparse near 0.
"""

from __future__ import annotations


def memory_schedule(
    start: int,
    current: int,
    budget: int,
    recent_tail: int = 0,
    include_anchor: bool = True,
    mid_span: int = 0,
    mid_count: int = 0,
) -> "list[int]":
    """Ascending frame indices, len <= budget, always containing
    ``start`` (anchor) and ``current``. budget must be >= 2.

    recent_tail=0 reproduces the v1 stride-2 schedule bit-for-bit (models
    trained on v1 must keep serving v1). recent_tail=N additionally forces
    the N consecutive predecessors of ``current`` into the sequence — the
    v2 fix: v1's stride-2 tail halved recent-motion temporal resolution vs
    the fixed-4 baseline (collisions 2.6x, SPL 0.24); a dense tail makes
    variable memory an information superset of the baseline."""
    start, current = int(start), int(current)
    budget = max(2, int(budget))
    if current <= start:
        return [start]
    if not include_anchor and budget <= int(recent_tail) + 1:
        lo = max(start, current - max(0, int(recent_tail)))
        return list(range(lo, current + 1))[-budget:]
    if include_anchor and budget == 2:
        return [start, current]
    if recent_tail > 0:
        recent = [
            i for i in range(current - int(recent_tail), current + 1)
            if i > start
        ]
        if not recent or recent[-1] != current:
            recent.append(current)
        # Reserve the anchor slot so enabling it leaves middle-frame capacity unchanged.
        head_slots = budget - 1
        recent = recent[-head_slots:]
        head_budget = head_slots - len(recent)
        # Anchor inclusion leaves middle-frame indices unchanged.
        # A positive mid_span limits selection to the most recent steps.
        lo = (
            start + 2 if int(mid_span) <= 0
            else max(start + 1, current - int(mid_span))
        )
        if int(mid_count) > 0:
            # Select a fixed frame count across the window, including both endpoints.
            window = list(range(lo, recent[0]))
            k = min(int(mid_count), head_budget, len(window))
            if k <= 0:
                mid = []
            elif k == 1:
                mid = [window[-1]]
            else:
                step = (len(window) - 1) / (k - 1)
                mid = sorted({window[round(j * step)] for j in range(k)})
        else:
            mid = list(range(lo, recent[0], 2))
            while len(mid) > head_budget:
                thinned = mid[::2]
                mid = thinned if len(thinned) < len(mid) else mid[1:]
        return ([start] if include_anchor else []) + mid + recent
    indices = list(range(start, current, 2))
    if not indices or indices[-1] != current:
        indices.append(current)
    while len(indices) > budget:
        before = len(indices)
        half = max(1, len(indices) // 2)
        head, tail = indices[:half], indices[half:]
        head = head[::2]
        if head[0] != start:
            head = [start] + head
        merged = head + tail
        if len(merged) >= before:
            # head cannot thin further — thin the tail interior, pinning
            # the current index
            merged = [merged[0]] + merged[1:-1:2] + [merged[-1]]
        if len(merged) >= before:
            # provable-termination fallback: anchor + newest (budget-1)
            merged = [start] + [i for i in indices[-(budget - 1):] if i > start]
            indices = merged
            break
        indices = merged
    if indices[0] != start:
        indices[0] = start
    if indices[-1] != current:
        indices[-1] = current
    return indices
