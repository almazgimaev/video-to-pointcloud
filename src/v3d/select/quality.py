"""Frame selection "quality_nonredundant": sharp frames without near-duplicate viewpoints (T046).

Plan §7. The candidate list (already uniform in time) is split into ``budget`` consecutive
windows, and one frame is taken per window:

1. candidates of the window are ordered by sharpness, sharpest first;
2. the first one whose difference from the previously selected frame is at least ``tau`` is
   taken;
3. if every candidate is too similar, the sharpest one is taken anyway and reported as a
   redundant fallback — a window is never left empty just to avoid similarity.

Windows are the protection against the obvious failure of "just take the sharpest frames":
a global sharpness ranking concentrates the selection where the camera moved slowly and loses
viewpoints, which is exactly what camera recovery needs. The budget is never exceeded.

``diff`` is the frame-difference proxy from :mod:`v3d.metrics`: it compares brightness, not
geometry, so "similar" means "looks similar", not "was taken from the same place".
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from v3d.artifacts import FrameRecord
from v3d.select.uniform import CandidateLike, make_frame_id

SELECTOR_NAME = "quality_nonredundant"
# Relative, not absolute: the frame may be perfectly sharp, another one in its window was
# sharper. Kept distinct from "blurry" so that warnings about blurry video are not triggered.
REJECT_LESS_SHARP = "less_sharp_in_window"
REJECT_REDUNDANT = "redundant"


def window_bounds(count: int, budget: int) -> list[tuple[int, int]]:
    """``budget`` consecutive, non-empty windows ``[start, end)`` over ``count`` candidates."""
    if budget <= 0:
        raise ValueError("budget must be positive")
    windows = min(budget, count)
    if windows == 0:
        return []
    edges = np.round(np.linspace(0, count, windows + 1)).astype(int)
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:], strict=True) if b > a]


def select_quality_nonredundant(
    candidates: list[CandidateLike],
    *,
    budget: int,
    tau: float,
    sharpness_by_index: dict[int, float],
    diff: Callable[[int, int], float],
) -> tuple[list[FrameRecord], dict]:
    """Select at most ``budget`` frames; return records for all candidates and a summary.

    Args:
        candidates: in increasing ``src_index`` order.
        budget: maximum number of selected frames.
        tau: redundancy threshold for ``diff`` (tuned on a separate video, never on the
            evaluation set).
        sharpness_by_index: sharpness for every candidate (``src_index`` → value).
        diff: ``diff(prev_src_index, cand_src_index)`` — frame difference in ``[0, 1]``.

    Returns:
        ``(records, info)``: records in the original order (unselected ones carry the reason:
        ``less_sharp_in_window`` — a sharper acceptable frame won the window; ``redundant`` —
        sharper but too similar to the previous selected frame), and ``info`` with the parameters, window count,
        the redundant fallbacks and the number of diff evaluations (the cost of selection).
    """
    if tau < 0:
        raise ValueError("tau must be non-negative")
    missing = [c.src_index for c in candidates if c.src_index not in sharpness_by_index]
    if missing:
        raise ValueError(f"sharpness is missing for {len(missing)} candidates")

    chosen: dict[int, float | None] = {}  # src_index → diff to the previous selected frame
    reasons: dict[int, str] = {}
    fallbacks: list[int] = []
    evaluations = 0
    previous: int | None = None

    for start, end in window_bounds(len(candidates), budget):
        window = candidates[start:end]
        ranked = sorted(window, key=lambda c: (-sharpness_by_index[c.src_index], c.src_index))
        pick, pick_diff = None, None
        too_similar: list[int] = []
        for cand in ranked:
            if previous is None:
                pick = cand.src_index
                break
            value = float(diff(previous, cand.src_index))
            evaluations += 1
            if value >= tau:
                pick, pick_diff = cand.src_index, value
                break
            too_similar.append(cand.src_index)
        if pick is None:  # every candidate is too similar: keep the sharpest, report it
            pick = ranked[0].src_index
            pick_diff = float(diff(previous, pick)) if previous is not None else None
            fallbacks.append(pick)
            too_similar = [s for s in too_similar if s != pick]
        chosen[pick] = pick_diff
        for src in too_similar:
            reasons[src] = REJECT_REDUNDANT
        for cand in window:
            if cand.src_index != pick and cand.src_index not in reasons:
                reasons[cand.src_index] = REJECT_LESS_SHARP
        previous = pick

    records: list[FrameRecord] = []
    seq = 0
    fallback_ids: list[str] = []
    for cand in candidates:
        src = int(cand.src_index)
        selected = src in chosen
        frame_id = make_frame_id(seq if selected else None, src)
        if selected:
            seq += 1
            if src in fallbacks:
                fallback_ids.append(frame_id)
        records.append(
            FrameRecord(
                frame_id=frame_id,
                src_index=src,
                timestamp_us=int(cand.timestamp_us),
                sharpness=float(sharpness_by_index[src]),
                diff_to_prev_selected=chosen.get(src) if selected else None,
                selected=selected,
                reject_reason=None if selected else reasons.get(src, REJECT_LESS_SHARP),
                image_file=f"{frame_id}.jpg" if selected else None,
            )
        )

    info = {
        "selector": SELECTOR_NAME,
        "tau": tau,
        "windows": len(window_bounds(len(candidates), budget)),
        "selected": len(chosen),
        "redundant_fallbacks": fallback_ids,
        "diff_evaluations": evaluations,
    }
    return records, info
