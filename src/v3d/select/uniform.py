"""Uniform frame selection (baseline, FR-008…FR-011).

The `uniform` strategy: `budget` frames are taken from the candidate list, spread
evenly by position in the list. Candidates are extracted from the video with a constant
stride (`v3d.extract`), so uniformity by position in the list is uniformity
over the time of the source video.

A record (`FrameRecord`) is created **for every considered frame**, not only for the
selected one: `frames.json` must explain the fate of every candidate (data-model.md §3.2).

The candidate type is described structurally (`CandidateLike`) rather than by importing
`v3d.extract.CandidateFrame`: the extraction module is written in parallel, and a hard
import would couple the two parts needlessly — `select_uniform` reads only `src_index`
and `timestamp_us` from a candidate.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from v3d.artifacts import FrameRecord

# Rejection reasons this module can set (data-model.md §3.2).
REJECT_BUDGET_EXHAUSTED = "budget_exhausted"

# Sequence number mask for an unselected frame: it has no place in `package/images/`.
UNSELECTED_SEQ_MASK = "----"


@runtime_checkable
class CandidateLike(Protocol):
    """The minimum that `select_uniform` reads from a candidate."""

    src_index: int
    timestamp_us: int


def _as_float_or_none(value: float | None) -> float | None:
    """`None` stays `None`: a missing metric is replaced by neither zero nor `nan`."""
    return None if value is None else float(value)


def make_frame_id(seq: int | None, src_index: int) -> str:
    """`NNNN_SSSSSS` for a selected frame, `----_SSSSSS` for an unselected one."""
    head = UNSELECTED_SEQ_MASK if seq is None else f"{seq:04d}"
    return f"{head}_{src_index:06d}"


def uniform_positions(count: int, budget: int) -> list[int]:
    """Positions in the candidate list, spread evenly; strictly increasing.

    The extreme candidates are always included: the start and end of the shoot are viewpoints too.
    """
    if budget <= 0:
        raise ValueError(f"frame budget must be positive, got {budget}")
    if count <= 0:
        return []
    if budget >= count:
        return list(range(count))
    if budget == 1:
        return [0]
    step = (count - 1) / (budget - 1)
    return [round(i * step) for i in range(budget)]


def select_uniform(
    candidates: list[CandidateLike],
    *,
    budget: int,
    seed: int,
    sharpness_by_index: dict[int, float] | None = None,
    diff_by_index: dict[int, float] | None = None,
    diff_threshold: float | None = None,
) -> list[FrameRecord]:
    """Select up to `budget` frames uniformly and describe the fate of every candidate.

    Args:
        candidates: candidates in increasing `src_index` order.
        budget: how many frames to select; if there are too few candidates, all are taken.
        seed: **does not affect the result.** The uniform grid is deterministic by itself,
            it has no tie-breaks. The parameter is present for a uniform look of all
            selection strategies and for recording in the manifest (FR-011); no
            randomness is used here.
        sharpness_by_index: sharpness by `src_index`; what is absent from the dict is `None`.
            Neither `0.0` (zero would mean a measured zero sharpness, FR-036) nor `nan`
            (`nan` cannot be expressed in strict JSON: `artifacts.write_json` writes with
            `allow_nan=False` and would fail when writing `frames.json`). The pipeline must
            supply metrics for all candidates; `None` marks a skip, not a measured value.
        diff_by_index: already computed differences from the previous selected frame by
            `src_index` (the metrics stage knows the selection grid because it is
            deterministic). An extension of the fixed signature: optional, old calls
            remain valid.
        diff_threshold: **not applied in the baseline.** The redundancy threshold for
            neighbouring viewpoints belongs to the `quality_nonredundant` strategy
            (`v3d.select.quality`, task T046); here it is accepted only for signature
            uniformity and ignored.

    Returns:
        Records for **all** candidates in the original order. Unselected ones have
        `selected=False`, `reject_reason="budget_exhausted"` and `image_file=None`.

    Raises:
        ValueError: `budget <= 0`.
    """
    positions = uniform_positions(len(candidates), budget)
    chosen = set(positions)
    sharpness_by_index = sharpness_by_index or {}
    diff_by_index = diff_by_index or {}

    records: list[FrameRecord] = []
    seq = 0
    first_selected_seen = False
    for pos, candidate in enumerate(candidates):
        src_index = int(candidate.src_index)
        selected = pos in chosen
        if selected:
            frame_id = make_frame_id(seq, src_index)
            seq += 1
        else:
            frame_id = make_frame_id(None, src_index)

        diff = diff_by_index.get(src_index)
        if selected and not first_selected_seen:
            # The first selected frame has nothing to compare with.
            diff = None
            first_selected_seen = True

        records.append(
            FrameRecord(
                frame_id=frame_id,
                src_index=src_index,
                timestamp_us=int(candidate.timestamp_us),
                sharpness=_as_float_or_none(sharpness_by_index.get(src_index)),
                diff_to_prev_selected=_as_float_or_none(diff),
                selected=selected,
                reject_reason=None if selected else REJECT_BUDGET_EXHAUSTED,
                image_file=f"{frame_id}.jpg" if selected else None,
            )
        )
    return records


def selection_summary(records: list[FrameRecord], *, budget: int, seed: int) -> dict:
    """Selection summary for `manifest.json` (FR-010): method, parameters, rejection reasons."""
    reasons: dict[str, int] = {}
    for record in records:
        if record.reject_reason is not None:
            reasons[record.reject_reason] = reasons.get(record.reject_reason, 0) + 1
    return {
        "selector": "uniform",
        "params": {"budget": budget, "seed": seed},
        "candidates_considered": len(records),
        "selected": sum(1 for r in records if r.selected),
        "reject_reasons": reasons,
    }
