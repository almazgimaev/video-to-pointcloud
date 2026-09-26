"""Frame metrics for the selection stage: sharpness and difference between neighbouring frames.

Rationale for the choice of metrics — research.md §R-7; requirements — spec.md FR-006, FR-007.

The module only **measures**. There are no "sharp/blurry" or "similar/dissimilar" thresholds
here: they are assigned by the selection stage (:mod:`v3d.select`), because a threshold
depends on the frame budget and on the particular video, not on the metric itself.

Both metrics are computed on a downscaled copy of the frame, are deterministic (the
resampling method is set explicitly) and need no dependencies beyond numpy + Pillow.

Input convention: the functions :func:`sharpness` and :func:`frame_difference` expect a
two-dimensional grayscale array with values in the range [0, 1] — exactly what
:func:`load_gray` returns.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

#: Side of the square that frames are resized to in :func:`frame_difference`.
#: Fixed so that the metric value does not depend on the resolution of the source video.
DIFF_SIZE = 64

#: Laplacian kernel applied in :func:`sharpness`. Kept for documentation and tests:
#: the convolution itself is done by adding shifted slices, without matrix multiplication.
LAPLACIAN_KERNEL = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)

# The resampling method is fixed explicitly: the same input must give the same array on
# any machine and in any Pillow version where this method is available.
_RESAMPLE_LOAD = Image.Resampling.LANCZOS
_RESAMPLE_DIFF = Image.Resampling.BILINEAR


def load_gray(path: Path, *, max_side: int = 256) -> np.ndarray:
    """Read a frame and return its downscaled grayscale copy.

    The image is converted to grayscale, then downscaled so that the long side does not
    exceed ``max_side``; proportions are preserved, a side never becomes smaller than one
    pixel. An image smaller than ``max_side`` is not enlarged.

    :param path: path to the frame file (in the pipeline — a JPEG from ``work/frames/``).
    :param max_side: upper bound of the long side of the result in pixels.
    :returns: two-dimensional ``float32`` array with values in [0, 1].
    :raises ValueError: if ``max_side`` is less than one.
    """
    if max_side < 1:
        raise ValueError(f"max_side must be >= 1, got {max_side}")

    with Image.open(path) as img:
        gray = img.convert("L")
        width, height = gray.size
        longest = max(width, height)
        if longest > max_side:
            scale = max_side / longest
            new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
            gray = gray.resize(new_size, _RESAMPLE_LOAD)
        array = np.asarray(gray, dtype=np.float32)

    return array / np.float32(255.0)


def sharpness(gray: np.ndarray) -> float:
    """Estimate frame sharpness as the variance of the Laplacian response.

    Convolution with the kernel ``[[0, 1, 0], [1, -4, 1], [0, 1, 0]]`` is done with numpy.
    The border is handled by dropping it: a one-pixel border is excluded, so the response
    is computed only where the kernel fits entirely.

    The larger the value, the stronger the brightness differences in the frame — that is,
    the sharper it is. The number is not normalised and is comparable only between frames
    of the same video brought to the same size (which is what ``max_side`` in
    :func:`load_gray` is for).

    Behaviour on degenerate input: for a single-colour image the response is zero
    everywhere and the result is ``0.0``; for an empty array or an image smaller than
    3×3 pixels, where no interior remains, ``0.0`` is also returned. ``NaN`` and division
    by zero do not occur here for any input.

    :param gray: two-dimensional grayscale array, values in [0, 1].
    :returns: variance of the Laplacian response, a non-negative number.
    :raises ValueError: if the array is not two-dimensional.
    """
    array = _as_2d_float(gray, name="gray")
    if array.shape[0] < 3 or array.shape[1] < 3:
        return 0.0

    response = (
        array[:-2, 1:-1]
        + array[2:, 1:-1]
        + array[1:-1, :-2]
        + array[1:-1, 2:]
        - 4.0 * array[1:-1, 1:-1]
    )
    return float(response.var())


def frame_difference(a: np.ndarray, b: np.ndarray) -> float:
    """Measure the mean absolute difference of two frames resized to 64×64.

    **This is a proxy for viewpoint change, not a measurement of camera displacement.**
    The metric compares pixel brightness and knows nothing about the shooting geometry.
    A large value only means that the frames differ noticeably from each other: a
    viewpoint change looks like that, and so do a lighting change, a foreign object
    entering the frame, and sensor noise. A small value means the frames are similar; it
    does not follow that the camera stood still. So the value may be presented to the
    user as an observation ("the frames differ little"), but not as a conclusion about
    what happened during shooting.

    Both frames are resized to 64×64 inside the function, so images of any, and even
    different, sizes can be passed in. The metric is symmetric: ``d(a, b) == d(b, a)``.

    Behaviour on degenerate input: identical frames give exactly ``0.0``; single-colour
    frames are handled like any others. An empty array is an error, not zero — silently
    returning ``0.0`` for a missing frame would amount to reporting "the frames match".

    :param a: first frame, two-dimensional grayscale array with values in [0, 1].
    :param b: second frame, same requirements.
    :returns: mean absolute difference in [0, 1].
    :raises ValueError: if an array is not two-dimensional or is empty.
    """
    small_a = _to_diff_size(a, name="a")
    small_b = _to_diff_size(b, name="b")
    return float(np.abs(small_a - small_b).mean())


def _as_2d_float(array: np.ndarray, *, name: str) -> np.ndarray:
    """Convert the input to a two-dimensional ``float32`` array or explain why it is impossible."""
    result = np.asarray(array, dtype=np.float32)
    if result.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array, got ndim={result.ndim}")
    return result


def _to_diff_size(array: np.ndarray, *, name: str) -> np.ndarray:
    """Resize a frame to ``DIFF_SIZE``×``DIFF_SIZE`` with the same deterministic resampling."""
    result = _as_2d_float(array, name=name)
    if result.size == 0:
        raise ValueError(f"{name} must not be an empty array")
    if result.shape == (DIFF_SIZE, DIFF_SIZE):
        return result

    # Mode "F" is a single-channel float32 image; values in [0, 1] are not rescaled.
    image = Image.fromarray(result, mode="F")
    resized = image.resize((DIFF_SIZE, DIFF_SIZE), _RESAMPLE_DIFF)
    return np.asarray(resized, dtype=np.float32)
