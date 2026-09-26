"""Unit tests for frame metrics (part of T023): loading, sharpness, frame difference.

Rationale for the metrics — research.md §R-7; requirements — spec.md FR-006, FR-007.

All images are synthetic and created right in the test: for :func:`v3d.metrics.load_gray`
real JPEG files are written to disk in ``tmp_path``, the other functions receive arrays.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageFilter

from v3d.metrics import DIFF_SIZE, frame_difference, load_gray, sharpness

# --------------------------------------------------------------------------------------
# Image builder helpers


def _checkerboard(size: int = 128, cell: int = 8) -> np.ndarray:
    """Checkerboard of ``cell``×``cell`` cells, values 0 and 255, ``uint8``."""
    indices = np.arange(size) // cell
    pattern = (indices[:, None] + indices[None, :]) % 2
    return (pattern * 255).astype(np.uint8)


def _half_black_half_white(size: int = 128) -> np.ndarray:
    """Left half black, right half white; values exactly 0.0 and 1.0, ``float32``."""
    image = np.zeros((size, size), dtype=np.float32)
    image[:, size // 2 :] = 1.0
    return image


def _square_on_black(size: int = 128, square: int = 32, shift: int = 0) -> np.ndarray:
    """White square on a black background, shifted horizontally by ``shift`` pixels."""
    image = np.zeros((size, size), dtype=np.float32)
    top = size // 2 - square // 2
    left = top + shift
    image[top : top + square, left : left + square] = 1.0
    return image


def _write_jpeg(path, array: np.ndarray) -> None:
    """Save a ``uint8`` array as a JPEG without changing its size."""
    Image.fromarray(array, mode="L").save(path, format="JPEG", quality=95)


# --------------------------------------------------------------------------------------
# load_gray


def test_load_gray_downscales_to_max_side_preserving_proportions(tmp_path):
    """The long side equals max_side, the short one is recomputed with the same scale."""
    path = tmp_path / "wide.jpg"
    _write_jpeg(path, _checkerboard(size=128)[:96, :].repeat(4, axis=0).repeat(4, axis=1))
    # The result is 384×512 (height×width): check that the source is really non-square.
    with Image.open(path) as probe:
        assert probe.size == (512, 384)

    gray = load_gray(path, max_side=256)

    assert gray.shape == (192, 256)
    assert gray.shape[1] / gray.shape[0] == pytest.approx(512 / 384)


def test_load_gray_does_not_enlarge_small_image(tmp_path):
    """An image smaller than max_side keeps its size."""
    path = tmp_path / "small.jpg"
    _write_jpeg(path, _checkerboard(size=64, cell=8))

    gray = load_gray(path, max_side=256)

    assert gray.shape == (64, 64)


def test_load_gray_returns_float32_in_zero_one_range(tmp_path):
    """Type and value range are part of the contract: the other metrics rely on them."""
    path = tmp_path / "board.jpg"
    _write_jpeg(path, _checkerboard())

    gray = load_gray(path)

    assert gray.dtype == np.float32
    assert gray.ndim == 2
    assert float(gray.min()) >= 0.0
    assert float(gray.max()) <= 1.0
    assert np.isfinite(gray).all()


def test_load_gray_is_deterministic(tmp_path):
    """A repeated call on the same file gives a bit-identical array."""
    path = tmp_path / "board.jpg"
    _write_jpeg(path, _checkerboard(size=300, cell=7))

    first = load_gray(path, max_side=256)
    second = load_gray(path, max_side=256)

    assert np.array_equal(first, second)


def test_load_gray_rejects_invalid_max_side(tmp_path):
    path = tmp_path / "board.jpg"
    _write_jpeg(path, _checkerboard(size=32, cell=4))

    with pytest.raises(ValueError, match="max_side"):
        load_gray(path, max_side=0)


# --------------------------------------------------------------------------------------
# sharpness


def test_sharpness_sharp_image_scores_higher_than_blurred(tmp_path):
    """The main property of the metric: blurring must lower it."""
    board = _checkerboard(size=128, cell=8)
    sharp_path = tmp_path / "sharp.jpg"
    blurred_path = tmp_path / "blurred.jpg"
    _write_jpeg(sharp_path, board)
    Image.fromarray(board, mode="L").filter(ImageFilter.GaussianBlur(radius=3)).save(
        blurred_path, format="JPEG", quality=95
    )

    sharp_value = sharpness(load_gray(sharp_path))
    blurred_value = sharpness(load_gray(blurred_path))

    assert sharp_value > blurred_value * 5


def test_sharpness_of_uniform_image_is_zero():
    """Degenerate input must not give NaN or division by zero."""
    flat = np.full((64, 64), 0.42, dtype=np.float32)

    value = sharpness(flat)

    assert np.isfinite(value)
    assert value == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("shape", [(0, 0), (1, 1), (2, 5), (3, 2)])
def test_sharpness_on_too_small_input_is_zero(shape):
    """If no interior remains after dropping the border — exactly 0.0, no NaN."""
    value = sharpness(np.zeros(shape, dtype=np.float32))

    assert value == 0.0


def test_sharpness_is_non_negative_and_finite_on_noise():
    rng = np.random.default_rng(seed=17)
    noise = rng.random((64, 64), dtype=np.float32)

    value = sharpness(noise)

    assert np.isfinite(value)
    assert value > 0.0


def test_sharpness_rejects_non_two_dimensional_input():
    with pytest.raises(ValueError, match="two-dimensional"):
        sharpness(np.zeros((8, 8, 3), dtype=np.float32))


# --------------------------------------------------------------------------------------
# frame_difference


def test_frame_difference_of_identical_frames_is_exactly_zero():
    board = _checkerboard().astype(np.float32) / 255.0

    assert frame_difference(board, board) == 0.0


def test_frame_difference_of_frame_and_its_inverse_is_one_without_resampling():
    """The frame is already 64×64, no resizing needed — opposite frames give exactly 1.0."""
    image = _half_black_half_white(size=DIFF_SIZE)

    assert frame_difference(image, 1.0 - image) == pytest.approx(1.0, abs=1e-6)


def test_frame_difference_of_frame_and_its_inverse_is_close_to_one_after_resampling():
    """A 128×128 frame is resized to 64×64, and values blend at the black-white boundary.

    So the result is slightly below one (the observed value is 0.9921875). This is a property
    of resampling, not of the metric: the upper bound of the range is only approximately reachable.
    """
    image = _half_black_half_white(size=128)

    value = frame_difference(image, 1.0 - image)

    assert value == pytest.approx(1.0, abs=0.02)
    assert value <= 1.0


def test_frame_difference_handles_frames_of_different_sizes():
    """Resizing to 64×64 happens inside the function, so the sizes may differ."""
    big = _half_black_half_white(size=256)
    small = _half_black_half_white(size=96)

    value = frame_difference(big, small)

    assert np.isfinite(value)
    assert 0.0 <= value <= 1.0
    assert value == pytest.approx(0.0, abs=0.05)


def test_frame_difference_is_symmetric():
    a = _square_on_black(shift=0)
    b = _square_on_black(shift=17)

    assert frame_difference(a, b) == frame_difference(b, a)


def test_frame_difference_is_within_zero_and_one_on_noise():
    rng = np.random.default_rng(seed=3)
    a = rng.random((80, 120), dtype=np.float32)
    b = rng.random((37, 51), dtype=np.float32)

    value = frame_difference(a, b)

    assert 0.0 <= value <= 1.0


def test_frame_difference_of_uniform_frames_equals_brightness_difference():
    a = np.full((50, 50), 0.25, dtype=np.float32)
    b = np.full((70, 30), 0.75, dtype=np.float32)

    assert frame_difference(a, b) == pytest.approx(0.5, abs=1e-6)


def test_frame_difference_rejects_empty_frame():
    """A silent 0.0 for a missing frame would mean "the frames match" — that would be false."""
    with pytest.raises(ValueError, match="empty"):
        frame_difference(np.zeros((0, 0), dtype=np.float32), _square_on_black())


def test_frame_difference_grows_with_shift():
    """The property the metric is used for in selection: more shift, larger number.

    This is the meaning of the proxy from FR-007: the metric reacts to the frame having
    become different, not to the camera having moved somewhere.
    """
    base = _square_on_black(shift=0)
    shifts = [2, 8, 16, 32]

    values = [frame_difference(base, _square_on_black(shift=shift)) for shift in shifts]

    assert values[0] > 0.0
    assert all(later > earlier for earlier, later in zip(values[:-1], values[1:], strict=True))


def test_frame_difference_resizes_frames_to_exactly_64x64():
    """The resize target is fixed by a constant: comparability of the numbers depends on it."""
    assert DIFF_SIZE == 64
