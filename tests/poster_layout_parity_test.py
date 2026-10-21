import os
import pathlib
from typing import Tuple

import evaluate
import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

from tests.conftest import poster_layout_eval

WIDTH = 513
HEIGHT = 750
POSTER_ATOL = 1e-6
SMALL_SAMPLE_COUNT = 10


def _metric_path(metrics_dir: pathlib.Path, name: str) -> str:
    return os.fspath(metrics_dir / name / f"{name.replace('_', '-')}.py")


@pytest.fixture
def small_poster_data(
    test_fixture_dir: pathlib.Path,
) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    boxes = np.asarray(
        torch.load(test_fixture_dir / "poster_layout_boxes.pt", weights_only=False)[
            :SMALL_SAMPLE_COUNT
        ],
        dtype=np.float32,
    )
    labels = np.asarray(
        torch.load(test_fixture_dir / "poster_layout_clses.pt", weights_only=False)[
            :SMALL_SAMPLE_COUNT
        ],
        dtype=np.int64,
    )
    names = torch.load(
        test_fixture_dir / "poster_layout_test_order.pt", weights_only=False
    )[:SMALL_SAMPLE_COUNT]
    return boxes, labels, names


def _pixel_boxes(boxes: np.ndarray, width: int = WIDTH, height: int = HEIGHT):
    pixel_boxes = np.array(boxes, copy=True)
    pixel_boxes[..., ::2] *= width
    pixel_boxes[..., 1::2] *= height
    return pixel_boxes


def _filtered_labels(labels: np.ndarray, boxes: np.ndarray):
    return poster_layout_eval.getRidOfInvalid(
        (WIDTH, HEIGHT), labels.copy(), _pixel_boxes(boxes)
    )


def _poster_dataset(
    tmp_path: pathlib.Path, test_fixture_dir: pathlib.Path
) -> pathlib.Path:
    source = test_fixture_dir / "PKU_PosterLayout_small" / "test"
    target = tmp_path / "Dataset" / "test"
    target.mkdir(parents=True)
    for directory_name in (
        "image_canvas",
        "saliencymaps_pfpn",
        "saliencymaps_basnet",
    ):
        (target / directory_name).symlink_to(
            source / directory_name, target_is_directory=True
        )
    return target


def _fixture_paths(
    test_fixture_dir: pathlib.Path, names: list[str], directory_name: str
) -> list[list[str]]:
    directory = test_fixture_dir / "PKU_PosterLayout_small" / "test" / directory_name
    return [[str(directory / name)] for name in names]


def test_validity_matches_poster_layout(metrics_dir: pathlib.Path, small_poster_data):
    boxes, labels, _ = small_poster_data
    actual = evaluate.load(
        path=_metric_path(metrics_dir, "layout_validity"),
        canvas_width=WIDTH,
        canvas_height=HEIGHT,
    ).compute(predictions=boxes, gold_labels=labels)
    expected = poster_layout_eval.metrics_val(
        (WIDTH, HEIGHT), labels, _pixel_boxes(boxes)
    )
    assert actual == expected


def test_overlay_matches_poster_layout(metrics_dir: pathlib.Path, small_poster_data):
    boxes, labels, _ = small_poster_data
    actual = evaluate.load(
        path=_metric_path(metrics_dir, "layout_overlay"),
        canvas_width=WIDTH,
        canvas_height=HEIGHT,
    ).compute(predictions=boxes, gold_labels=labels)
    expected = poster_layout_eval.metrics_ove(
        _filtered_labels(labels, boxes), _pixel_boxes(boxes)
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=POSTER_ATOL)


def test_underlay_effectiveness_matches_poster_layout(
    metrics_dir: pathlib.Path, small_poster_data
):
    boxes, labels, _ = small_poster_data
    actual = evaluate.load(
        path=_metric_path(metrics_dir, "layout_underlay_effectiveness"),
        canvas_width=WIDTH,
        canvas_height=HEIGHT,
    ).compute(predictions=boxes, gold_labels=labels)
    filtered_labels = _filtered_labels(labels, boxes)
    pixel_boxes = _pixel_boxes(boxes)
    expected = {
        "und_l": poster_layout_eval.metrics_und_l(filtered_labels, pixel_boxes),
        "und_s": poster_layout_eval.metrics_und_s(filtered_labels, pixel_boxes),
    }
    for key in ("und_l", "und_s"):
        np.testing.assert_allclose(
            actual[key], expected[key], rtol=0.0, atol=POSTER_ATOL
        )


def test_unreadability_differs_from_original_due_to_uint8_sobel_clipping_and_overflow(
    metrics_dir: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    image_path = tmp_path / "Dataset" / "test" / "image_canvas" / "vertical-stripe.png"
    image_path.parent.mkdir(parents=True)
    image = Image.new("L", (WIDTH, HEIGHT), 0)
    ImageDraw.Draw(image).rectangle((128, 0, 255, HEIGHT - 1), fill=255)
    image.save(image_path)

    boxes = np.array([[[0.0, 0.0, 1.0, 1.0]]])
    labels = np.array([[[1]]])
    actual = evaluate.load(
        path=_metric_path(metrics_dir, "layout_unreadability"),
        canvas_width=WIDTH,
        canvas_height=HEIGHT,
    ).compute(
        predictions=boxes,
        gold_labels=labels,
        image_canvases=[[str(image_path)]],
    )
    monkeypatch.chdir(tmp_path)
    original = poster_layout_eval.metrics_rea(
        ["vertical-stripe.png"], labels.copy(), _pixel_boxes(boxes)
    )

    # Float Sobel keeps both signed stripe edges; uint8 clipping/overflow keeps one.
    assert actual == pytest.approx(4 / WIDTH, abs=1e-12)
    assert original == pytest.approx(2 / WIDTH, abs=1e-12)
    assert actual != original


def test_occlusion_matches_poster_layout(
    metrics_dir: pathlib.Path,
    test_fixture_dir: pathlib.Path,
    small_poster_data,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    boxes, labels, names = small_poster_data
    _poster_dataset(tmp_path, test_fixture_dir)
    monkeypatch.chdir(tmp_path)
    actual = evaluate.load(
        path=_metric_path(metrics_dir, "layout_occlusion"),
        canvas_width=WIDTH,
        canvas_height=HEIGHT,
    ).compute(
        predictions=boxes,
        gold_labels=labels,
        saliency_maps_1=_fixture_paths(
            test_fixture_dir,
            [name.replace(".", "_pred.") for name in names],
            "saliencymaps_pfpn",
        ),
        saliency_maps_2=_fixture_paths(test_fixture_dir, names, "saliencymaps_basnet"),
    )
    expected = poster_layout_eval.metrics_occ(
        names, _filtered_labels(labels, boxes), _pixel_boxes(boxes)
    )
    assert actual == expected


def test_utility_matches_poster_layout(
    metrics_dir: pathlib.Path,
    test_fixture_dir: pathlib.Path,
    small_poster_data,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    boxes, labels, names = small_poster_data
    _poster_dataset(tmp_path, test_fixture_dir)
    monkeypatch.chdir(tmp_path)
    actual = evaluate.load(
        path=_metric_path(metrics_dir, "layout_utility"),
        canvas_width=WIDTH,
        canvas_height=HEIGHT,
    ).compute(
        predictions=boxes,
        gold_labels=labels,
        saliency_maps_1=_fixture_paths(
            test_fixture_dir,
            [name.replace(".", "_pred.") for name in names],
            "saliencymaps_pfpn",
        ),
        saliency_maps_2=_fixture_paths(test_fixture_dir, names, "saliencymaps_basnet"),
    )
    expected = poster_layout_eval.metrics_uti(
        names, _filtered_labels(labels, boxes), _pixel_boxes(boxes)
    )
    assert actual == expected


def test_non_alignment_differs_from_original_due_to_global_minimum_loop_bug(
    metrics_dir: pathlib.Path,
):
    boxes = np.array(
        [
            [
                [0.1, 0.1, 0.3, 0.3],
                [0.15, 0.5, 0.35, 0.7],
                [0.5, 0.2, 0.7, 0.4],
            ]
        ]
    )
    labels = np.array([[[1], [1], [1]]])
    actual = evaluate.load(
        path=_metric_path(metrics_dir, "layout_non_alignment"),
        canvas_width=WIDTH,
        canvas_height=HEIGHT,
    ).compute(predictions=boxes, gold_labels=labels)
    original = poster_layout_eval.metrics_ali(
        _filtered_labels(labels, boxes), _pixel_boxes(boxes)
    )

    # The fixed definition sums each element's nearest penalty; PosterLayout
    # repeats the global minimum for all three elements.
    assert actual == pytest.approx(0.0903102799829795, abs=1e-12)
    assert original == pytest.approx(0.06682918413345675, abs=1e-12)
    assert actual != original


def test_underlay_empty_batch_difference_is_explicit(metrics_dir: pathlib.Path):
    boxes = np.array([[[0.1, 0.1, 0.9, 0.9], [0.2, 0.2, 0.8, 0.8]]])
    labels = np.array([[[1], [2]]])
    with pytest.warns(RuntimeWarning):
        actual = evaluate.load(
            path=_metric_path(metrics_dir, "layout_underlay_effectiveness"),
            canvas_width=WIDTH,
            canvas_height=HEIGHT,
        ).compute(predictions=boxes, gold_labels=labels)
    expected_l = poster_layout_eval.metrics_und_l(labels.copy(), _pixel_boxes(boxes))
    expected_s = poster_layout_eval.metrics_und_s(labels.copy(), _pixel_boxes(boxes))
    assert np.isnan(actual["und_l"])
    assert np.isnan(actual["und_s"])
    assert expected_l == 0
    assert expected_s == 0


def test_underlay_tautological_right_edge_condition_is_explicit(
    metrics_dir: pathlib.Path,
):
    boxes = np.array([[[0.1, 0.1, 0.2, 0.3], [0.15, 0.15, 0.4, 0.25]]])
    labels = np.array([[[3], [1]]])
    actual = evaluate.load(
        path=_metric_path(metrics_dir, "layout_underlay_effectiveness"),
        canvas_width=WIDTH,
        canvas_height=HEIGHT,
    ).compute(predictions=boxes, gold_labels=labels)
    expected = poster_layout_eval.metrics_und_s(labels.copy(), _pixel_boxes(boxes))
    assert boxes[0, 1, 2] > boxes[0, 0, 2]
    assert actual["und_s"] == 1
    assert expected == 1


def test_utility_zero_utility_samples_match_original_after_denominator_fix(
    metrics_dir: pathlib.Path, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    white = tmp_path / "white.png"
    black = tmp_path / "black.png"
    Image.new("L", (WIDTH, HEIGHT), 255).save(white)
    Image.new("L", (WIDTH, HEIGHT), 0).save(black)
    names = ["black.png", "white.png"]
    boxes = np.array([[[0.0, 0.0, 1.0, 1.0]], [[0.0, 0.0, 1.0, 1.0]]])
    labels = np.array([[[1]], [[1]]])
    map_1 = [[str(black)], [str(white)]]
    map_2 = [[str(black)], [str(white)]]
    actual = evaluate.load(
        path=_metric_path(metrics_dir, "layout_utility"),
        canvas_width=WIDTH,
        canvas_height=HEIGHT,
    ).compute(
        predictions=boxes,
        gold_labels=labels,
        saliency_maps_1=map_1,
        saliency_maps_2=map_2,
    )
    dataset = tmp_path / "Dataset" / "test"
    (dataset / "saliencymaps_pfpn").mkdir(parents=True)
    (dataset / "saliencymaps_basnet").mkdir()
    for directory_name, suffix in (
        ("saliencymaps_pfpn", "_pred"),
        ("saliencymaps_basnet", ""),
    ):
        output = dataset / directory_name
        for name, source in (("white.png", white), ("black.png", black)):
            target_name = name.replace(".", f"{suffix}.")
            target = output / target_name
            if target.exists() or target.is_symlink():
                target.unlink()
            target.symlink_to(source)
    monkeypatch.chdir(tmp_path)
    expected = poster_layout_eval.metrics_uti(names, labels.copy(), _pixel_boxes(boxes))
    assert actual == expected == 0.5

    zero_names = ["white.png", "white.png"]
    zero_actual = evaluate.load(
        path=_metric_path(metrics_dir, "layout_utility"),
        canvas_width=WIDTH,
        canvas_height=HEIGHT,
    ).compute(
        predictions=boxes,
        gold_labels=labels,
        saliency_maps_1=[[str(white)], [str(white)]],
        saliency_maps_2=[[str(white)], [str(white)]],
    )
    zero_original = poster_layout_eval.metrics_uti(
        zero_names, labels.copy(), _pixel_boxes(boxes)
    )
    assert zero_actual == zero_original == 0.0


def test_non_alignment_canvas_difference_is_explicit(metrics_dir: pathlib.Path):
    boxes = np.array([[[0.1, 0.1, 0.4, 0.4], [0.2, 0.5, 0.5, 0.8]]])
    labels = np.array([[[1], [2]]])
    actual = evaluate.load(
        path=_metric_path(metrics_dir, "layout_non_alignment"),
        canvas_width=100,
        canvas_height=100,
    ).compute(predictions=boxes, gold_labels=labels)
    pixel_boxes = _pixel_boxes(boxes, width=100, height=100)
    expected = poster_layout_eval.metrics_ali(
        poster_layout_eval.getRidOfInvalid(
            (100, 100), labels.copy(), pixel_boxes.copy()
        ),
        pixel_boxes,
    )
    assert actual == pytest.approx(0.09151498112135023, abs=1e-12)
    assert expected == pytest.approx(0.01709876011177775, abs=1e-12)
    assert actual > expected
