import os
import pathlib
import pickle

import evaluate
import numpy as np
import pytest
import torch
from tests.conftest import (
    const_layout_compute_alignment,
    const_layout_compute_maximum_iou_for_groups,
    const_layout_compute_overlap,
    const_layout_get_cond2layouts,
)
from trainer.helpers.metric import (
    compute_alignment as layout_dm_compute_alignment,
    compute_maximum_iou as layout_dm_compute_maximum_iou,
    compute_overlap as layout_dm_compute_overlap,
)


ATOL = 1e-12
MAXIMUM_IOU_ATOL = 1e-12
ALIGNMENT_CONST_LAYOUT_DIFFS = {
    1: 0.7047896154460861,
    2: 0.63911725936018,
    3: 0.07302368559612943,
}
SEED = 264
SEEDED_ALIGNMENT_CONST_LAYOUT_DIFF = 0.13075950522778584


def _metric_path(metrics_dir: pathlib.Path, name: str) -> str:
    return os.fspath(metrics_dir / name / f"{name.replace('_', '-')}.py")


def _load_layouts(test_fixture_dir: pathlib.Path, name: str):
    with (test_fixture_dir / f"{name}.pkl").open("rb") as file:
        return pickle.load(file)


def _layout_dicts(layouts):
    return [
        {
            "bboxes": np.asarray(layout["bboxes"], dtype=np.float64),
            "categories": np.asarray(layout["categories"], dtype=np.int64),
        }
        for layout in layouts
    ]


def _layout_tuples(layouts):
    return [
        (layout["bboxes"], layout["categories"]) for layout in _layout_dicts(layouts)
    ]


def _const_layout_maximum_iou(layouts1, layouts2) -> float:
    groups1 = const_layout_get_cond2layouts(layouts1)
    groups2 = const_layout_get_cond2layouts(layouts2)
    scores = [
        const_layout_compute_maximum_iou_for_groups((groups1[key], groups2[key]))
        for key in groups1.keys() & groups2.keys()
    ]
    return np.asarray([score for group in scores for score in group]).mean().item()


def _assert_array_parity(actual, expected) -> None:
    np.testing.assert_allclose(
        np.asarray(actual), expected.detach().numpy(), rtol=0.0, atol=ATOL
    )


def _alignment_inputs(test_fixture_dir: pathlib.Path, fixture_number: int):
    bbox = np.load(test_fixture_dir / f"batch_bbox{fixture_number}.npy").astype(
        np.float64
    )
    mask = np.load(test_fixture_dir / f"batch_mask{fixture_number}.npy")
    return bbox, mask


@pytest.mark.parametrize("fixture_number", (1, 2, 3))
def test_alignment_three_way_fixture(
    metrics_dir: pathlib.Path,
    test_fixture_dir: pathlib.Path,
    fixture_number: int,
):
    bbox, mask = _alignment_inputs(test_fixture_dir, fixture_number)
    actual = evaluate.load(path=_metric_path(metrics_dir, "layout_alignment")).compute(
        bbox=bbox, mask=mask
    )
    torch_bbox = torch.from_numpy(bbox)
    torch_mask = torch.from_numpy(mask)
    layout_dm = layout_dm_compute_alignment(torch_bbox, torch_mask)
    const_layout = const_layout_compute_alignment(torch_bbox, torch_mask)

    for key, value in layout_dm.items():
        _assert_array_parity(actual[key], value)

    # const_layout returns the normalized AC-LayoutGAN score, corresponding to
    # LayoutGAN++ here. It masks padded columns too; layout-dm masks rows only.
    _assert_array_parity(
        actual["alignment-LayoutGAN++"], layout_dm["alignment-LayoutGAN++"]
    )
    difference = np.max(
        np.abs(const_layout.numpy() - layout_dm["alignment-LayoutGAN++"].numpy())
    )
    assert difference == pytest.approx(
        ALIGNMENT_CONST_LAYOUT_DIFFS[fixture_number], abs=ATOL
    )
    assert np.max(
        np.abs(const_layout.numpy() - np.asarray(actual["alignment-LayoutGAN++"]))
    ) == pytest.approx(ALIGNMENT_CONST_LAYOUT_DIFFS[fixture_number], abs=ATOL)


def test_alignment_three_way_seeded_random(metrics_dir):
    generator = np.random.default_rng(SEED)
    counts = np.arange(1, 9)
    bbox = generator.random((len(counts), counts[-1], 4))
    bbox[..., 2:] = generator.uniform(0.05, 0.5, bbox[..., 2:].shape)
    mask = np.arange(counts[-1]) < counts[:, None]

    actual = evaluate.load(path=_metric_path(metrics_dir, "layout_alignment")).compute(
        bbox=bbox, mask=mask
    )
    torch_bbox = torch.from_numpy(bbox)
    torch_mask = torch.from_numpy(mask)
    layout_dm = layout_dm_compute_alignment(torch_bbox, torch_mask)
    const_layout = const_layout_compute_alignment(torch_bbox, torch_mask)
    for key, value in layout_dm.items():
        _assert_array_parity(actual[key], value)
    _assert_array_parity(
        actual["alignment-LayoutGAN++"], layout_dm["alignment-LayoutGAN++"]
    )
    difference = np.max(
        np.abs(const_layout.numpy() - layout_dm["alignment-LayoutGAN++"].numpy())
    )
    assert difference == pytest.approx(SEEDED_ALIGNMENT_CONST_LAYOUT_DIFF, abs=ATOL)
    assert np.max(
        np.abs(const_layout.numpy() - np.asarray(actual["alignment-LayoutGAN++"]))
    ) == pytest.approx(SEEDED_ALIGNMENT_CONST_LAYOUT_DIFF, abs=ATOL)


@pytest.mark.parametrize("fixture_number", (1, 2, 3))
def test_overlap_three_way_fixture(
    metrics_dir: pathlib.Path,
    test_fixture_dir: pathlib.Path,
    fixture_number: int,
):
    bbox, mask = _alignment_inputs(test_fixture_dir, fixture_number)
    actual = evaluate.load(path=_metric_path(metrics_dir, "layout_overlap")).compute(
        bbox=bbox, mask=mask
    )
    torch_bbox = torch.from_numpy(bbox)
    torch_mask = torch.from_numpy(mask)
    layout_dm = layout_dm_compute_overlap(torch_bbox, torch_mask)
    const_layout = const_layout_compute_overlap(torch_bbox, torch_mask)
    for key, value in layout_dm.items():
        _assert_array_parity(actual[key], value)
    _assert_array_parity(actual["overlap-LayoutGAN++"], const_layout)
    np.testing.assert_allclose(
        const_layout.numpy(),
        layout_dm["overlap-LayoutGAN++"].numpy(),
        rtol=0.0,
        atol=ATOL,
    )


def test_overlap_three_way_seeded_random(metrics_dir):
    generator = np.random.default_rng(SEED)
    counts = np.arange(1, 9)
    bbox = generator.random((len(counts), counts[-1], 4))
    bbox[..., 2:] = generator.uniform(0.05, 0.5, bbox[..., 2:].shape)
    mask = np.arange(counts[-1]) < counts[:, None]

    actual = evaluate.load(path=_metric_path(metrics_dir, "layout_overlap")).compute(
        bbox=bbox, mask=mask
    )
    torch_bbox = torch.from_numpy(bbox)
    torch_mask = torch.from_numpy(mask)
    layout_dm = layout_dm_compute_overlap(torch_bbox, torch_mask)
    const_layout = const_layout_compute_overlap(torch_bbox, torch_mask)
    for key, value in layout_dm.items():
        _assert_array_parity(actual[key], value)
    _assert_array_parity(actual["overlap-LayoutGAN++"], const_layout)
    np.testing.assert_allclose(
        const_layout.numpy(),
        layout_dm["overlap-LayoutGAN++"].numpy(),
        rtol=0.0,
        atol=ATOL,
    )


def test_maximum_iou_three_way_fixture(
    metrics_dir: pathlib.Path, test_fixture_dir: pathlib.Path
):
    layouts1 = _load_layouts(test_fixture_dir, "layouts_main")[:128]
    layouts2 = _load_layouts(test_fixture_dir, "layouts_sub")[:128]
    actual = evaluate.load(
        path=_metric_path(metrics_dir, "layout_maximum_iou")
    ).compute(layouts1=_layout_dicts(layouts1), layouts2=_layout_dicts(layouts2))
    layout_dm = layout_dm_compute_maximum_iou(
        _layout_tuples(layouts1), _layout_tuples(layouts2), disable_parallel=True
    )
    const_layout = _const_layout_maximum_iou(
        _layout_tuples(layouts1), _layout_tuples(layouts2)
    )
    np.testing.assert_allclose(
        [actual, layout_dm, const_layout],
        actual,
        rtol=0.0,
        atol=MAXIMUM_IOU_ATOL,
    )


def test_maximum_iou_three_way_seeded_random(metrics_dir):
    generator = np.random.default_rng(2644)
    layouts1 = []
    layouts2 = []
    conditions = (
        np.array([0]),
        np.array([1, 1]),
        np.array([0, 1, 2]),
        np.array([2, 2, 3]),
        np.array([0, 0, 1, 2]),
        np.array([1, 1, 1, 3]),
    )
    for categories in conditions:
        layouts1.append((generator.random((len(categories), 4)), categories.copy()))
        layouts2.append((generator.random((len(categories), 4)), categories.copy()))
    # Unmatched category multisets are ignored by all three references; the
    # matched groups still vary element counts and include unequal collection sizes.
    layouts1.append((generator.random((2, 4)), np.array([9, 9])))
    layouts2.append((generator.random((1, 4)), np.array([8])))

    actual = evaluate.load(
        path=_metric_path(metrics_dir, "layout_maximum_iou")
    ).compute(
        layouts1=_layout_dicts(
            [
                {"bboxes": boxes, "categories": categories}
                for boxes, categories in layouts1
            ]
        ),
        layouts2=_layout_dicts(
            [
                {"bboxes": boxes, "categories": categories}
                for boxes, categories in layouts2
            ]
        ),
    )
    layout_dm = layout_dm_compute_maximum_iou(layouts1, layouts2, disable_parallel=True)
    const_layout = _const_layout_maximum_iou(layouts1, layouts2)
    np.testing.assert_allclose(
        [actual, layout_dm, const_layout],
        actual,
        rtol=0.0,
        atol=MAXIMUM_IOU_ATOL,
    )
