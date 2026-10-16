import os
import pathlib
import pickle

import evaluate
import numpy as np
import pytest
import torch
from tests.conftest import load_vendor_package
from trainer.helpers.metric import (
    compute_alignment,
    compute_average_iou,
    compute_generative_model_scores,
    compute_maximum_iou,
    compute_overlap,
)


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


def _assert_torch_reference(actual, expected, atol: float = 1e-12):
    for key, value in expected.items():
        np.testing.assert_allclose(
            np.asarray(actual[key]), value.detach().numpy(), rtol=0.0, atol=atol
        )


def test_vendor_loader_reports_missing_submodule(tmp_path: pathlib.Path):
    with pytest.raises(RuntimeError, match="vendored package 'missing' is missing"):
        load_vendor_package("missing", tmp_path / "missing")


@pytest.fixture
def seeded_batch():
    generator = np.random.default_rng(264)
    counts = np.arange(1, 9)
    bbox = generator.random((len(counts), counts[-1], 4))
    bbox[..., 2:] = generator.uniform(0.05, 0.5, bbox[..., 2:].shape)
    mask = np.arange(counts[-1]) < counts[:, None]
    return bbox, mask


@pytest.mark.parametrize("fixture_number", (1, 2, 3))
def test_alignment_matches_layout_dm_fixtures(
    metrics_dir: pathlib.Path,
    test_fixture_dir: pathlib.Path,
    fixture_number: int,
):
    bbox = np.load(test_fixture_dir / f"batch_bbox{fixture_number}.npy").astype(
        np.float64
    )
    mask = np.load(test_fixture_dir / f"batch_mask{fixture_number}.npy")
    actual = evaluate.load(path=_metric_path(metrics_dir, "layout_alignment")).compute(
        bbox=bbox, mask=mask
    )
    expected = compute_alignment(torch.from_numpy(bbox), torch.from_numpy(mask))
    _assert_torch_reference(actual, expected)


def test_alignment_matches_layout_dm_seeded_random(metrics_dir, seeded_batch):
    bbox, mask = seeded_batch
    actual = evaluate.load(path=_metric_path(metrics_dir, "layout_alignment")).compute(
        bbox=bbox, mask=mask
    )
    expected = compute_alignment(torch.from_numpy(bbox), torch.from_numpy(mask))
    _assert_torch_reference(actual, expected)


@pytest.mark.parametrize("fixture_number", (1, 2, 3))
def test_overlap_matches_layout_dm_fixtures(
    metrics_dir: pathlib.Path,
    test_fixture_dir: pathlib.Path,
    fixture_number: int,
):
    bbox = np.load(test_fixture_dir / f"batch_bbox{fixture_number}.npy").astype(
        np.float64
    )
    mask = np.load(test_fixture_dir / f"batch_mask{fixture_number}.npy")
    actual = evaluate.load(path=_metric_path(metrics_dir, "layout_overlap")).compute(
        bbox=bbox, mask=mask
    )
    expected = compute_overlap(torch.from_numpy(bbox), torch.from_numpy(mask))
    _assert_torch_reference(actual, expected)


def test_overlap_matches_layout_dm_seeded_random(metrics_dir, seeded_batch):
    bbox, mask = seeded_batch
    actual = evaluate.load(path=_metric_path(metrics_dir, "layout_overlap")).compute(
        bbox=bbox, mask=mask
    )
    expected = compute_overlap(torch.from_numpy(bbox), torch.from_numpy(mask))
    _assert_torch_reference(actual, expected)


def test_maximum_iou_matches_layout_dm_fixture(
    metrics_dir: pathlib.Path, test_fixture_dir: pathlib.Path
):
    layouts1 = _load_layouts(test_fixture_dir, "layouts_main")[:128]
    layouts2 = _load_layouts(test_fixture_dir, "layouts_sub")[:128]
    actual = evaluate.load(
        path=_metric_path(metrics_dir, "layout_maximum_iou")
    ).compute(layouts1=_layout_dicts(layouts1), layouts2=_layout_dicts(layouts2))
    expected = compute_maximum_iou(_layout_tuples(layouts1), _layout_tuples(layouts2))
    assert actual == expected


def test_maximum_iou_matches_layout_dm_seeded_random(metrics_dir):
    generator = np.random.default_rng(264)
    layouts1, layouts2 = [], []
    for count in range(1, 8):
        categories = np.arange(count) % 3
        layouts1.append(
            {"bboxes": generator.random((count, 4)), "categories": categories}
        )
        layouts2.append(
            {"bboxes": generator.random((count, 4)), "categories": categories}
        )
    actual = evaluate.load(
        path=_metric_path(metrics_dir, "layout_maximum_iou")
    ).compute(layouts1=_layout_dicts(layouts1), layouts2=_layout_dicts(layouts2))
    expected = compute_maximum_iou(_layout_tuples(layouts1), _layout_tuples(layouts2))
    assert actual == expected


def test_average_iou_matches_layout_dm_fixture(
    metrics_dir: pathlib.Path, test_fixture_dir: pathlib.Path
):
    layouts = _load_layouts(test_fixture_dir, "layouts_sub")[:128]
    actual = evaluate.load(
        path=_metric_path(metrics_dir, "layout_average_iou")
    ).compute(layouts=_layout_dicts(layouts))
    expected = compute_average_iou(_layout_tuples(layouts))
    assert actual["average-iou_BLT"] == expected["average_iou-BLT"]
    assert actual["average-iou_VTN"] == expected["average_iou-VTN"]


def test_average_iou_matches_layout_dm_seeded_random(metrics_dir):
    generator = np.random.default_rng(264)
    layouts = []
    for count in range(1, 8):
        bboxes = generator.random((count, 4))
        bboxes[:, 2:] = generator.uniform(0.05, 0.5, bboxes[:, 2:].shape)
        layouts.append({"bboxes": bboxes, "categories": np.arange(count) % 3})
    actual = evaluate.load(
        path=_metric_path(metrics_dir, "layout_average_iou")
    ).compute(layouts=_layout_dicts(layouts))
    expected = compute_average_iou(_layout_tuples(layouts))
    assert actual["average-iou_BLT"] == expected["average_iou-BLT"]
    assert actual["average-iou_VTN"] == expected["average_iou-VTN"]


def test_generative_model_scores_match_layout_dm_fixture(
    metrics_dir: pathlib.Path, test_fixture_dir: pathlib.Path
):
    feats_real = np.load(test_fixture_dir / "feats_real.npy")[:64].astype(np.float64)
    feats_fake = np.load(test_fixture_dir / "feats_fake.npy")[:64].astype(np.float64)
    actual = evaluate.load(
        path=_metric_path(metrics_dir, "layout_generative_model_scores")
    ).compute(feats_real=feats_real, feats_fake=feats_fake)
    expected = compute_generative_model_scores(
        [torch.from_numpy(feats_real)], [torch.from_numpy(feats_fake)]
    )
    assert actual == expected


def test_generative_model_scores_match_layout_dm_seeded_random(metrics_dir):
    generator = np.random.default_rng(264)
    feats_real = generator.normal(size=(32, 16))
    feats_fake = generator.normal(size=(32, 16))
    actual = evaluate.load(
        path=_metric_path(metrics_dir, "layout_generative_model_scores")
    ).compute(feats_real=feats_real, feats_fake=feats_fake)
    expected = compute_generative_model_scores(
        [torch.from_numpy(feats_real)], [torch.from_numpy(feats_fake)]
    )
    assert actual == expected
