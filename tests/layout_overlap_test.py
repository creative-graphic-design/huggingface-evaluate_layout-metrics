import math
import os
import pathlib
from typing import Dict

import evaluate
import numpy as np
import pytest
import torch


@pytest.fixture
def base_dir() -> str:
    return "layout_overlap"


@pytest.fixture
def metric_path(base_dir: str) -> str:
    return os.path.join(base_dir, "layout_overlap.py")


@pytest.fixture
def test_fixture_dir() -> pathlib.Path:
    return pathlib.Path(__file__).parents[1] / "test_fixtures"


def test_metric_random(metric_path: str):
    metric = evaluate.load(path=metric_path)

    batch_bbox = np.random.rand(512, 25, 4)
    batch_mask = np.random.choice(a=[True, False], size=(512, 25))

    metric.add_batch(bbox=batch_bbox, mask=batch_mask)
    scores = metric.compute()
    assert scores is not None


@pytest.mark.parametrize(
    argnames=("num", "expected_scores"),
    argvalues=(
        (
            1,
            {
                "overlap-ACLayoutGAN": 6280.4189453125,
                "overlap-LayoutGAN++": 468.6614685058594,
                "overlap-LayoutGAN": 200.9669189453125,
            },
        ),
        (
            2,
            {
                "overlap-ACLayoutGAN": 6368.20654296875,
                "overlap-LayoutGAN++": 473.6511535644531,
                "overlap-LayoutGAN": 204.1142578125,
            },
        ),
        (
            3,
            {
                "overlap-ACLayoutGAN": 6527.58251953125,
                "overlap-LayoutGAN++": 474.0531005859375,
                "overlap-LayoutGAN": 198.0200653076172,
            },
        ),
    ),
)
def test_metric(
    metric_path: str,
    test_fixture_dir: pathlib.Path,
    num: int,
    expected_scores: Dict[str, float],
):
    batch_bbox = np.load(test_fixture_dir / f"batch_bbox{num}.npy")
    batch_mask = np.load(test_fixture_dir / f"batch_mask{num}.npy")

    #
    # Load Align. metric
    #
    metric = evaluate.load(path=metric_path)

    #
    # Batch processing
    #
    metric.add_batch(bbox=batch_bbox, mask=batch_mask)
    scores = metric.compute()
    assert scores is not None

    for k in expected_scores.keys():
        score = sum(scores[k])
        expected_score = expected_scores[k]
        assert math.isclose(score, expected_score, rel_tol=1e-5)

    #
    # Reload the metric
    #
    metric = evaluate.load(path=metric_path)

    #
    # Single processing
    #
    assert len(batch_bbox) == len(batch_mask)
    batch_size = len(batch_bbox)
    for i in range(batch_size):
        metric.add(bbox=batch_bbox[i], mask=batch_mask[i])

    scores = metric.compute()
    assert scores is not None
    for k in expected_scores.keys():
        score = sum(scores[k])
        expected_score = expected_scores[k]
        assert math.isclose(score, expected_score, rel_tol=1e-5)


@pytest.mark.parametrize(
    argnames="arr_func",
    argvalues=(np.array, torch.Tensor),
)
@pytest.mark.parametrize(
    argnames="num",
    argvalues=(1, 2, 3),
)
def test_array_variant(
    metric_path: str, test_fixture_dir: pathlib.Path, num: int, arr_func
):
    batch_bbox = np.load(test_fixture_dir / f"bbox{num}.npy")
    batch_mask = np.load(test_fixture_dir / f"mask{num}.npy")

    metric = evaluate.load(path=metric_path)
    metric.add_batch(
        batch_bbox=arr_func(batch_bbox),
        batch_mask=arr_func(batch_mask),
    )
