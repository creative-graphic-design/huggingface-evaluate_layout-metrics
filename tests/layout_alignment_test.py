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
    return "layout_alignment"


@pytest.fixture
def metric_path(base_dir: str) -> str:
    return os.path.join(base_dir, "layout_alignment.py")


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
                "alignment-ACLayoutGAN": 5.670039176940918,
                "alignment-LayoutGAN++": 1.373178482055664,
                "alignment-NDN": 41.2618408203125,
            },
        ),
        (
            2,
            {
                "alignment-ACLayoutGAN": 5.950614929199219,
                "alignment-LayoutGAN++": 1.5703489780426025,
                "alignment-NDN": 43.25517272949219,
            },
        ),
        (
            3,
            {
                "alignment-ACLayoutGAN": 4.884631633758545,
                "alignment-LayoutGAN++": 0.7807250022888184,
                "alignment-NDN": 41.5506477355957,
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
    batch_bbox = np.load(test_fixture_dir / f"batch_bbox{num}.npy")
    batch_mask = np.load(test_fixture_dir / f"batch_mask{num}.npy")

    metric = evaluate.load(path=metric_path)
    metric.add_batch(
        bbox=arr_func(batch_bbox),
        mask=arr_func(batch_mask),
    )
