import math
import pathlib
from typing import Dict

import evaluate
import numpy as np
import pytest
import torch


@pytest.fixture
def metric_path() -> str:
    return "layout_alignment.py"


@pytest.fixture
def test_fixture_dir() -> pathlib.Path:
    return pathlib.Path(__file__).parents[1] / "test_fixtures"


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
    batch_bbox = np.load(test_fixture_dir / f"bbox{num}.npy")
    batch_mask = np.load(test_fixture_dir / f"mask{num}.npy")

    metric = evaluate.load(path=metric_path)
    metric.add_batch(
        batch_bbox=batch_bbox,
        batch_mask=batch_mask,
    )
    scores = metric.compute()
    assert scores is not None

    for k in expected_scores.keys():
        score = sum(scores[k])
        expected_score = expected_scores[k]
        assert math.isclose(score, expected_score, rel_tol=0.001)


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
