import os

import evaluate
import pytest
import torch


@pytest.fixture
def base_dir() -> str:
    return "layout_underlay_effectiveness"


@pytest.fixture
def metric_path(base_dir: str) -> str:
    return os.path.join(base_dir, "layout-underlay-effectiveness.py")


@pytest.fixture
def expected_und_l_score(is_CI: bool) -> float:
    return 0.8025722887508048 if is_CI else 0.8314594966165644


@pytest.fixture
def expected_und_s_score(is_CI: bool) -> float:
    return 0.21428571428571427 if is_CI else 0.43197278911564624


def test_metric(
    metric_path: str,
    poster_predictions: torch.Tensor,
    poster_gold_labels: torch.Tensor,
    poster_width: int,
    poster_height: int,
    # https://github.com/PKU-ICST-MIPL/PosterLayout-CVPR2023/blob/main/output/results.txt#L5-L6
    expected_und_l_score: float,
    expected_und_s_score: float,
):
    metric = evaluate.load(
        path=metric_path,
        canvas_width=poster_width,
        canvas_height=poster_height,
    )
    metric.add_batch(
        predictions=poster_predictions,
        gold_labels=poster_gold_labels,
    )
    score = metric.compute()
    assert score is not None

    assert score["und_l"] == pytest.approx(expected_und_l_score, 1e-5)
    assert score["und_s"] == pytest.approx(expected_und_s_score, 1e-5)
