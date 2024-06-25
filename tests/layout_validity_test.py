import os

import evaluate
import pytest
import torch


@pytest.fixture
def base_dir() -> str:
    return "layout_validity"


@pytest.fixture
def metric_path(base_dir: str) -> str:
    return os.path.join(base_dir, "layout-validity.py")


@pytest.fixture
def expected_score(is_CI: bool) -> float:
    # https://github.com/PKU-ICST-MIPL/PosterLayout-CVPR2023/blob/main/output/results.txt#L2C14-L2C31
    return 0.8478260869565217 if is_CI else 0.878844169246646


def test_metric(
    metric_path: str,
    poster_predictions: torch.Tensor,
    poster_gold_labels: torch.Tensor,
    poster_width: int,
    poster_height: int,
    expected_score: float,
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
    assert score is not None and score == expected_score
