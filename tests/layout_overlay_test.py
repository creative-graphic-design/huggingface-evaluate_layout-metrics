import os
import pathlib

import evaluate
import pytest
import torch


@pytest.fixture
def base_dir(metrics_dir: pathlib.Path) -> pathlib.Path:
    return metrics_dir / "layout_overlay"


@pytest.fixture
def metric_path(base_dir: pathlib.Path) -> str:
    return os.path.join(base_dir, "layout-overlay.py")


@pytest.fixture
def expected_score(is_CI: bool) -> float:
    # https://github.com/PKU-ICST-MIPL/PosterLayout-CVPR2023/blob/main/output/results.txt#L3C14-L3C34
    return 0.020658400935720658 if is_CI else 0.022033710565543454


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
    assert score is not None and score == pytest.approx(expected_score, 1e-5)
