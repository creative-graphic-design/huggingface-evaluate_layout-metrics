import os
import pathlib
from typing import List

import evaluate
import pytest
import torch


@pytest.fixture
def base_dir(metrics_dir: pathlib.Path) -> pathlib.Path:
    return metrics_dir / "layout_unreadability"


@pytest.fixture
def metric_path(base_dir: pathlib.Path) -> str:
    return os.path.join(base_dir, "layout-unreadability.py")


@pytest.fixture
def expected_score(is_CI: bool) -> float:
    # https://github.com/PKU-ICST-MIPL/PosterLayout-CVPR2023/blob/main/output/results.txt#L9C14-L9C33
    return 0.2119157190596277 if is_CI else 0.18741514749764512


def test_metric(
    metric_path: str,
    poster_predictions: torch.Tensor,
    poster_gold_labels: torch.Tensor,
    poster_width: int,
    poster_height: int,
    poster_image_canvas_filepaths: List[pathlib.Path],
    expected_score: float,
):
    poster_image_canvas_filepaths = [
        [str(path)]  # type: ignore
        for path in poster_image_canvas_filepaths
    ]

    metric = evaluate.load(
        path=metric_path,
        canvas_width=poster_width,
        canvas_height=poster_height,
    )
    metric.add_batch(
        predictions=poster_predictions,
        gold_labels=poster_gold_labels,
        image_canvases=poster_image_canvas_filepaths,
    )
    score = metric.compute()
    assert score is not None and score == expected_score
