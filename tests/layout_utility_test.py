import os
import pathlib
from typing import List

import evaluate
import pytest
import torch


@pytest.fixture
def base_dir() -> str:
    return "layout_utility"


@pytest.fixture
def metric_path(base_dir: str) -> str:
    return os.path.join(base_dir, "layout-utility.py")


@pytest.fixture
def expected_score(is_CI: bool) -> float:
    # https://github.com/PKU-ICST-MIPL/PosterLayout-CVPR2023/blob/main/output/results.txt#L2C14-L2C31
    return 0.24395973228151718 if is_CI else 0.25410159915056757


def test_metric(
    metric_path: str,
    poster_predictions: torch.Tensor,
    poster_gold_labels: torch.Tensor,
    poster_width: int,
    poster_height: int,
    saliency_map_filepaths_1: List[pathlib.Path],
    saliency_map_filepaths_2: List[pathlib.Path],
    expected_score: float,
):
    assert len(saliency_map_filepaths_1) == len(saliency_map_filepaths_2)

    # Convert pathlib.Path to str
    saliency_map_filepaths_1 = [[str(path)] for path in saliency_map_filepaths_1]  # type: ignore
    saliency_map_filepaths_2 = [[str(path)] for path in saliency_map_filepaths_2]  # type: ignore

    metric = evaluate.load(
        path=metric_path,
        canvas_width=poster_width,
        canvas_height=poster_height,
    )
    metric.add_batch(
        predictions=poster_predictions,
        gold_labels=poster_gold_labels,
        saliency_maps_1=saliency_map_filepaths_1,
        saliency_maps_2=saliency_map_filepaths_2,
    )
    score = metric.compute()
    assert score is not None and score == pytest.approx(expected_score, 1e-5)
