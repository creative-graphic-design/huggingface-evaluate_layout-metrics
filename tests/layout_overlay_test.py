import os
import pathlib

import evaluate
import pytest
import torch


@pytest.fixture
def base_dir() -> str:
    return "layout_overlay"


@pytest.fixture
def metric_path(base_dir: str) -> str:
    return os.path.join(base_dir, "layout-overlay.py")


@pytest.fixture
def test_fixture_dir() -> pathlib.Path:
    return pathlib.Path(__file__).parents[1] / "test_fixtures"


@pytest.fixture
def poster_width() -> int:
    return 513


@pytest.fixture
def poster_height() -> int:
    return 750


def test_metric(
    metric_path: str,
    test_fixture_dir: pathlib.Path,
    poster_width: int,
    poster_height: int,
    # https://github.com/PKU-ICST-MIPL/PosterLayout-CVPR2023/blob/main/output/results.txt#L3C14-L3C34
    expected_score: float = 0.022033710565543454,
):
    # shape: (batch_size, max_elements, 4)
    predictions = torch.load(test_fixture_dir / "poster_layout_boxes.pt")
    # shape: (batch_size, max_elements, 1)
    gold_labels = torch.load(test_fixture_dir / "poster_layout_clses.pt")

    metric = evaluate.load(
        path=metric_path,
        canvas_width=poster_width,
        canvas_height=poster_height,
    )
    metric.add_batch(predictions=predictions, gold_labels=gold_labels)
    score = metric.compute()
    assert score is not None and score == pytest.approx(expected_score, 1e-5)
