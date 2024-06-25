import os
import pathlib

import evaluate
import pytest
import torch


@pytest.fixture
def base_dir() -> str:
    return "layout_unreadability"


@pytest.fixture
def metric_path(base_dir: str) -> str:
    return os.path.join(base_dir, "layout-unreadability.py")


@pytest.fixture
def test_fixture_dir() -> pathlib.Path:
    return pathlib.Path(__file__).parents[1] / "test_fixtures"


@pytest.fixture
def poster_width() -> int:
    return 513


@pytest.fixture
def poster_height() -> int:
    return 750


@pytest.fixture
def image_canvas_dir(test_fixture_dir: pathlib.Path):
    return test_fixture_dir / "PKU_PosterLayout" / "test" / "image_canvas"


def test_metric(
    metric_path: str,
    test_fixture_dir: pathlib.Path,
    poster_width: int,
    poster_height: int,
    image_canvas_dir: pathlib.Path,
    # https://github.com/PKU-ICST-MIPL/PosterLayout-CVPR2023/blob/main/output/results.txt#L9C14-L9C33
    expected_score: float = 0.18741514749764512,
):
    image_names = torch.load(test_fixture_dir / "poster_layout_test_order.pt")

    image_canvas_filepaths = [image_canvas_dir / name for name in image_names]
    image_canvas_filepaths = [[str(path)] for path in image_canvas_filepaths]

    # shape: (batch_size, max_elements, 4)
    predictions = torch.load(test_fixture_dir / "poster_layout_boxes.pt")
    # shape: (batch_size, max_elements, 1)
    gold_labels = torch.load(test_fixture_dir / "poster_layout_clses.pt")

    metric = evaluate.load(
        path=metric_path,
        canvas_width=poster_width,
        canvas_height=poster_height,
    )
    metric.add_batch(
        predictions=predictions,
        gold_labels=gold_labels,
        image_canvases=image_canvas_filepaths,
    )
    score = metric.compute()
    assert score is not None and score == expected_score
