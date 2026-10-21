import os
import pathlib
from typing import List

import evaluate
import pytest
import torch
from PIL import Image, ImageDraw


@pytest.fixture
def base_dir(metrics_dir: pathlib.Path) -> pathlib.Path:
    return metrics_dir / "layout_unreadability"


@pytest.fixture
def metric_path(base_dir: pathlib.Path) -> str:
    return os.path.join(base_dir, "layout-unreadability.py")


@pytest.fixture
def expected_score(is_CI: bool) -> float:
    if not is_CI:
        pytest.skip(
            "The complete PosterLayout canvas dataset is not included in fixtures."
        )
    # Recomputed after using signed float32 Sobel gradients.
    return 0.025522707038422725


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
    assert score is not None and score == pytest.approx(expected_score, abs=1e-12)


def test_metric_uses_signed_sobel_gradients_before_squaring(
    metric_path: str, tmp_path: pathlib.Path
):
    canvas = Image.new("L", (20, 10), 0)
    ImageDraw.Draw(canvas).rectangle((6, 0, 13, 9), fill=255)
    image_path = tmp_path / "vertical-stripe.png"
    canvas.save(image_path)

    metric = evaluate.load(path=metric_path, canvas_width=20, canvas_height=10)
    result = metric.compute(
        predictions=[[[0, 0, 1, 1]]],
        gold_labels=[[[1]]],
        image_canvases=[[str(image_path)]],
    )

    # The two vertical edges contribute four 255-valued columns across the
    # 20-by-10 text region: (4 * 10) / (20 * 10) = 0.2.
    assert result == pytest.approx(0.2, abs=1e-12)
