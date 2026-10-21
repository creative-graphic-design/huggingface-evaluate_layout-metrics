import os
import pathlib

import evaluate
import pytest
import torch


@pytest.fixture
def base_dir(metrics_dir: pathlib.Path) -> pathlib.Path:
    return metrics_dir / "layout_non_alignment"


@pytest.fixture
def metric_path(base_dir: pathlib.Path) -> str:
    return os.path.join(base_dir, "layout-non-alignment.py")


@pytest.fixture
def expected_score(is_CI: bool) -> float:
    # Recomputed after summing each element's nearest alignment penalty.
    return 0.0196173130229257 if is_CI else 0.025721124329557678


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


def test_metric_sums_each_elements_nearest_alignment_penalty(
    metric_path: str,
):
    metric = evaluate.load(path=metric_path, canvas_width=100, canvas_height=100)
    result = metric.compute(
        predictions=[
            [
                [0.1, 0.1, 0.3, 0.3],
                [0.15, 0.5, 0.35, 0.7],
                [0.5, 0.2, 0.7, 0.4],
            ]
        ],
        gold_labels=[[[1], [1], [1]]],
    )

    # Each element's minimum normalized alignment distance is 0.05, 0.05, 0.1.
    # Applying -log10(1 - distance) and summing gives this hand-computed value.
    assert result == pytest.approx(0.0903102799829795, abs=1e-12)
