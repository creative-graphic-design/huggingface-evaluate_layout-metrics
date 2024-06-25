import math
import os
import pathlib
import pickle

import evaluate
import numpy as np
import pytest
import torch


@pytest.fixture
def base_dir(metrics_dir: pathlib.Path) -> pathlib.Path:
    return metrics_dir / "layout_maximum_iou"


@pytest.fixture
def metric_path(base_dir: pathlib.Path) -> str:
    return os.path.join(base_dir, "layout-maximum-iou.py")


@pytest.fixture
def layouts1(test_fixture_dir: pathlib.Path):
    layouts1_path = test_fixture_dir / "layouts_main.pkl"
    with layouts1_path.open("rb") as rf:
        layouts1 = pickle.load(rf)
    return layouts1


@pytest.fixture
def layouts2(test_fixture_dir: pathlib.Path):
    layouts2_path = test_fixture_dir / "layouts_sub.pkl"
    with layouts2_path.open("rb") as rf:
        layouts2 = pickle.load(rf)
    return layouts2


def test_metric_random(
    metric_path: str, num_samples: int = 16, num_categories: int = 24
):
    metric = evaluate.load(path=metric_path)

    layout1 = {
        "bboxes": np.random.rand(num_samples, 4),
        "categories": np.random.randint(0, num_categories, size=(num_samples,)),
    }
    layout2 = {
        "bboxes": np.random.rand(num_samples, 4),
        "categories": np.random.randint(0, num_categories, size=(num_samples,)),
    }
    metric.add(layouts1=layout1, layouts2=layout2)

    scores = metric.compute()
    assert scores is not None


def test_metric(
    metric_path: str,
    layouts1,
    layouts2,
    expected_score: float = 0.2770548066027329,
):
    #
    # Load Max. IoU metric
    #
    metric = evaluate.load(path=metric_path)

    #
    # Batch processing
    #
    metric.add_batch(layouts1=layouts1, layouts2=layouts2)

    score = metric.compute()
    assert score is not None and isinstance(score, float)
    assert math.isclose(score, expected_score, rel_tol=1e-5)

    #
    # Reload the metric
    #
    metric = evaluate.load(path=metric_path)

    #
    # Single processing
    #
    assert len(layouts1) == len(layouts2)
    for i in range(len(layouts1)):
        metric.add(layouts1=layouts1[i], layouts2=layouts2[i])

    score = metric.compute()
    assert score is not None and isinstance(score, float)
    assert math.isclose(score, expected_score, rel_tol=1e-5)


@pytest.mark.parametrize(
    argnames="arr_func",
    argvalues=(np.array, torch.Tensor),
)
def test_array_variant(metric_path: str, layouts1, layouts2, arr_func):
    metric = evaluate.load(path=metric_path)

    metric.add_batch(
        layouts1=[{k: arr_func(v) for k, v in layout.items()} for layout in layouts1],
        layouts2=[{k: arr_func(v) for k, v in layout.items()} for layout in layouts2],
    )
