import math
import os
import pathlib
import pickle

import evaluate
import numpy as np
import pytest
import torch


@pytest.fixture
def base_dir() -> str:
    return "layout_maximum_iou"


@pytest.fixture
def metric_path(base_dir: str) -> str:
    return os.path.join(base_dir, "layout_maximum_iou.py")


@pytest.fixture
def test_fixture_dir() -> pathlib.Path:
    return pathlib.Path(__file__).parents[1] / "test_fixtures"


@pytest.fixture
def layouts1(test_fixture_dir: pathlib.Path):
    layouts1_path = test_fixture_dir / "layouts1.pkl"
    with layouts1_path.open("rb") as rf:
        layouts1 = pickle.load(rf)
    return layouts1


@pytest.fixture
def layouts2(test_fixture_dir: pathlib.Path):
    layouts2_path = test_fixture_dir / "layouts2.pkl"
    with layouts2_path.open("rb") as rf:
        layouts2 = pickle.load(rf)
    return layouts2


def test_metric(
    metric_path: str,
    layouts1,
    layouts2,
    expected_score: float = 0.2770548066027329,
):
    metric = evaluate.load(path=metric_path)
    metric.add(layouts1=layouts1, layouts2=layouts2)

    score = metric.compute()
    assert score is not None and isinstance(score, float)
    assert math.isclose(score, expected_score, rel_tol=1e-5)


@pytest.mark.parametrize(
    argnames="arr_func",
    argvalues=(np.array, torch.Tensor),
)
def test_array_variant(metric_path: str, layouts1, layouts2, arr_func):
    metric = evaluate.load(path=metric_path)

    metric.add(
        layouts1=[{k: arr_func(v) for k, v in layout.items()} for layout in layouts1],
        layouts2=[{k: arr_func(v) for k, v in layout.items()} for layout in layouts2],
    )
