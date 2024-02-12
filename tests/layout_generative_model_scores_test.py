import math
import os
import pathlib
import pickle
from typing import Dict

import evaluate
import numpy as np
import pytest
import torch


@pytest.fixture
def base_dir() -> str:
    return "layout_generative_model_scores"


@pytest.fixture
def metric_path(base_dir: str) -> str:
    return os.path.join(base_dir, "layout_generative_model_scores.py")


@pytest.fixture
def test_fixture_dir() -> pathlib.Path:
    return pathlib.Path(__file__).parents[1] / "test_fixtures"


@pytest.fixture
def feats_real(test_fixture_dir: pathlib.Path):
    return np.load(test_fixture_dir / "feats_real.npy")


@pytest.fixture
def feats_fake(test_fixture_dir: pathlib.Path):
    return np.load(test_fixture_dir / "feats_fake.npy")


@pytest.fixture
def expected_scores() -> Dict[str, float]:
    return {
        "precision": 0.7633949739212897,
        "recall": 0.9025604551920341,
        "density": 0.6361308677098152,
        "coverage": 0.7889995258416311,
        "fid": 3.553096292242742,
    }


def test_metric(
    metric_path: str,
    feats_real,
    feats_fake,
    expected_scores: Dict[str, float],
):
    metric = evaluate.load(path=metric_path)
    metric.add(feats_real=feats_real, feats_fake=feats_fake)

    scores = metric.compute()
    assert scores is not None

    for k in expected_scores.keys():
        score = scores[k]
        expected_score = expected_scores[k]
        assert math.isclose(score, expected_score, rel_tol=1e-5)
