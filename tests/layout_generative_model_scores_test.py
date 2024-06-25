import math
import os
import pathlib
from typing import Dict

import evaluate
import numpy as np
import pytest


@pytest.fixture
def base_dir(metrics_dir: pathlib.Path) -> pathlib.Path:
    return metrics_dir / "layout_generative_model_scores"


@pytest.fixture
def metric_path(base_dir: pathlib.Path) -> str:
    return os.path.join(base_dir, "layout-generative-model-scores.py")


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


def test_metric_random(metric_path: str):
    metric = evaluate.load(path=metric_path)

    feats_real = np.random.rand(512, 256)
    feats_fake = np.random.rand(512, 256)
    metric.add_batch(feats_real=feats_real, feats_fake=feats_fake)

    scores = metric.compute()
    assert scores is not None


def test_metric(
    metric_path: str,
    feats_real,
    feats_fake,
    expected_scores: Dict[str, float],
):
    #
    # Load generative model scores
    #
    metric = evaluate.load(path=metric_path)

    #
    # Batch proceessing
    #
    metric.add_batch(feats_real=feats_real, feats_fake=feats_fake)

    scores = metric.compute()
    assert scores is not None

    for k in expected_scores.keys():
        score = scores[k]
        expected_score = expected_scores[k]
        assert math.isclose(score, expected_score, rel_tol=1e-5)

    #
    # Reload the metric
    #
    metric = evaluate.load(path=metric_path)

    #
    # Single proceessing
    #
    assert len(feats_real) == len(feats_fake)
    num_feats = len(feats_real)
    for i in range(num_feats):
        metric.add(feats_real=feats_real[i], feats_fake=feats_fake[i])

    scores = metric.compute()
    assert scores is not None

    for k in expected_scores.keys():
        score = scores[k]
        expected_score = expected_scores[k]
        assert math.isclose(score, expected_score, rel_tol=1e-5)
