import pathlib

import evaluate
import numpy as np
import pytest


PRDC_KEYS = ("precision", "recall", "density", "coverage")


def _metric_path(metrics_dir: pathlib.Path) -> str:
    return str(
        metrics_dir
        / "layout_generative_model_scores"
        / "layout-generative-model-scores.py"
    )


def _pairwise_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    differences = left[:, None, :] - right[None, :, :]
    return np.sqrt(np.sum(differences * differences, axis=-1))


def _nearest_neighbour_radii(features: np.ndarray, nearest_k: int) -> np.ndarray:
    distances = _pairwise_distances(features, features)
    # PRDC includes each point's zero self-distance, so its kth neighbour is
    # the zero-based nearest_k entry in the sorted distances.
    return np.partition(distances, nearest_k, axis=1)[:, nearest_k]


def _independent_prdc(
    feats_real: np.ndarray, feats_fake: np.ndarray, nearest_k: int
) -> dict[str, float]:
    real_radii = _nearest_neighbour_radii(feats_real, nearest_k)
    fake_radii = _nearest_neighbour_radii(feats_fake, nearest_k)
    distances = _pairwise_distances(feats_real, feats_fake)
    real_neighbourhoods = distances < real_radii[:, None]
    fake_neighbourhoods = distances < fake_radii[None, :]
    return {
        "precision": real_neighbourhoods.any(axis=0).mean(),
        "recall": fake_neighbourhoods.any(axis=1).mean(),
        "density": real_neighbourhoods.sum(axis=0).mean() / nearest_k,
        "coverage": (distances.min(axis=1) < real_radii).mean(),
    }


def _assert_definition_parity(actual: dict[str, float], expected: dict[str, float]):
    for key in PRDC_KEYS:
        np.testing.assert_allclose(actual[key], expected[key], rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("n_points, feature_dim", ((12, 2), (19, 3), (29, 5)))
@pytest.mark.parametrize("nearest_k", (1, 3, 5))
def test_prdc_scores_match_independent_definitions(
    metrics_dir: pathlib.Path,
    n_points: int,
    feature_dim: int,
    nearest_k: int,
):
    generator = np.random.default_rng(264 + n_points + feature_dim)
    feats_real = generator.normal(size=(n_points, feature_dim))
    feats_fake = generator.normal(loc=0.5, size=(n_points, feature_dim))
    metric = evaluate.load(path=_metric_path(metrics_dir))
    actual = metric.compute(
        feats_real=feats_real, feats_fake=feats_fake, nearest_k=nearest_k
    )
    expected = _independent_prdc(feats_real, feats_fake, nearest_k)
    _assert_definition_parity(actual, expected)


def test_prdc_scores_match_independent_definitions_fixture(
    metrics_dir: pathlib.Path, test_fixture_dir: pathlib.Path
):
    feats_real = np.load(test_fixture_dir / "feats_real.npy")[:64].astype(np.float64)
    feats_fake = np.load(test_fixture_dir / "feats_fake.npy")[:64].astype(np.float64)
    actual = evaluate.load(path=_metric_path(metrics_dir)).compute(
        feats_real=feats_real, feats_fake=feats_fake
    )
    expected = _independent_prdc(feats_real, feats_fake, nearest_k=5)
    _assert_definition_parity(actual, expected)
