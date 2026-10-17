import pathlib

import evaluate
import numpy as np
import pytest
from scipy import linalg


FID_ATOL = 1e-12
SQRTM_IMAGINARY_ATOL = 1e-3


def _metric_path(metrics_dir: pathlib.Path) -> str:
    return str(
        metrics_dir
        / "layout_generative_model_scores"
        / "layout-generative-model-scores.py"
    )


def _independent_fid(feats_real: np.ndarray, feats_fake: np.ndarray) -> float:
    mu_real = np.mean(feats_real, axis=0)
    mu_fake = np.mean(feats_fake, axis=0)
    sigma_real = np.cov(feats_real, rowvar=False)
    sigma_fake = np.cov(feats_fake, rowvar=False)
    covariance_mean, _ = linalg.sqrtm(sigma_real.dot(sigma_fake), disp=False)
    if np.iscomplexobj(covariance_mean):
        np.testing.assert_allclose(
            np.diagonal(covariance_mean).imag, 0.0, atol=SQRTM_IMAGINARY_ATOL
        )
        covariance_mean = covariance_mean.real
    difference = mu_real - mu_fake
    return float(
        difference.dot(difference)
        + np.trace(sigma_real)
        + np.trace(sigma_fake)
        - 2 * np.trace(covariance_mean)
    )


def _assert_fid_parity(actual: dict[str, float], expected: float) -> None:
    np.testing.assert_allclose(actual["fid"], expected, rtol=0.0, atol=FID_ATOL)


@pytest.mark.parametrize("n_points, feature_dim", ((24, 2), (29, 5), (32, 16)))
def test_fid_matches_independent_definition_on_seeded_gaussians(
    metrics_dir: pathlib.Path, n_points: int, feature_dim: int
):
    generator = np.random.default_rng(264 + n_points + feature_dim)
    feats_real = generator.normal(size=(n_points, feature_dim))
    feats_fake = generator.normal(loc=0.5, size=(n_points, feature_dim))
    metric = evaluate.load(path=_metric_path(metrics_dir))
    actual = metric.compute(feats_real=feats_real, feats_fake=feats_fake)
    expected = _independent_fid(feats_real, feats_fake)
    _assert_fid_parity(actual, expected)


def test_fid_matches_independent_definition_on_feature_fixture(
    metrics_dir: pathlib.Path, test_fixture_dir: pathlib.Path
):
    feats_real = np.load(test_fixture_dir / "feats_real.npy")[:64].astype(np.float64)
    feats_fake = np.load(test_fixture_dir / "feats_fake.npy")[:64].astype(np.float64)
    metric = evaluate.load(path=_metric_path(metrics_dir))
    actual = metric.compute(feats_real=feats_real, feats_fake=feats_fake)
    expected = _independent_fid(feats_real, feats_fake)
    _assert_fid_parity(actual, expected)
