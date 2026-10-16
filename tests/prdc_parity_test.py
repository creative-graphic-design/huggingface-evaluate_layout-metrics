import pathlib

import evaluate
import numpy as np
import pytest
from prdc import compute_prdc


PRDC_KEYS = ("precision", "recall", "density", "coverage")


def _metric_path(metrics_dir: pathlib.Path) -> str:
    return str(
        metrics_dir
        / "layout_generative_model_scores"
        / "layout-generative-model-scores.py"
    )


@pytest.fixture(params=("fixture", "seeded"))
def feature_pair(request: pytest.FixtureRequest, test_fixture_dir: pathlib.Path):
    if request.param == "fixture":
        return (
            np.load(test_fixture_dir / "feats_real.npy")[:64].astype(np.float64),
            np.load(test_fixture_dir / "feats_fake.npy")[:64].astype(np.float64),
        )

    generator = np.random.default_rng(264)
    return (
        generator.normal(size=(32, 16)),
        generator.normal(size=(32, 16)),
    )


@pytest.mark.parametrize(
    "nearest_k",
    (None, 3, 10),
    ids=("default", "nearest_k_3", "nearest_k_10"),
)
def test_prdc_scores_match_pinned_library(
    metrics_dir: pathlib.Path,
    feature_pair,
    nearest_k: int | None,
):
    feats_real, feats_fake = feature_pair
    metric = evaluate.load(path=_metric_path(metrics_dir))
    kwargs = {"feats_real": feats_real, "feats_fake": feats_fake}
    expected_nearest_k = 5 if nearest_k is None else nearest_k
    if nearest_k is not None:
        kwargs["nearest_k"] = nearest_k

    actual = metric.compute(**kwargs)
    expected = compute_prdc(
        real_features=feats_real,
        fake_features=feats_fake,
        nearest_k=expected_nearest_k,
    )

    for key in PRDC_KEYS:
        assert actual[key] == expected[key]
