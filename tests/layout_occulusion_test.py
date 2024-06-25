import os
import pathlib

import evaluate
import pytest
import torch


@pytest.fixture
def base_dir() -> str:
    return "layout_occlusion"


@pytest.fixture
def metric_path(base_dir: str) -> str:
    return os.path.join(base_dir, "layout-occlusion.py")


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
def saliency_maps_1_dir(test_fixture_dir: pathlib.Path):
    return test_fixture_dir / "PKU_PosterLayout" / "test" / "saliencymaps_pfpn"


@pytest.fixture
def saliency_maps_2_dir(test_fixture_dir: pathlib.Path):
    return test_fixture_dir / "PKU_PosterLayout" / "test" / "saliencymaps_basnet"


def test_metric(
    metric_path: str,
    test_fixture_dir: pathlib.Path,
    poster_width: int,
    poster_height: int,
    saliency_maps_1_dir: pathlib.Path,
    saliency_maps_2_dir: pathlib.Path,
    # https://github.com/PKU-ICST-MIPL/PosterLayout-CVPR2023/blob/main/output/results.txt#L8
    expected_score: float = 0.20880194364379892,
):
    image_names = torch.load(test_fixture_dir / "poster_layout_test_order.pt")

    saliency_map_filepaths_1 = [
        saliency_maps_1_dir / name.replace(".", "_pred.") for name in image_names
    ]
    saliency_map_filepaths_2 = [saliency_maps_2_dir / name for name in image_names]
    assert len(saliency_map_filepaths_1) == len(saliency_map_filepaths_2)

    # Convert pathlib.Path to str
    saliency_map_filepaths_1 = [[str(path)] for path in saliency_map_filepaths_1]
    saliency_map_filepaths_2 = [[str(path)] for path in saliency_map_filepaths_2]

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
        saliency_maps_1=saliency_map_filepaths_1,
        saliency_maps_2=saliency_map_filepaths_2,
    )
    score = metric.compute()
    assert score is not None and score == pytest.approx(expected_score, 1e-5)
