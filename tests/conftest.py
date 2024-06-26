import os
import pathlib
from typing import List

import pytest
import torch


@pytest.fixture
def metrics_dir() -> pathlib.Path:
    return pathlib.Path(__file__).parents[1] / "metrics"


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
def batch_size() -> int:
    return 512


@pytest.fixture
def max_layout_elements() -> int:
    return 25


@pytest.fixture
def num_coordinates() -> int:
    return 4


@pytest.fixture
def num_class_labels() -> int:
    return 10


@pytest.fixture
def is_CI() -> bool:
    return bool(os.environ.get("CI", False))


@pytest.fixture
def poster_layout_dir_name(is_CI: bool) -> str:
    return "PKU_PosterLayout_small" if is_CI else "PKU_PosterLayout_all"


@pytest.fixture
def poster_layout_saliency_maps_1_dir(
    test_fixture_dir: pathlib.Path, poster_layout_dir_name: str
) -> pathlib.Path:
    return test_fixture_dir / poster_layout_dir_name / "test" / "saliencymaps_pfpn"


@pytest.fixture
def poster_layout_saliency_maps_2_dir(
    test_fixture_dir: pathlib.Path, poster_layout_dir_name: str
) -> pathlib.Path:
    return test_fixture_dir / poster_layout_dir_name / "test" / "saliencymaps_basnet"


@pytest.fixture
def poster_layout_image_canvas_dir(
    test_fixture_dir: pathlib.Path, poster_layout_dir_name: str
) -> pathlib.Path:
    return test_fixture_dir / poster_layout_dir_name / "test" / "image_canvas"


@pytest.fixture
def num_small_test_samples() -> int:
    return 10


@pytest.fixture
def poster_image_names(
    test_fixture_dir: pathlib.Path, is_CI: bool, num_small_test_samples: int
) -> List[str]:
    image_names = torch.load(test_fixture_dir / "poster_layout_test_order.pt")
    return image_names[:num_small_test_samples] if is_CI else image_names


@pytest.fixture
def saliency_map_filepaths_1(
    poster_layout_saliency_maps_1_dir: pathlib.Path,
    poster_image_names: List[str],
) -> List[pathlib.Path]:
    return [
        poster_layout_saliency_maps_1_dir / name.replace(".", "_pred.")
        for name in poster_image_names
    ]


@pytest.fixture
def saliency_map_filepaths_2(
    poster_layout_saliency_maps_2_dir: pathlib.Path, poster_image_names: List[str]
) -> List[pathlib.Path]:
    return [poster_layout_saliency_maps_2_dir / name for name in poster_image_names]


@pytest.fixture
def poster_image_canvas_filepaths(
    poster_layout_image_canvas_dir: pathlib.Path, poster_image_names: List[str]
) -> List[pathlib.Path]:
    return [poster_layout_image_canvas_dir / name for name in poster_image_names]


@pytest.fixture
def poster_predictions(
    test_fixture_dir: pathlib.Path, is_CI: bool, num_small_test_samples: int
) -> torch.Tensor:
    # shape: (batch_size, max_elements, 4)
    predictions = torch.load(test_fixture_dir / "poster_layout_boxes.pt")
    return predictions[:num_small_test_samples] if is_CI else predictions


@pytest.fixture
def poster_gold_labels(
    test_fixture_dir: pathlib.Path, is_CI: bool, num_small_test_samples: int
) -> torch.Tensor:
    # shape: (batch_size, max_elements, 1)
    gold_labels = torch.load(test_fixture_dir / "poster_layout_clses.pt")
    return gold_labels[:num_small_test_samples] if is_CI else gold_labels
