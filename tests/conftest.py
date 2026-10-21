import importlib.machinery
import importlib.util
import os
import pathlib
import sys
from types import ModuleType
from typing import List

import pytest
import torch


def load_vendor_package(
    name: str, package_path: pathlib.Path, *, execute_init: bool = True
) -> ModuleType:
    """Load one package from a vendored source tree without exposing its parent tree."""
    if not package_path.is_dir():
        raise RuntimeError(
            f"vendored package '{name}' is missing; initialize its submodule with "
            "'git submodule update --init --recursive'"
        )

    init_path = package_path / "__init__.py"
    if execute_init and init_path.is_file():
        spec = importlib.util.spec_from_file_location(
            name, init_path, submodule_search_locations=[str(package_path)]
        )
    else:
        spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        spec.submodule_search_locations = [str(package_path)]
    if spec is None:
        raise RuntimeError(f"could not load the vendored package '{name}'")

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    if spec.loader is not None:
        spec.loader.exec_module(module)
    return module


_VENDOR_DIR = pathlib.Path(__file__).parents[1] / "vendor"
load_vendor_package("layout_eval", _VENDOR_DIR / "ltsim" / "src" / "layout_eval")
load_vendor_package(
    "trainer", _VENDOR_DIR / "layout-dm" / "src" / "trainer" / "trainer"
)
_CONST_LAYOUT_DIR = _VENDOR_DIR / "const-layout"
_CONST_LAYOUT_MODULES_BEFORE = set(sys.modules)
load_vendor_package("const_layout", _CONST_LAYOUT_DIR)
# const_layout.metric uses top-level legacy imports; avoid its unrelated data registry.
load_vendor_package("data", _CONST_LAYOUT_DIR / "data", execute_init=False)
sys.path.insert(0, str(_CONST_LAYOUT_DIR))
try:
    from const_layout.metric import (
        __compute_maximum_iou as const_layout_compute_maximum_iou_for_groups,
        __get_cond2layouts as const_layout_get_cond2layouts,
        compute_alignment as const_layout_compute_alignment,
        compute_overlap as const_layout_compute_overlap,
    )
finally:
    sys.path.remove(str(_CONST_LAYOUT_DIR))
    for module_name in set(sys.modules) - _CONST_LAYOUT_MODULES_BEFORE:
        if module_name.split(".", 1)[0] in {"data", "model", "util"}:
            sys.modules.pop(module_name, None)

load_vendor_package(
    "poster_layout",
    _VENDOR_DIR / "pku-poster-layout",
    execute_init=False,
)
poster_layout_eval = importlib.import_module("poster_layout.eval")

__all__ = [
    "const_layout_compute_alignment",
    "const_layout_compute_maximum_iou_for_groups",
    "const_layout_compute_overlap",
    "const_layout_get_cond2layouts",
    "poster_layout_eval",
]


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
    image_names = torch.load(
        test_fixture_dir / "poster_layout_test_order.pt", weights_only=False
    )
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
    predictions = torch.load(
        test_fixture_dir / "poster_layout_boxes.pt", weights_only=False
    )
    return predictions[:num_small_test_samples] if is_CI else predictions


@pytest.fixture
def poster_gold_labels(
    test_fixture_dir: pathlib.Path, is_CI: bool, num_small_test_samples: int
) -> torch.Tensor:
    # shape: (batch_size, max_elements, 1)
    gold_labels = torch.load(
        test_fixture_dir / "poster_layout_clses.pt", weights_only=False
    )
    return gold_labels[:num_small_test_samples] if is_CI else gold_labels
