import io
import os
import pathlib
from typing import List

import evaluate
import numpy as np
import pytest
import torch
from PIL import Image


@pytest.fixture
def base_dir(metrics_dir: pathlib.Path) -> pathlib.Path:
    return metrics_dir / "layout_occlusion"


@pytest.fixture
def metric_path(base_dir: pathlib.Path) -> str:
    return os.path.join(base_dir, "layout-occlusion.py")


@pytest.fixture
def expected_score(is_CI: bool) -> float:
    # https://github.com/PKU-ICST-MIPL/PosterLayout-CVPR2023/blob/main/output/results.txt#L8
    return 0.15746160746433283 if is_CI else 0.20880194364379892


def create_in_memory_saliency_maps(
    batch_size: int, poster_width: int, poster_height: int
):
    def _create_random_gaussian_image(w, h):
        """Create a random black image with a white Gaussian."""
        # Generate random parameters for Gaussian
        x, y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))

        # Generate random center for Gaussian
        mu_x = np.random.rand() - 0.5
        mu_y = np.random.rand() - 0.5
        d = np.sqrt((x - mu_x) ** 2 + (y - mu_y) ** 2)

        # Generate random sigma for Gaussian
        sigma = 0.2 + np.random.rand() * 0.4

        g = np.exp(-((d) ** 2 / (2.0 * sigma**2)))

        # Create a new image with black background
        image = Image.fromarray(g * 255).convert("L")

        return image

    def _create_in_memory_saliency_maps(batch_size: int):
        images = [
            _create_random_gaussian_image(w=poster_width, h=poster_height)
            for _ in range(batch_size)
        ]

        image_filepaths = []
        for image in images:
            image_io = io.BytesIO()
            image.save(image_io, format="PNG")
            image_io.seek(0)
            image_filepaths.append(image_io)
        return image_filepaths

    return _create_in_memory_saliency_maps(batch_size)


def test_metric_random(
    metric_path: str,
    batch_size: int,
    poster_width: int,
    poster_height: int,
    max_layout_elements: int,
    num_coordinates: int,
    num_class_labels: int,
):
    metric = evaluate.load(
        path=metric_path,
        canvas_width=poster_width,
        canvas_height=poster_height,
    )
    batch_predictions = np.random.rand(
        batch_size,
        max_layout_elements,
        num_coordinates,
    )
    batch_gold_labels = np.random.randint(
        num_class_labels,
        size=(
            batch_size,
            max_layout_elements,
            1,
        ),
    )
    metric.add_batch(
        predictions=batch_predictions,
        gold_labels=batch_gold_labels,
        saliency_maps_1=create_in_memory_saliency_maps(
            batch_size=batch_size,
            poster_width=poster_width,
            poster_height=poster_height,
        ),
        saliency_maps_2=create_in_memory_saliency_maps(
            batch_size=batch_size,
            poster_width=poster_width,
            poster_height=poster_height,
        ),
    )
    score = metric.compute()
    assert score is not None


def test_metric(
    metric_path: str,
    poster_predictions: torch.Tensor,
    poster_gold_labels: torch.Tensor,
    poster_width: int,
    poster_height: int,
    saliency_map_filepaths_1: List[pathlib.Path],
    saliency_map_filepaths_2: List[pathlib.Path],
    expected_score: float,
):
    assert len(saliency_map_filepaths_1) == len(saliency_map_filepaths_2)

    # Convert pathlib.Path to str
    saliency_map_filepaths_1 = [[str(path)] for path in saliency_map_filepaths_1]  # type: ignore
    saliency_map_filepaths_2 = [[str(path)] for path in saliency_map_filepaths_2]  # type: ignore

    metric = evaluate.load(
        path=metric_path,
        canvas_width=poster_width,
        canvas_height=poster_height,
    )
    metric.add_batch(
        predictions=poster_predictions,
        gold_labels=poster_gold_labels,
        saliency_maps_1=saliency_map_filepaths_1,
        saliency_maps_2=saliency_map_filepaths_2,
    )
    score = metric.compute()
    assert score is not None and score == pytest.approx(expected_score, 1e-5)
