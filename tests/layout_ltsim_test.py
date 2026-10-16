import json
import math
import os
import pathlib
import runpy
import warnings

import evaluate
import numpy as np
import pytest
from layout_eval.measures.ltsim import (
    compute_emd_between_layout,
    compute_ltsim_between_layout,
    provide_cost_mat_and_dual_var,
)
from layout_eval.measures.mmd import convert_emd_to_affinity, estimate_mmd


@pytest.fixture
def base_dir(metrics_dir: pathlib.Path) -> pathlib.Path:
    return metrics_dir / "layout_ltsim"


@pytest.fixture
def metric_path(base_dir: pathlib.Path) -> str:
    return os.path.join(base_dir, "layout-ltsim.py")


@pytest.fixture
def layouts(test_fixture_dir: pathlib.Path):
    with (test_fixture_dir / "ltsim_layouts.json").open() as file:
        return json.load(file)


def _layout(item):
    return {
        "bbox": [box for box, valid in zip(item["bboxes"], item["mask"]) if valid],
        "category": [
            label for label, valid in zip(item["labels"], item["mask"]) if valid
        ],
    }


def _batch(items):
    return [
        {
            "bboxes": [
                box for box, valid in zip(item["bboxes"], item["mask"]) if valid
            ],
            "categories": [
                label for label, valid in zip(item["labels"], item["mask"]) if valid
            ],
        }
        for item in items
    ]


def _compute(metric_path, reference, generated):
    metric = evaluate.load(path=metric_path)
    metric.add_batch(layouts1=_batch(reference), layouts2=_batch(generated))
    return metric.compute()


def test_identical_layout_is_maximal(metric_path, layouts):
    score = _compute(metric_path, layouts["reference"][:1], layouts["reference"][:1])
    assert score["ltsim"] == 1.0


def test_element_permutation_is_invariant(metric_path, layouts):
    score = _compute(metric_path, layouts["reference"][1:], layouts["generated"][1:])
    assert score["ltsim"] == 1.0


def test_label_mismatch_penalty(metric_path, layouts):
    reference = [layouts["reference"][0]]
    generated = [{**layouts["generated"][0], "labels": [2]}]
    score = _compute(metric_path, reference, generated)
    assert math.isclose(score["ltsim"], math.exp(-0.5), rel_tol=1e-12)


def test_unequal_element_counts_use_fractional_transport(metric_path):
    reference = [
        {
            "bboxes": [[0.25, 0.25, 0.2, 0.2]],
            "mask": [True],
            "labels": [1],
        }
    ]
    generated = [
        {
            "bboxes": [[0.25, 0.25, 0.2, 0.2], [0.75, 0.75, 0.2, 0.2]],
            "mask": [True, True],
            "labels": [1, 1],
        }
    ]
    score = _compute(metric_path, reference, generated)
    assert math.isclose(score["ltsim"], math.exp(-45 / 196), rel_tol=1e-12)


def test_partially_overlapping_boxes_use_generalized_iou(metric_path, layouts):
    score = _compute(
        metric_path,
        layouts["partial_overlap"]["reference"],
        layouts["partial_overlap"]["generated"],
    )
    assert score["ltsim"] == pytest.approx(math.exp(-99 / 575), rel=1e-12)


def test_empty_layout_behavior_matches_official(metric_path):
    empty = {"bboxes": [], "mask": [], "labels": []}
    nonempty = {
        "bboxes": [[0.25, 0.25, 0.2, 0.2]],
        "mask": [True],
        "labels": [1],
    }
    score = _compute(metric_path, [empty], [nonempty])
    assert score["ltsim"] == pytest.approx(math.exp(-1.0))
    assert (
        compute_emd_between_layout(
            _layout(empty), _layout(empty), provide_cost_mat_and_dual_var
        )
        == 0.0
    )
    assert (
        compute_emd_between_layout(
            _layout(empty), _layout(nonempty), provide_cost_mat_and_dual_var
        )
        == 1.0
    )


def test_pairwise_and_mmd_match_pinned_official(metric_path, layouts):
    reference = [_layout(item) for item in layouts["reference"]]
    generated = [_layout(item) for item in layouts["generated"]]
    expected_pairwise = np.mean(
        [compute_ltsim_between_layout(x, y) for x, y in zip(reference, generated)]
    )
    emd11 = np.asarray(
        [
            [
                compute_emd_between_layout(x, y, provide_cost_mat_and_dual_var)
                for y in reference
            ]
            for x in reference
        ]
    )
    emd22 = np.asarray(
        [
            [
                compute_emd_between_layout(x, y, provide_cost_mat_and_dual_var)
                for y in generated
            ]
            for x in generated
        ]
    )
    emd12 = np.asarray(
        [
            [
                compute_emd_between_layout(x, y, provide_cost_mat_and_dual_var)
                for y in generated
            ]
            for x in reference
        ]
    )
    sigma = np.median(emd11[np.triu_indices(len(emd11), k=1)])
    expected_mmd = estimate_mmd(
        convert_emd_to_affinity(emd11, sigma),
        convert_emd_to_affinity(emd22, sigma),
        convert_emd_to_affinity(emd12, sigma),
    )
    score = _compute(metric_path, layouts["reference"], layouts["generated"])
    assert score["ltsim"] == pytest.approx(expected_pairwise, rel=1e-12)
    assert score["ltsim_mmd"] == pytest.approx(expected_mmd, rel=1e-12)


def test_asymmetric_mmd_uses_reference_sigma(metric_path, layouts):
    module = runpy.run_path(metric_path)
    reference = [_layout(item) for item in layouts["asymmetric_mmd"]["reference"]]
    generated = [_layout(item) for item in layouts["asymmetric_mmd"]["generated"]]
    emd11 = np.asarray(
        [
            [
                compute_emd_between_layout(x, y, provide_cost_mat_and_dual_var)
                for y in reference
            ]
            for x in reference
        ]
    )
    emd22 = np.asarray(
        [
            [
                compute_emd_between_layout(x, y, provide_cost_mat_and_dual_var)
                for y in generated
            ]
            for x in generated
        ]
    )
    emd12 = np.asarray(
        [
            [
                compute_emd_between_layout(x, y, provide_cost_mat_and_dual_var)
                for y in generated
            ]
            for x in reference
        ]
    )
    sigma = np.median(emd11[np.triu_indices(len(emd11), k=1)])
    generated_sigma = np.median(emd22[np.triu_indices(len(emd22), k=1)])
    assert sigma != generated_sigma
    expected = estimate_mmd(
        convert_emd_to_affinity(emd11, sigma),
        convert_emd_to_affinity(emd22, sigma),
        convert_emd_to_affinity(emd12, sigma),
    )
    assert module["_ltsim_mmd"](reference, generated) == pytest.approx(
        expected, rel=1e-12
    )


@pytest.mark.parametrize("case", ["one_layout", "identical_reference"])
def test_undefined_mmd_bandwidth_is_nan_without_warning(metric_path, layouts, case):
    if case == "one_layout":
        reference = layouts["reference"][:1]
        generated = layouts["generated"][:1]
    else:
        reference = layouts["reference"][:1] * 2
        generated = layouts["generated"][:2]
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        score = _compute(metric_path, reference, generated)
    assert math.isnan(score["ltsim_mmd"])
