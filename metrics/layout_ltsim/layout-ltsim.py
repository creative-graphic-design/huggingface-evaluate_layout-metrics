"""LTSim and LTSim-MMD.

The implementation follows the pinned ``mayu-ot/ltsim`` tag
``v1.0.0-alpha`` (commit ``544b8bccfce4630591589cdec5c5473392190039``):

* element costs and uniform OT marginals mirror
  ``src/layout_eval/measures/ltsim.py`` lines 8-36;
* equality and empty-layout handling mirror lines 53-65;
* the exponential LTSim score mirrors lines 78-80;
* GIoU mirrors ``src/layout_eval/measures/utils.py`` lines 19-75; and
* the MMD affinity and upper-triangle calculation mirror
  ``src/layout_eval/measures/mmd.py`` lines 47-55. The reference evaluation
  selects sigma as the strict-upper-triangle real-real EMD median in
  ``src/experiments/eval_unconditional.py`` lines 40-43.
* the paper's Dataflow evaluation uses a different empty-layout order in
  ``dataflow/src/comp_pair/ltsim.py`` lines 52-59; this module deliberately
  follows the library variant above.

Line-anchored source links at the pinned revision:

* https://github.com/mayu-ot/ltsim/blob/544b8bccfce4630591589cdec5c5473392190039/src/layout_eval/measures/ltsim.py#L8-L36
* https://github.com/mayu-ot/ltsim/blob/544b8bccfce4630591589cdec5c5473392190039/src/layout_eval/measures/ltsim.py#L53-L81
* https://github.com/mayu-ot/ltsim/blob/544b8bccfce4630591589cdec5c5473392190039/src/layout_eval/measures/utils.py#L19-L75
* https://github.com/mayu-ot/ltsim/blob/544b8bccfce4630591589cdec5c5473392190039/src/layout_eval/measures/mmd.py#L47-L55
* https://github.com/mayu-ot/ltsim/blob/544b8bccfce4630591589cdec5c5473392190039/src/experiments/eval_unconditional.py#L40-L43
* https://github.com/mayu-ot/ltsim/blob/544b8bccfce4630591589cdec5c5473392190039/dataflow/src/comp_pair/ltsim.py#L52-L59

Reference source:
https://github.com/mayu-ot/ltsim/tree/544b8bccfce4630591589cdec5c5473392190039
"""

from typing import Dict, List, TypedDict

import datasets as ds
import evaluate
import numpy as np
import numpy.typing as npt
import ot
from evaluate.utils.file_utils import add_start_docstrings

_DESCRIPTION = """\
Computes pairwise LTSim and collection-level LTSim-MMD for layout generation.
"""


class Layout(TypedDict):
    bboxes: List[List[float]]
    categories: List[int]


class InternalLayout(TypedDict):
    bbox: List[List[float]]
    category: List[int]


_KWARGS_DESCRIPTION = """\
Args:
    layouts1 (`list` of `dict`): First layout collection. Each layout has
        `bboxes` in normalized center-x, center-y, width, height format and
        integer `categories`.
    layouts2 (`list` of `dict`): Second layout collection with the same
        structure as `layouts1`.

Returns:
    dict: Mean pairwise LTSim and collection-level LTSim-MMD.

The two batches contain aligned layout pairs for `ltsim`; the full batches are
also compared as collections for `ltsim_mmd`.
"""

_CITATION = """\
@article{otani2024ltsim,
  title={LTSim: Layout Transportation-based Similarity Measure for Evaluating Layout Generation},
  author={Otani, Mayu and Inoue, Naoto and Kikuchi, Kotaro and Togashi, Ryo},
  journal={arXiv preprint arXiv:2407.12356},
  year={2024}
}
"""


def _convert_xywh_to_ltrb(
    bbox: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], ...]:
    xc, yc, width, height = bbox.T
    return (
        xc - width / 2.0,
        yc - height / 2.0,
        xc + width / 2.0,
        yc + height / 2.0,
    )


def _compute_giou(
    bbox1: npt.NDArray[np.float64], bbox2: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    left1, top1, right1, bottom1 = _convert_xywh_to_ltrb(bbox1)
    left2, top2, right2, bottom2 = _convert_xywh_to_ltrb(bbox2)
    area1 = (right1 - left1) * (bottom1 - top1)
    area2 = (right2 - left2) * (bottom2 - top2)
    giou = np.zeros((len(bbox1), len(bbox2)), dtype=np.float64)

    for i in range(len(bbox1)):
        for j in range(len(bbox2)):
            left = max(left1[i], left2[j])
            right = min(right1[i], right2[j])
            top = max(top1[i], top2[j])
            bottom = min(bottom1[i], bottom2[j])
            intersection = (
                (right - left) * (bottom - top)
                if left < right and top < bottom
                else 0.0
            )
            union = area1[i] + area2[j] - intersection
            iou = intersection / union
            hull_left = min(left1[i], left2[j])
            hull_right = max(right1[i], right2[j])
            hull_top = min(top1[i], top2[j])
            hull_bottom = max(bottom1[i], bottom2[j])
            hull = (hull_right - hull_left) * (hull_bottom - hull_top)
            giou[i, j] = iou - (hull - union) / hull

    return giou


def _layout(layout: Layout) -> InternalLayout:
    boxes = np.asarray(layout["bboxes"], dtype=np.float64)
    if boxes.size == 0:
        boxes = np.empty((0, 4), dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("bboxes must have shape (num_elements, 4)")
    categories = np.asarray(layout["categories"], dtype=np.int64).reshape(-1)
    if len(boxes) != len(categories):
        raise ValueError("bboxes and categories must have the same length")
    return {"bbox": boxes.tolist(), "category": categories.tolist()}


def _cost_matrix(
    layout1: InternalLayout, layout2: InternalLayout
) -> npt.NDArray[np.float64]:
    bbox1 = np.asarray(layout1["bbox"], dtype=np.float64)
    bbox2 = np.asarray(layout2["bbox"], dtype=np.float64)
    giou = _compute_giou(bbox1, bbox2)
    spatial_cost = 1.0 - (giou + 1.0) * 0.5
    labels1 = np.asarray(layout1["category"])
    labels2 = np.asarray(layout2["category"])
    label_cost = (labels1[:, None] != labels2[None, :]).astype(np.float64)
    return spatial_cost * 0.5 + label_cost * 0.5


def _emd(layout1: InternalLayout, layout2: InternalLayout) -> float:
    if layout1 == layout2:
        return 0.0
    n_elem = len(layout1["category"])
    m_elem = len(layout2["category"])
    if n_elem == 0 or m_elem == 0:
        return 1.0
    a = np.ones(n_elem, dtype=np.float64) / n_elem
    b = np.ones(m_elem, dtype=np.float64) / m_elem
    cost = _cost_matrix(layout1, layout2)
    plan = ot.emd(a, b, cost)
    return float(np.sum(plan * cost))


def _ltsim(layout1: InternalLayout, layout2: InternalLayout) -> float:
    return float(np.exp(-_emd(layout1, layout2)))


def _emd_matrix(
    layouts1: List[InternalLayout], layouts2: List[InternalLayout]
) -> npt.NDArray[np.float64]:
    return np.asarray(
        [[_emd(x, y) for y in layouts2] for x in layouts1], dtype=np.float64
    )


def _ltsim_mmd(layouts1: List[InternalLayout], layouts2: List[InternalLayout]) -> float:
    if len(layouts1) < 2:
        return float("nan")
    emd11 = _emd_matrix(layouts1, layouts1)
    emd22 = _emd_matrix(layouts2, layouts2)
    emd12 = _emd_matrix(layouts1, layouts2)
    sigma = np.median(emd11[np.triu_indices(len(emd11), k=1)])
    if sigma == 0.0:
        return float("nan")
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        affinity11 = np.exp(-emd11 / sigma)
        affinity22 = np.exp(-emd22 / sigma)
        affinity12 = np.exp(-emd12 / sigma)
    upper11 = np.triu_indices(len(affinity11))
    upper22 = np.triu_indices(len(affinity22))
    return float(
        np.mean(affinity11[upper11])
        + np.mean(affinity22[upper22])
        - 2.0 * np.mean(affinity12)
    )


@add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class LayoutLTSim(evaluate.Metric):
    def _info(self) -> evaluate.EvaluationModuleInfo:
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=ds.Features(
                {
                    "layouts1": {
                        "bboxes": ds.Sequence(ds.Sequence(ds.Value("float64"))),
                        "categories": ds.Sequence(ds.Value("int64")),
                    },
                    "layouts2": {
                        "bboxes": ds.Sequence(ds.Sequence(ds.Value("float64"))),
                        "categories": ds.Sequence(ds.Value("int64")),
                    },
                }
            ),
            codebase_urls=[
                "https://github.com/mayu-ot/ltsim/tree/544b8bccfce4630591589cdec5c5473392190039",
                "https://pythonot.github.io/master/all.html#ot.emd",
            ],
        )

    def _compute(
        self,
        *,
        layouts1: List[Layout],
        layouts2: List[Layout],
    ) -> Dict[str, float]:
        if len(layouts1) != len(layouts2):
            raise ValueError("all LTSim inputs must have the same batch length")
        normalised_layouts1 = [_layout(layout) for layout in layouts1]
        normalised_layouts2 = [_layout(layout) for layout in layouts2]
        pairwise = np.asarray(
            [_ltsim(x, y) for x, y in zip(normalised_layouts1, normalised_layouts2)]
        )
        return {
            "ltsim": float(pairwise.mean()),
            "ltsim_mmd": _ltsim_mmd(normalised_layouts1, normalised_layouts2),
        }
