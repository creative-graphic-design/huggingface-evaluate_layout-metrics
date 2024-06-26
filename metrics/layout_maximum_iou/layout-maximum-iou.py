from collections import defaultdict
from itertools import chain
from typing import Dict, List, Tuple, TypedDict

import datasets as ds
import evaluate
import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment


class Layout(TypedDict):
    bboxes: npt.NDArray[np.float64]
    categories: npt.NDArray[np.int64]


_DESCRIPTION = """\
Compute the maximum IoU between two sets of layouts.
"""

_KWARGS_DESCRIPTION = """\
Args:
    layouts1 (`list` of `dict`): A list of dictionaries representing layouts including `list` of `bboxes` (float) and `list` of `categories` (int).
    layouts2 (`list` of `dict`): A list of dictionaries representing layouts including `list` of `bboxes` (float) and `list` of `categories` (int).

Returns:
    float: The maximum IoU score.

Examples:

    Example 1: Single processing
        >>> metric = evaluate.load("creative-graphic-design/layout-maximum-iou")
        >>> num_samples, num_categories = 24, 4
        >>> layout1 = {
        >>>     "bboxes": np.random.rand(num_samples, num_categories),
        >>>     "categories": np.random.randint(0, num_categories, size=(num_samples,)),
        >>> }
        >>> layout2 = {
        >>>     "bboxes": np.random.rand(num_samples, num_categories),
        >>>     "categories": np.random.randint(0, num_categories, size=(num_samples,)),
        >>> }
        >>> metric.add(layouts1=layout1, layouts2=layout2)
        >>> print(metric.compute())

    Example 2: Batch processing
        >>> metric = evaluate.load("creative-graphic-design/layout-maximum-iou")
        >>> batch_size, num_samples, num_categories = 512, 24, 4
        >>> layouts1 = [
        >>>     {
        >>>         "bboxes": np.random.rand(num_samples, num_categories),
        >>>         "categories": np.random.randint(0, num_categories, size=(num_samples,)),
        >>>     }
        >>>     for _ in range(batch_size)
        >>> ]
        >>> layouts2 = [
        >>>     {
        >>>         "bboxes": np.random.rand(num_samples, num_categories),
        >>>         "categories": np.random.randint(0, num_categories, size=(num_samples,)),
        >>>     }
        >>>     for _ in range(batch_size)
        >>> ]
        >>> metric.add_batch(layouts1=layouts1, layouts2=layouts2)
        >>> print(metric.compute())
"""

_CITATION = """\
@inproceedings{kikuchi2021constrained,
  title={Constrained graphic layout generation via latent optimization},
  author={Kikuchi, Kotaro and Simo-Serra, Edgar and Otani, Mayu and Yamaguchi, Kota},
  booktitle={Proceedings of the 29th ACM International Conference on Multimedia},
  pages={88--96},
  year={2021}
}
"""


def convert_xywh_to_ltrb(
    batch_bbox: npt.NDArray[np.float64],
) -> Tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    xc, yc, w, h = batch_bbox
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return (x1, y1, x2, y2)


def _compute_iou(
    bbox1: npt.NDArray[np.float64],
    bbox2: npt.NDArray[np.float64],
    generalized: bool = False,
):
    # shape: bbox1 (N, 4), bbox2 (N, 4)
    assert bbox1.shape[0] == bbox2.shape[0]
    assert bbox1.shape[1] == bbox1.shape[1] == 4

    l1, t1, r1, b1 = convert_xywh_to_ltrb(bbox1.T)
    l2, t2, r2, b2 = convert_xywh_to_ltrb(bbox2.T)
    a1, a2 = (r1 - l1) * (b1 - t1), (r2 - l2) * (b2 - t2)

    # intersection
    l_max = np.maximum(l1, l2)
    r_min = np.minimum(r1, r2)
    t_max = np.maximum(t1, t2)
    b_min = np.minimum(b1, b2)
    cond = (l_max < r_min) & (t_max < b_min)
    ai = np.where(cond, (r_min - l_max) * (b_min - t_max), np.zeros_like(a1[0]))

    au = a1 + a2 - ai
    iou = ai / au

    if not generalized:
        return iou

    # outer region
    l_min = np.minimum(l1, l2)
    r_max = np.maximum(r1, r2)
    t_min = np.minimum(t1, t2)
    b_max = np.maximum(b1, b2)
    ac = (r_max - l_min) * (b_max - t_min)

    giou = iou - (ac - au) / ac

    return giou


def _compute_maximum_iou_for_layout(layout1: Layout, layout2: Layout):
    score = 0.0
    bi, ci = layout1["bboxes"], layout1["categories"]
    bj, cj = layout2["bboxes"], layout2["categories"]
    N = len(bi)

    for c in list(set(ci.tolist())):
        _bi = bi[np.where(ci == c)]
        _bj = bj[np.where(cj == c)]
        n = len(_bi)
        ii, jj = np.meshgrid(range(n), range(n))
        ii, jj = ii.flatten(), jj.flatten()
        iou = _compute_iou(_bi[ii], _bj[jj]).reshape(n, n)
        # Note: maximize is supported only when scipy >= 1.4
        ii, jj = linear_sum_assignment(iou, maximize=True)
        score += iou[ii, jj].sum().item()
    return score / N


def _compute_maximum_iou(
    layouts_1_and_2: Tuple[List[Layout], List[Layout]],
) -> npt.NDArray[np.float64]:
    assert len(layouts_1_and_2) == 2
    layouts1, layouts2 = layouts_1_and_2

    N, M = len(layouts1), len(layouts2)
    ii, jj = np.meshgrid(range(N), range(M))
    ii, jj = ii.flatten(), jj.flatten()
    scores = np.asarray(
        [
            _compute_maximum_iou_for_layout(layouts1[i], layouts2[j])
            for i, j in zip(ii, jj)
        ]
    )
    scores = scores.reshape(N, M)
    ii, jj = linear_sum_assignment(scores, maximize=True)
    return scores[ii, jj]


def _get_cond_to_layouts(layouts: List[Layout]) -> Dict[str, List[Layout]]:
    out = defaultdict(list)

    for layout in layouts:
        bboxes = layout["bboxes"]
        categories = layout["categories"]

        # e.g., [18, 2, 1, 20, 0, 0, 0, 0, 0, 9, 9, 5, 0, 5, 0, 0]
        # -> "[0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 5, 5, 9, 9, 18, 20]"
        cond_key = str(sorted(categories))

        categories = np.array(categories)
        layout_dict: Layout = {
            "bboxes": np.asarray(bboxes),
            "categories": np.asarray(categories),
        }
        out[cond_key].append(layout_dict)

    return out


def compute_maximum_iou(args):
    return [_compute_maximum_iou(a) for a in args]


class LayoutMaximumIoU(evaluate.Metric):
    def _info(self) -> evaluate.EvaluationModuleInfo:
        return evaluate.EvaluationModuleInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=ds.Features(
                {
                    "layouts1": {
                        "bboxes": ds.Sequence(ds.Sequence((ds.Value("float64")))),
                        "categories": ds.Sequence(ds.Value("int64")),
                    },
                    "layouts2": {
                        "bboxes": ds.Sequence(ds.Sequence((ds.Value("float64")))),
                        "categories": ds.Sequence(ds.Value("int64")),
                    },
                }
            ),
            codebase_urls=[
                "https://github.com/CyberAgentAILab/layout-dm/blob/main/src/trainer/trainer/helpers/metric.py#L206-L247",
                "https://github.com/CyberAgentAILab/layout-dm/blob/main/src/trainer/trainer/helpers/metric.py#L250-L297",
                "https://github.com/CyberAgentAILab/layout-dm/blob/main/src/trainer/trainer/helpers/metric.py#L300-L314",
                "https://github.com/CyberAgentAILab/layout-dm/blob/main/src/trainer/trainer/helpers/metric.py#L317-L329",
                "https://github.com/CyberAgentAILab/layout-dm/blob/main/src/trainer/trainer/helpers/metric.py#L332-L340",
                "https://github.com/CyberAgentAILab/layout-dm/blob/main/src/trainer/trainer/helpers/metric.py#L343-L371",
            ],
        )

    def _compute(
        self,
        *,
        layouts1: List[Layout],
        layouts2: List[Layout],
    ) -> float:
        c2bl_1 = _get_cond_to_layouts(layouts1)
        keys_1 = set(c2bl_1.keys())
        c2bl_2 = _get_cond_to_layouts(layouts2)
        keys_2 = set(c2bl_2.keys())
        keys = list(keys_1.intersection(keys_2))
        args = [(c2bl_1[key], c2bl_2[key]) for key in keys]

        # to check actual number of layouts for evaluation
        # ans = 0
        # for x in args:
        #     ans += len(x[0])

        scores = compute_maximum_iou(args)
        scores = np.asarray(list(chain.from_iterable(scores)))

        return scores.mean().item() if len(scores) != 0 else 0.0
