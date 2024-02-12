from typing import Dict, List, Tuple, TypedDict

import datasets as ds
import evaluate
import numpy as np
import numpy.typing as npt

_DESCRIPTION = """\
Computes some average IoU metrics that are different to each other in previous works.
"""

_CITATION = """\
@inproceedings{arroyo2021variational,
  title={Variational transformer networks for layout generation},
  author={Arroyo, Diego Martin and Postels, Janis and Tombari, Federico},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={13642--13652},
  year={2021}
}

@inproceedings{kong2022blt,
  title={BLT: bidirectional layout transformer for controllable layout generation},
  author={Kong, Xiang and Jiang, Lu and Chang, Huiwen and Zhang, Han and Hao, Yuan and Gong, Haifeng and Essa, Irfan},
  booktitle={European Conference on Computer Vision},
  pages={474--490},
  year={2022},
  organization={Springer}
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


class Layout(TypedDict):
    bboxes: npt.NDArray[np.float64]
    categories: npt.NDArray[np.int64]


def compute_iou(
    bbox1: npt.NDArray[np.float64],
    bbox2: npt.NDArray[np.float64],
    generalized: bool = False,
) -> npt.NDArray[np.float64]:
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


def compute_perceptual_iou(
    bbox1: npt.NDArray[np.float64],
    bbox2: npt.NDArray[np.float64],
    N: int = 32,
) -> npt.NDArray[np.float64]:
    """
    Computes 'Perceptual' IoU [Kong+, BLT'22]
    """

    # shape: bbox1 (N, 4), bbox2 (N, 4)
    assert bbox1.shape[0] == bbox2.shape[0]
    assert bbox1.shape[1] == bbox1.shape[1] == 4

    l1, t1, r1, b1 = convert_xywh_to_ltrb(bbox1.T)
    l2, t2, r2, b2 = convert_xywh_to_ltrb(bbox2.T)
    a1 = (r1 - l1) * (b1 - t1)

    # intersection
    l_max = np.maximum(l1, l2)
    r_min = np.minimum(r1, r2)
    t_max = np.maximum(t1, t2)
    b_min = np.minimum(b1, b2)
    cond = (l_max < r_min) & (t_max < b_min)
    ai = np.where(cond, (r_min - l_max) * (b_min - t_max), np.zeros_like(a1[0]))

    unique_box_1 = np.unique(bbox1, axis=0)

    l1, t1, r1, b1 = [
        (x * N).round().astype(np.int32).clip(0, N)
        for x in convert_xywh_to_ltrb(unique_box_1.T)
    ]
    canvas = np.zeros((N, N))
    for left, top, right, bottom in zip(l1, t1, r1, b1):
        canvas[top:bottom, left:right] = 1
    global_area_union = canvas.sum() / (N**2)

    return ai / global_area_union if global_area_union > 0.0 else np.zeros((1,))


def compute_average_iou(layout: Layout, perceptual: bool) -> float:
    bboxes = np.asarray(layout["bboxes"])

    N = len(bboxes)
    if N in [0, 1]:
        return 0.0  # no overlap in principle

    ii, jj = np.meshgrid(range(N), range(N))
    ii, jj = ii.flatten(), jj.flatten()
    is_non_diag = ii != jj  # IoU for diag is always 1.0
    ii, jj = ii[is_non_diag], jj[is_non_diag]

    iou = (
        compute_perceptual_iou(bboxes[ii], bboxes[jj])
        if perceptual
        else compute_iou(bboxes[ii], bboxes[jj])
    )
    # pick all pairs of overlapped objects
    cond = iou > np.finfo(np.float32).eps  # to avoid very-small nonzero
    score = iou[cond].mean().item() if len(iou[cond]) > 0 else 0.0

    return score


class LayoutAverageIoU(evaluate.Metric):
    def _info(self) -> evaluate.EvaluationModuleInfo:
        return evaluate.EvaluationModuleInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            features=ds.Features(
                {
                    "layouts": {
                        "bboxes": ds.Sequence(ds.Sequence((ds.Value("float64")))),
                        "categories": ds.Sequence(ds.Value("int64")),
                    }
                }
            ),
            codebase_urls=[
                "https://github.com/CyberAgentAILab/layout-dm/blob/main/src/trainer/trainer/helpers/metric.py#L399-L431",
            ],
        )

    def _compute(self, *, layouts: List[Layout]) -> Dict[str, float]:
        scores_blt = [
            compute_average_iou(layout, perceptual=True) for layout in layouts
        ]
        scores_vnt = [
            compute_average_iou(layout, perceptual=False) for layout in layouts
        ]
        score_blt = np.mean(scores_blt).item()
        score_vnt = np.mean(scores_vnt).item()

        results = {
            "average-iou_BLT": score_blt,
            "average-iou_VTN": score_vnt,
        }
        return results
