from typing import List, Union

import datasets as ds
import evaluate
import numpy as np
import numpy.typing as npt
from evaluate.utils.file_utils import add_start_docstrings

_DESCRIPTION = r"""\
Computes the average IoU of all pairs of elements except for underlay.
"""

_KWARGS_DESCRIPTION = """\
Args:
    predictions (`list` of `list` of `float`): A list of lists of floats representing normalized `ltrb`-format bounding boxes.
    gold_labels (`list` of `list` of `int`): A list of lists of integers representing class labels.
    canvas_width (`int`, *optional*): Width of the canvas in pixels. Can be provided at initialization or during computation.
    canvas_height (`int`, *optional*): Height of the canvas in pixels. Can be provided at initialization or during computation.
    decoration_label_index (`int`, *optional*, defaults to 3): The label index for decoration (underlay) elements to exclude from overlay computation.

Returns:
    float: Average IoU (Intersection over Union) of all pairs of elements except decoration (underlay) elements. Higher values indicate more overlap between elements.

Examples:
    >>> import evaluate
    >>> metric = evaluate.load("creative-graphic-design/layout-overlay")
    >>> # Normalized bounding boxes (left, top, right, bottom)
    >>> predictions = [[[0.1, 0.1, 0.5, 0.5], [0.3, 0.3, 0.7, 0.7]]]  # Overlapping elements
    >>> gold_labels = [[1, 2]]  # Both are non-decoration elements
    >>> result = metric.compute(predictions=predictions, gold_labels=gold_labels, canvas_width=512, canvas_height=512)
    >>> print(f"Overlay score: {result:.4f}")
"""

_CITATION = """\
@inproceedings{hsu2023posterlayout,
  title={Posterlayout: A new benchmark and approach for content-aware visual-textual presentation layout},
  author={Hsu, Hsiao Yuan and He, Xiangteng and Peng, Yuxin and Kong, Hao and Zhang, Qing},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={6018--6026},
  year={2023}
}
"""


@add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class LayoutOverlay(evaluate.Metric):
    def __init__(
        self,
        canvas_width: int | None = None,
        canvas_height: int | None = None,
        decoration_label_index: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.decoration_label_index = decoration_label_index

    def _info(self) -> evaluate.EvaluationModuleInfo:
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=ds.Features(
                {
                    "predictions": ds.Sequence(ds.Sequence(ds.Value("float64"))),
                    "gold_labels": ds.Sequence(ds.Sequence(ds.Value("int64"))),
                }
            ),
            codebase_urls=[
                "https://github.com/PKU-ICST-MIPL/PosterLayout-CVPR2023/blob/main/eval.py#L205-L222",
            ],
        )

    def get_rid_of_invalid(
        self,
        predictions: npt.NDArray[np.float64],
        gold_labels: npt.NDArray[np.int64],
        canvas_width: int,
        canvas_height: int,
    ) -> npt.NDArray[np.int64]:
        assert len(predictions) == len(gold_labels)

        w = canvas_width / 100
        h = canvas_height / 100

        for i, prediction in enumerate(predictions):
            for j, b in enumerate(prediction):
                xl, yl, xr, yr = b
                xl = max(0, xl)
                yl = max(0, yl)
                xr = min(canvas_width, xr)
                yr = min(canvas_height, yr)
                if abs((xr - xl) * (yr - yl)) < w * h * 10:
                    if gold_labels[i, j]:
                        gold_labels[i, j] = 0
        return gold_labels

    def metrics_iou(
        self, bb1: npt.NDArray[np.float64], bb2: npt.NDArray[np.float64]
    ) -> float:
        # shape: bb1 = (4,), bb2 = (4,)
        xl_1, yl_1, xr_1, yr_1 = bb1
        xl_2, yl_2, xr_2, yr_2 = bb2

        w_1 = xr_1 - xl_1
        w_2 = xr_2 - xl_2
        h_1 = yr_1 - yl_1
        h_2 = yr_2 - yl_2

        w_inter = min(xr_1, xr_2) - max(xl_1, xl_2)
        h_inter = min(yr_1, yr_2) - max(yl_1, yl_2)

        a_1 = w_1 * h_1
        a_2 = w_2 * h_2
        a_inter = w_inter * h_inter
        if w_inter <= 0 or h_inter <= 0:
            a_inter = 0

        return a_inter / (a_1 + a_2 - a_inter)

    def _compute(
        self,
        *,
        predictions: Union[npt.NDArray[np.float64], List[List[float]]],
        gold_labels: Union[npt.NDArray[np.int64], List[int]],
        canvas_width: int | None = None,
        canvas_height: int | None = None,
        decoration_label_index: int | None = None,
    ) -> float:
        # パラメータの優先順位処理
        canvas_width = canvas_width if canvas_width is not None else self.canvas_width
        canvas_height = (
            canvas_height if canvas_height is not None else self.canvas_height
        )
        decoration_label_index = (
            decoration_label_index
            if decoration_label_index is not None
            else self.decoration_label_index
        )

        if canvas_width is None or canvas_height is None:
            raise ValueError(
                "canvas_width and canvas_height must be provided either "
                "at initialization or during computation"
            )

        predictions = np.array(predictions)
        gold_labels = np.array(gold_labels)

        predictions[:, :, ::2] *= canvas_width
        predictions[:, :, 1::2] *= canvas_height

        gold_labels = self.get_rid_of_invalid(
            predictions=predictions,
            gold_labels=gold_labels,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
        )

        score = 0.0

        for gold_label, prediction in zip(gold_labels, predictions):
            ove = 0.0

            cond1 = (gold_label > 0).reshape(-1)
            cond2 = (gold_label != decoration_label_index).reshape(-1)

            mask = cond1 & cond2
            mask_box = prediction[mask]

            n = len(mask_box)
            for i in range(n):
                bb1 = mask_box[i]
                for j in range(i + 1, n):
                    bb2 = mask_box[j]
                    ove += self.metrics_iou(bb1, bb2)
            score += ove / n
        return score / len(gold_labels)
