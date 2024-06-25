import os
from typing import List, Literal, Optional, Union

import cv2
import datasets as ds
import evaluate
import numpy as np
import numpy.typing as npt
from PIL import Image
from PIL.Image import Image as PilImage

_DESCRIPTION = r"""\
Computes the average IoU of all pairs of elements except for underlay.
"""

_KWARGS_DESCRIPTION = """\
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


class LayoutOverlay(evaluate.Metric):
    def __init__(
        self,
        canvas_width: int,
        canvas_height: int,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height

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
        self, predictions: npt.NDArray[np.float64], gold_labels: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.int64]:
        assert len(predictions) == len(gold_labels)

        w = self.canvas_width / 100
        h = self.canvas_height / 100

        for i, prediction in enumerate(predictions):
            for j, b in enumerate(prediction):
                xl, yl, xr, yr = b
                xl = max(0, xl)
                yl = max(0, yl)
                xr = min(self.canvas_width, xr)
                yr = min(self.canvas_height, yr)
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
    ) -> float:
        predictions = np.array(predictions)
        gold_labels = np.array(gold_labels)

        predictions[:, :, ::2] *= self.canvas_width
        predictions[:, :, 1::2] *= self.canvas_height

        gold_labels = self.get_rid_of_invalid(
            predictions=predictions, gold_labels=gold_labels
        )

        score: float = 0.0

        for gold_label, prediction in zip(gold_labels, predictions):
            ove = 0.0
            mask = (gold_label > 0).reshape(-1) & (gold_label != 3).reshape(-1)
            mask_box = prediction[mask]
            n = len(mask_box)
            for i in range(n):
                bb1 = mask_box[i]
                for j in range(i + 1, n):
                    bb2 = mask_box[j]
                    ove += self.metrics_iou(bb1, bb2)
            score += ove / n
        return score / len(gold_labels)
