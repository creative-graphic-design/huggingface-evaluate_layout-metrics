from typing import Dict, List, Union

import datasets as ds
import evaluate
import numpy as np
import numpy.typing as npt
from evaluate.utils.file_utils import add_start_docstrings

_DESCRIPTION = r"""\
Computes the ratio of valid underlay elements to total underlay elements used in PosterLayout. Intuitively, underlay should be placed under other non-underlay elements.
- strict: scoring the underlay as:
    - 1: there is a non-underlay element completely inside
    - 0: otherwise
- loose: Calcurate (ai/a2).
"""

_KWARGS_DESCRIPTION = """\
Args:
    predictions (`list` of `lists` of `float`): A list of lists of floats representing normalized `ltrb`-format bounding boxes.
    gold_labels (`list` of `lists` of `int`): A list of lists of integers representing class labels.

Returns:
    float: The ratio of valid underlay elements to total underlay elements.

Examples:
    FIXME
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
class LayoutUnderlayEffectiveness(evaluate.Metric):
    def __init__(
        self,
        canvas_width: int,
        canvas_height: int,
        text_label_index: int = 1,
        decoration_label_index: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height

        self.text_label_index = text_label_index
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
                "https://github.com/PKU-ICST-MIPL/PosterLayout-CVPR2023/blob/main/eval.py#L224-L252",
                "https://github.com/PKU-ICST-MIPL/PosterLayout-CVPR2023/blob/main/eval.py#L265-L292",
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

    def metrics_inter_oneside(self, bb1, bb2):
        xl_1, yl_1, xr_1, yr_1 = bb1
        xl_2, yl_2, xr_2, yr_2 = bb2

        # w_1 = xr_1 - xl_1
        w_2 = xr_2 - xl_2
        # h_1 = yr_1 - yl_1
        h_2 = yr_2 - yl_2

        w_inter = min(xr_1, xr_2) - max(xl_1, xl_2)
        h_inter = min(yr_1, yr_2) - max(yl_1, yl_2)

        # a_1 = w_1 * h_1
        a_2 = w_2 * h_2
        a_inter = w_inter * h_inter
        if w_inter <= 0 or h_inter <= 0:
            a_inter = 0

        return a_inter / a_2

    def _compute_und_l(
        self, predictions: npt.NDArray[np.float64], gold_labels: npt.NDArray[np.int64]
    ) -> float:
        # metrics, avali = 0.0, 0
        metrics = []
        avali = 0

        for gold_label, prediction in zip(gold_labels, predictions):
            und = 0
            mask_deco = (gold_label == 3).reshape(-1)
            mask_other = (gold_label > 0).reshape(-1) & (gold_label != 3).reshape(-1)
            box_deco = prediction[mask_deco]
            box_other = prediction[mask_other]
            n1, n2 = len(box_deco), len(box_other)
            if not n1:
                continue

            avali += 1
            for i in range(n1):
                max_ios = 0
                bb1 = box_deco[i]
                for j in range(n2):
                    bb2 = box_other[j]
                    ios = self.metrics_inter_oneside(bb1, bb2)
                    max_ios = max(max_ios, ios)
                und += max_ios
            # metrics += und / n1
            metrics.append(und / n1)

        # return metrics / avali if avali > 0 else 0.0
        # return {"mean": np.mean(metrics), "std": np.std(metrics)}
        return np.mean(metrics)

    def _compute_und_s(
        self, predictions: npt.NDArray[np.float64], gold_labels: npt.NDArray[np.int64]
    ) -> float:
        def is_contain(bb1, bb2):
            xl_1, yl_1, xr_1, yr_1 = bb1
            xl_2, yl_2, xr_2, yr_2 = bb2

            c1 = xl_1 <= xl_2
            c2 = yl_1 <= yl_2
            c3 = xr_2 >= xr_2
            c4 = yr_1 >= yr_2

            return c1 and c2 and c3 and c4

        # metrics, avali = 0.0, 0
        metrics = []
        avali = 0

        for gold_label, prediction in zip(gold_labels, predictions):
            und = 0
            mask_deco = (gold_label == 3).reshape(-1)
            mask_other = (gold_label > 0).reshape(-1) & (gold_label != 3).reshape(-1)
            box_deco = prediction[mask_deco]
            box_other = prediction[mask_other]
            n1, n2 = len(box_deco), len(box_other)
            if not n1:
                continue

            avali += 1
            for i in range(n1):
                bb1 = box_deco[i]
                for j in range(n2):
                    bb2 = box_other[j]
                    if is_contain(bb1, bb2):
                        und += 1
                        break
            # metrics += und / n1
            metrics.append(und / n1)

        # return metrics / avali if avali > 0 else 0.0
        return np.mean(metrics)

    def _compute(
        self,
        *,
        predictions: Union[npt.NDArray[np.float64], List[List[float]]],
        gold_labels: Union[npt.NDArray[np.int64], List[int]],
    ) -> Dict[str, float]:
        predictions = np.array(predictions)
        gold_labels = np.array(gold_labels)

        predictions[:, :, ::2] *= self.canvas_width
        predictions[:, :, 1::2] *= self.canvas_height

        gold_labels = self.get_rid_of_invalid(
            predictions=predictions, gold_labels=gold_labels
        )
        return {
            "und_l": self._compute_und_l(
                predictions=predictions, gold_labels=gold_labels
            ),
            "und_s": self._compute_und_s(
                predictions=predictions, gold_labels=gold_labels
            ),
        }
