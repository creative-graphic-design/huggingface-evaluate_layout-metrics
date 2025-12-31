import copy
import math
from typing import List, Union

import datasets as ds
import evaluate
import numpy as np
import numpy.typing as npt

_DESCRIPTION = r"""\
Computes the extent of spatial non-alignment between elements.
"""

_KWARGS_DESCRIPTION = """\
Args:
    predictions (`list` of `list` of `float`): A list of lists of floats representing normalized `ltrb`-format bounding boxes.
    gold_labels (`list` of `list` of `int`): A list of lists of integers representing class labels.
    canvas_width (`int`, *optional*): Width of the canvas in pixels. Can be provided at initialization or during computation.
    canvas_height (`int`, *optional*): Height of the canvas in pixels. Can be provided at initialization or during computation.

Returns:
    float: The extent of spatial non-alignment between elements. Lower values indicate better alignment. Evaluates alignment across six aspects: left edge, top edge, center X, center Y, right edge, and bottom edge.

Examples:
    >>> import evaluate
    >>> metric = evaluate.load("creative-graphic-design/layout-non-alignment")
    >>> # Normalized bounding boxes (left, top, right, bottom)
    >>> predictions = [[[0.1, 0.1, 0.3, 0.3], [0.1, 0.4, 0.3, 0.6]]]  # Left-aligned elements
    >>> gold_labels = [[1, 2]]
    >>> result = metric.compute(predictions=predictions, gold_labels=gold_labels, canvas_width=512, canvas_height=512)
    >>> print(f"Non-alignment score: {result:.4f}")
"""

_CITATION = """\
@inproceedings{hsu2023posterlayout,
  title={Posterlayout: A new benchmark and approach for content-aware visual-textual presentation layout},
  author={Hsu, Hsiao Yuan and He, Xiangteng and Peng, Yuxin and Kong, Hao and Zhang, Qing},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={6018--6026},
  year={2023}
}

@article{li2020attribute,
  title={Attribute-conditioned layout gan for automatic graphic design},
  author={Li, Jianan and Yang, Jimei and Zhang, Jianming and Liu, Chang and Wang, Christina and Xu, Tingfa},
  journal={IEEE Transactions on Visualization and Computer Graphics},
  volume={27},
  number={10},
  pages={4039--4048},
  year={2020},
  publisher={IEEE}
}
"""


class LayoutNonAlignment(evaluate.Metric):
    def __init__(
        self,
        canvas_width: int | None = None,
        canvas_height: int | None = None,
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
                "https://github.com/PKU-ICST-MIPL/PosterLayout-CVPR2023/blob/main/eval.py#L306-L339"
            ],
        )

    def ali_delta(self, xs: npt.NDArray[np.float64]) -> float:
        n = len(xs)
        min_delta = np.inf
        for i in range(n):
            for j in range(i + 1, n):
                delta = abs(xs[i] - xs[j])
                min_delta = min(min_delta, delta)
        return min_delta

    def ali_g(self, x: float) -> float:
        return -math.log(1 - x, 10)

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

    def _compute(
        self,
        *,
        predictions: Union[npt.NDArray[np.float64], List[List[float]]],
        gold_labels: Union[npt.NDArray[np.int64], List[int]],
        canvas_width: int | None = None,
        canvas_height: int | None = None,
    ) -> float:
        # パラメータの優先順位処理
        canvas_width = canvas_width if canvas_width is not None else self.canvas_width
        canvas_height = (
            canvas_height if canvas_height is not None else self.canvas_height
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

        metrics: float = 0.0
        for gold_label, prediction in zip(gold_labels, predictions):
            ali = 0.0
            mask = (gold_label > 0).reshape(-1)
            mask_box = prediction[mask]

            theda = []
            for mb in mask_box:
                pos = copy.deepcopy(mb)
                pos[0] /= canvas_width
                pos[2] /= canvas_width
                pos[1] /= canvas_height
                pos[3] /= canvas_height
                theda.append(
                    [
                        pos[0],
                        pos[1],
                        (pos[0] + pos[2]) / 2,
                        (pos[1] + pos[3]) / 2,
                        pos[2],
                        pos[3],
                    ]
                )
            theda_arr = np.array(theda)
            if theda_arr.shape[0] <= 1:
                continue

            n = len(mask_box)
            for _ in range(n):
                g_val = []
                for j in range(6):
                    xys = theda_arr[:, j]
                    delta = self.ali_delta(xys)
                    g_val.append(self.ali_g(delta))
                ali += min(g_val)
            metrics += ali

        return metrics / len(gold_labels)
