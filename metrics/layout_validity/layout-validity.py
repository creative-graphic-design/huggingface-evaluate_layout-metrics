from typing import List, Union

import datasets as ds
import evaluate
import numpy as np
import numpy.typing as npt
from evaluate.utils.file_utils import add_start_docstrings

_DESCRIPTION = r"""\
Computes the ratio of valid elements to all elements in the layout, where the area within the canvas of a valid element must be greater than 0.1% of the canvas.
"""

_KWARGS_DESCRIPTION = """\
Args:
    predictions (`list` of `list` of `float`): A list of lists of floats representing normalized `ltrb`-format bounding boxes.
    gold_labels (`list` of `list` of `int`): A list of lists of integers representing class labels.
    canvas_width (`int`, *optional*): Width of the canvas in pixels. Can be provided at initialization or during computation.
    canvas_height (`int`, *optional*): Height of the canvas in pixels. Can be provided at initialization or during computation.

Returns:
    float: The ratio of valid elements to all elements (0.0 to 1.0). An element is considered valid if its area within the canvas is greater than 0.1% of the canvas area.

Examples:
    >>> import evaluate
    >>> import numpy as np
    >>> metric = evaluate.load("creative-graphic-design/layout-validity")
    >>> # Normalized bounding boxes (left, top, right, bottom)
    >>> predictions = [[[0.1, 0.1, 0.5, 0.5], [0.6, 0.6, 0.9, 0.9]]]
    >>> gold_labels = [[1, 2]]  # Non-zero labels indicate valid elements
    >>> result = metric.compute(predictions=predictions, gold_labels=gold_labels, canvas_width=512, canvas_height=512)
    >>> print(result)
    1.0
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
class LayoutValidity(evaluate.Metric):
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
                "https://github.com/PKU-ICST-MIPL/PosterLayout-CVPR2023/blob/main/eval.py#L105-L127"
            ],
        )

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

        total_elements, empty_elements = 0, 0

        w = canvas_width / 100
        h = canvas_height / 100

        assert len(predictions) == len(gold_labels)

        for gold_label, prediction in zip(gold_labels, predictions):
            mask = (gold_label > 0).reshape(-1)
            mask_prediction = prediction[mask]
            total_elements += len(mask_prediction)
            for mp in mask_prediction:
                xl, yl, xr, yr = mp
                xl = max(0, xl)
                yl = max(0, yl)
                xr = min(canvas_width, xr)
                yr = min(canvas_height, yr)

                if abs((xr - xl) * (yr - yl)) < w * h * 10:
                    empty_elements += 1

        return 1 - empty_elements / total_elements
