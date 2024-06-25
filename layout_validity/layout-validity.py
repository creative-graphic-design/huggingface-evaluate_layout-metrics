from typing import List, Union

import datasets as ds
import evaluate
import numpy as np
import numpy.typing as npt

_DESCRIPTION = r"""\
Computes the ratio of valid elements to all elements in the layout, where the area within the canvas of a valid element must be greater than 0.1% of the canvas.
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


class LayoutValidity(evaluate.Metric):
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
                "https://github.com/PKU-ICST-MIPL/PosterLayout-CVPR2023/blob/main/eval.py#L105-L127"
            ],
        )

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

        total_elements, empty_elements = 0, 0

        w = self.canvas_width / 100
        h = self.canvas_height / 100

        assert len(predictions) == len(gold_labels)

        for gold_label, prediction in zip(gold_labels, predictions):
            mask = (gold_label > 0).reshape(-1)
            mask_box = prediction[mask]
            total_elements += len(mask_box)
            for mb in mask_box:
                xl, yl, xr, yr = mb
                xl = max(0, xl)
                yl = max(0, yl)
                xr = min(self.canvas_width, xr)
                yr = min(self.canvas_height, yr)

                if abs((xr - xl) * (yr - yl)) < w * h * 10:
                    empty_elements += 1

        return 1 - empty_elements / total_elements
