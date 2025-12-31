import os
from typing import List, Union

import datasets as ds
import evaluate
import numpy as np
import numpy.typing as npt
from evaluate.utils.file_utils import add_start_docstrings
from PIL import Image

_DESCRIPTION = r"""\
Computes the utilization rate of space suitable for arranging elements, implemented by the negative image S' of the compounded saliency map S.
"""

_KWARGS_DESCRIPTION = """\
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
class LayoutUtility(evaluate.Metric):
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
                    "saliency_maps_1": ds.Sequence(ds.Value("string")),
                    "saliency_maps_2": ds.Sequence(ds.Value("string")),
                }
            ),
            codebase_urls=[
                "https://github.com/PKU-ICST-MIPL/PosterLayout-CVPR2023/blob/main/eval.py#L144-L171"
            ],
        )

    def load_saliency_map(
        self,
        filepath: Union[os.PathLike, List[os.PathLike]],
    ) -> npt.NDArray[np.float64]:
        if isinstance(filepath, list):
            assert len(filepath) == 1, filepath
            filepath = filepath[0]

        map_pil = Image.open(filepath)  # type: ignore
        map_pil = map_pil.convert("L")  # type: ignore

        if map_pil.size != (self.canvas_width, self.canvas_height):
            map_pil = map_pil.resize((self.canvas_width, self.canvas_height))  # type: ignore

        map_arr = np.array(map_pil)
        map_arr = map_arr / 255.0
        return map_arr

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

    def _compute(
        self,
        *,
        predictions: Union[npt.NDArray[np.float64], List[List[float]]],
        gold_labels: Union[npt.NDArray[np.int64], List[int]],
        saliency_maps_1: List[os.PathLike],
        saliency_maps_2: List[os.PathLike],
    ) -> float:
        predictions = np.array(predictions)
        gold_labels = np.array(gold_labels)

        predictions[:, :, ::2] *= self.canvas_width
        predictions[:, :, 1::2] *= self.canvas_height

        gold_labels = self.get_rid_of_invalid(
            predictions=predictions, gold_labels=gold_labels
        )

        score = []

        assert (
            len(predictions)
            == len(gold_labels)
            == len(saliency_maps_1)
            == len(saliency_maps_2)
        )
        it = zip(predictions, gold_labels, saliency_maps_1, saliency_maps_2)

        for prediction, gold_label, smap_1, smap_2 in it:
            smap_arr_1 = self.load_saliency_map(smap_1)
            smap_arr_2 = self.load_saliency_map(smap_2)

            smap_arr = np.maximum(smap_arr_1, smap_arr_2)
            c_smap = np.ones_like(smap_arr) - smap_arr

            cal_mask = np.zeros_like(smap_arr)

            prediction = np.array(prediction, dtype=int)
            gold_label = np.array(gold_label, dtype=int)

            mask = (gold_label > 0).reshape(-1)
            mask_prediction = prediction[mask]

            for mp in mask_prediction:
                xl, yl, xr, yr = mp
                cal_mask[yl:yr, xl:xr] = 1

            total_not_sal = np.sum(c_smap)
            total_utils = np.sum(c_smap * cal_mask)

            if total_not_sal and total_utils:
                # score += total_utils / total_not_sal
                score.append(total_utils / total_not_sal)

        # return score / num_predictions
        return np.mean(score)
