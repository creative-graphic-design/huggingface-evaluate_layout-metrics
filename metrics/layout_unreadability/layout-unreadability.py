import os
from typing import List, Literal, Optional, Union

import cv2
import datasets as ds
import evaluate
import numpy as np
import numpy.typing as npt
from evaluate.utils.file_utils import add_start_docstrings
from PIL import Image
from PIL.Image import Image as PilImage

_DESCRIPTION = r"""\
Computes the non-flatness of regions that text elements are solely put on, referring to CGL-GAN.
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

ReqType = Literal["pil2cv", "cv2pil"]


@add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class LayoutUnreadability(evaluate.Metric):
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
                    "image_canvases": ds.Sequence(ds.Value("string")),
                }
            ),
            codebase_urls=[
                "https://github.com/PKU-ICST-MIPL/PosterLayout-CVPR2023/blob/main/eval.py#L144-L171"
            ],
        )

    def cvt_pilcv(
        self,
        img: Union[PilImage, npt.NDArray[np.float64]],
        req: ReqType = "pil2cv",
        color_code: Optional[int] = None,
    ) -> Union[PilImage, npt.NDArray[np.float64]]:
        if req == "pil2cv":
            assert isinstance(img, PilImage)
            color_code = color_code or cv2.COLOR_RGB2BGR
            return cv2.cvtColor(np.asarray(img), color_code)  # type: ignore
        elif req == "cv2pil":
            assert isinstance(img, np.ndarray)
            color_code = color_code or cv2.COLOR_BGR2RGB
            return Image.fromarray(cv2.cvtColor(img, color_code))
        else:
            raise ValueError("req should be 'pil2cv' or 'cv2pil'")

    def img_to_g_xy(self, img):
        img_cv_gs = self.cvt_pilcv(img, req="pil2cv", color_code=cv2.COLOR_RGB2GRAY)
        assert isinstance(img_cv_gs, np.ndarray)
        img_cv_gs = np.uint8(img_cv_gs)

        # Sobel(src, ddepth, dx, dy)
        grad_x = cv2.Sobel(img_cv_gs, -1, 1, 0)
        grad_y = cv2.Sobel(img_cv_gs, -1, 0, 1)
        grad_xy = ((grad_x**2 + grad_y**2) / 2) ** 0.5
        grad_xy = grad_xy / np.max(grad_xy) * 255
        img_g_xy = Image.fromarray(grad_xy).convert("L")
        return img_g_xy

    def load_image_canvas(
        self,
        filepath: Union[os.PathLike, List[os.PathLike]],
    ) -> npt.NDArray[np.float64]:
        if isinstance(filepath, list):
            assert len(filepath) == 1, filepath
            filepath = filepath[0]

        canvas_pil = Image.open(filepath)  # type: ignore
        canvas_pil = canvas_pil.convert("RGB")  # type: ignore
        if canvas_pil.size != (self.canvas_width, self.canvas_height):
            canvas_pil = canvas_pil.resize((self.canvas_width, self.canvas_height))  # type: ignore

        canvas_pil = self.img_to_g_xy(canvas_pil)
        assert isinstance(canvas_pil, PilImage)
        canvas_arr = np.array(canvas_pil) / 255.0

        return canvas_arr

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
        image_canvases: List[os.PathLike],
    ):
        predictions = np.array(predictions)
        gold_labels = np.array(gold_labels)

        predictions[:, :, ::2] *= self.canvas_width
        predictions[:, :, 1::2] *= self.canvas_height

        gold_labels = self.get_rid_of_invalid(
            predictions=predictions, gold_labels=gold_labels
        )
        score = 0.0

        assert len(predictions) == len(gold_labels) == len(image_canvases)
        num_predictions = len(predictions)
        it = zip(predictions, gold_labels, image_canvases)

        for prediction, gold_label, image_canvas in it:
            canvas_arr = self.load_image_canvas(
                image_canvas,
            )
            cal_mask = np.zeros_like(canvas_arr)

            prediction = np.array(prediction, dtype=int)
            gold_label = np.array(gold_label, dtype=int)

            is_text = (gold_label == self.text_label_index).reshape(-1)
            prediction_text = prediction[is_text]

            is_decoration = (gold_label == self.decoration_label_index).reshape(-1)
            prediction_deco = prediction[is_decoration]

            for mp in prediction_text:
                xl, yl, xr, yr = mp
                cal_mask[yl:yr, xl:xr] = 1
            for mp in prediction_deco:
                xl, yl, xr, yr = mp
                cal_mask[yl:yr, xl:xr] = 0

            total_area = np.sum(cal_mask)
            total_grad = np.sum(canvas_arr[cal_mask == 1])
            if total_area and total_grad:
                score += total_grad / total_area
        return score / num_predictions
