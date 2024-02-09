from typing import Dict, List, Tuple, Union

import datasets as ds
import evaluate
import numpy as np
import numpy.typing as npt

_DESCRIPTION = """\
Computes some alignment metrics that are different to each other in previous works.
"""

_CITATION = """\
@inproceedings{lee2020neural,
  title={Neural design network: Graphic layout generation with constraints},
  author={Lee, Hsin-Ying and Jiang, Lu and Essa, Irfan and Le, Phuong B and Gong, Haifeng and Yang, Ming-Hsuan and Yang, Weilong},
  booktitle={Computer Vision--ECCV 2020: 16th European Conference, Glasgow, UK, August 23--28, 2020, Proceedings, Part III 16},
  pages={491--506},
  year={2020},
  organization={Springer}
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


class LayoutAlignment(evaluate.Metric):
    def _info(self) -> evaluate.EvaluationModuleInfo:
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            features=ds.Features(
                {
                    "batch_bbox": ds.Sequence(ds.Sequence(ds.Value("float64"))),
                    "batch_mask": ds.Sequence(ds.Value("bool")),
                }
            ),
            codebase_urls=[
                "https://github.com/ktrk115/const_layout/blob/master/metric.py#L167-L188",
                "https://github.com/CyberAgentAILab/layout-dm/blob/main/src/trainer/trainer/helpers/metric.py#L98-L147",
            ],
        )

    def _compute_ac_layout_gan(
        self,
        S: int,
        xl: npt.NDArray[np.float64],
        xc: npt.NDArray[np.float64],
        xr: npt.NDArray[np.float64],
        yt: npt.NDArray[np.float64],
        yc: npt.NDArray[np.float64],
        yb: npt.NDArray[np.float64],
        batch_mask: npt.NDArray,
    ) -> npt.NDArray[np.float64]:
        # shape: (B, 6, S)
        X = np.stack((xl, xc, xr, yt, yc, yb), axis=1)
        # shape: (B, 6, S, 1) - (B, 6, 1, S) = (B, 6 S, S)
        X = X[:, :, :, None] - X[:, :, None, :]

        # shape: (S,)
        indices = np.arange(S)
        X[:, :, indices, indices] = 1.0
        # shape: (B, 6, S, S -> (B, S, 6, S)
        X = np.abs(X).transpose(0, 2, 1, 3)
        X[~batch_mask] = 1.0

        # shape: (B, S)
        X = X.min(axis=-1).min(axis=-1)
        X[X == 1.0] = 0.0
        X = -np.log(1 - X)

        # shape: (B, S) -> (B,)
        return X.sum(axis=1)

    def _compute_layout_gan_pp(
        self,
        score_ac_layout_gan: npt.NDArray[np.float64],
        batch_mask: npt.NDArray[np.bool_],
    ) -> npt.NDArray[np.float64]:
        # shape: (B, S) -> (B,)
        batch_mask = batch_mask.sum(axis=1)

        # shape: (B,)
        score_normalized = score_ac_layout_gan / batch_mask
        score_normalized[np.isnan(score_normalized)] = 0.0
        return score_normalized

    def _compute_neural_design_network(
        self,
        xl: npt.NDArray[np.float64],
        xc: npt.NDArray[np.float64],
        xr: npt.NDArray[np.float64],
        batch_mask: npt.NDArray[np.bool_],
        S: int,
    ):
        # shape: (B, 3, S)
        Y = np.stack((xl, xc, xr), axis=1)
        # shape: (B, 3, S, S)
        Y = Y[:, :, None, :] - Y[:, :, :, None]

        # shape: (B, S) -> (B, S, S)
        batch_mask = ~batch_mask[:, None, :] | ~batch_mask[:, :, None]
        # shape: (B,)
        indices = np.arange(S)
        batch_mask[:, indices, indices] = True

        # shape: (B, S, S) -> (B, 1, S, S) -> (B, 3, S, S)
        batch_mask = np.repeat(batch_mask[:, None, :, :], repeats=3, axis=1)
        Y[batch_mask] = 1.0

        # shape: (B, 3, S, S) -> (B, S, S) -> (B, S)
        Y = np.abs(Y).min(axis=1).min(axis=2)
        Y[Y == 1.0] = 0.0

        # shape: (B, S) -> (B,)
        score = Y.sum(axis=1)
        return score

    def _compute(
        self,
        *,
        batch_bbox: Union[npt.NDArray[np.float64], List[List[int]]],
        batch_mask: Union[npt.NDArray[np.bool_], List[List[bool]]],
    ) -> Dict[str, npt.NDArray[np.float64]]:
        # shape: (B, model_max_length, C)
        batch_bbox = np.array(batch_bbox)
        # shape: (B, model_max_length)
        batch_mask = np.array(batch_mask)

        # S: model_max_length
        _, S, _ = batch_bbox.shape

        # shape: (B, S, C) -> (C, B, S)
        batch_bbox = batch_bbox.transpose(2, 0, 1)
        xl, yt, xr, yb = convert_xywh_to_ltrb(batch_bbox)
        xc, yc = batch_bbox[0], batch_bbox[1]

        # shape: (B,)
        score_ac_layout_gan = self._compute_ac_layout_gan(
            S=S, xl=xl, xc=xc, xr=xr, yt=yt, yc=yc, yb=yb, batch_mask=batch_mask
        )
        # shape: (B,)
        score_layout_gan_pp = self._compute_layout_gan_pp(
            score_ac_layout_gan=score_ac_layout_gan, batch_mask=batch_mask
        )
        score_ndn = self._compute_neural_design_network(
            xl=xl, xc=xc, xr=xr, batch_mask=batch_mask, S=S
        )
        return {
            "alignment-ACLayoutGAN": score_ac_layout_gan,
            "alignment-LayoutGAN++": score_layout_gan_pp,
            "alignment-NDN": score_ndn,
        }
