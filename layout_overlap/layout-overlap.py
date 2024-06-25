from typing import Dict, List, Tuple, TypedDict, Union

import datasets as ds
import evaluate
import numpy as np
import numpy.typing as npt

_DESCRIPTION = """\
Some overlap metrics that are different to each other in previous works.
"""

_KWARGS_DESCRIPTION = """\
FIXME
"""

_CITATION = """\
@inproceedings{li2018layoutgan,
  title={LayoutGAN: Generating Graphic Layouts with Wireframe Discriminators},
  author={Li, Jianan and Yang, Jimei and Hertzmann, Aaron and Zhang, Jianming and Xu, Tingfa},
  booktitle={International Conference on Learning Representations},
  year={2019}
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


class A(TypedDict):
    a1: npt.NDArray[np.float64]
    ai: npt.NDArray[np.float64]


class LayoutOverlap(evaluate.Metric):
    def _info(self) -> evaluate.EvaluationModuleInfo:
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=ds.Features(
                {
                    "bbox": ds.Sequence(ds.Sequence(ds.Value("float64"))),
                    "mask": ds.Sequence(ds.Value("bool")),
                }
            ),
            codebase_urls=[
                "https://github.com/ktrk115/const_layout/blob/master/metric.py#L138-L164",
                "https://github.com/CyberAgentAILab/layout-dm/blob/main/src/trainer/trainer/helpers/metric.py#L150-L203",
            ],
        )

    def __calculate_a1_ai(self, batch_bbox: npt.NDArray[np.float64]) -> A:
        l1, t1, r1, b1 = convert_xywh_to_ltrb(batch_bbox[:, :, :, None])
        l2, t2, r2, b2 = convert_xywh_to_ltrb(batch_bbox[:, :, None, :])
        a1 = (r1 - l1) * (b1 - t1)

        # shape: (B, S, S)
        l_max = np.maximum(l1, l2)
        r_min = np.minimum(r1, r2)
        t_max = np.maximum(t1, t2)
        b_min = np.minimum(b1, b2)
        cond = (l_max < r_min) & (t_max < b_min)
        ai = np.where(cond, (r_min - l_max) * (b_min - t_max), 0.0)

        return {"a1": a1, "ai": ai}

    def _compute_ac_layout_gan(
        self,
        S: int,
        ai: npt.NDArray[np.float64],
        a1: npt.NDArray[np.float64],
        batch_mask: npt.NDArray[np.bool_],
    ) -> npt.NDArray[np.float64]:
        # shape: (B, S) -> (B, S, S)
        batch_mask = ~batch_mask[:, None, :] | ~batch_mask[:, :, None]
        indices = np.arange(S)
        batch_mask[:, indices, indices] = True
        ai[batch_mask] = 0.0

        # shape: (B, S, S)
        ar = np.nan_to_num(ai / a1)
        score = ar.sum(axis=(1, 2))

        return score

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

    def _compute_layout_gan(
        self, S: int, B: int, ai: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        indices = np.arange(S)
        ii, jj = np.meshgrid(indices, indices, indexing="ij")

        # shape: ii (S, S) -> (1, S, S), jj (S, S) -> (1, S, S)
        # shape: (1, S, S) -> (B, S, S)
        ai[np.repeat((ii[None, :] >= jj[None, :]), axis=0, repeats=B)] = 0.0

        # shape: (B, S, S) -> (B,)
        score = ai.sum(axis=(1, 2))

        return score

    def _compute(
        self,
        *,
        bbox: Union[npt.NDArray[np.float64], List[List[int]]],
        mask: Union[npt.NDArray[np.bool_], List[List[bool]]],
    ) -> Dict[str, npt.NDArray[np.float64]]:
        # shape: (B, model_max_length, C)
        bbox = np.array(bbox)
        # shape: (B, model_max_length)
        mask = np.array(mask)

        assert bbox.ndim == 3
        assert mask.ndim == 2

        # S: model_max_length
        B, S, C = bbox.shape

        # shape: batch_bbox (B, S, C), batch_mask (B, S) -> (B, S, 1) -> (B, S, C)
        bbox[np.repeat(~mask[:, :, None], axis=2, repeats=C)] = 0.0
        # shape: (C, B, S)
        bbox = bbox.transpose(2, 0, 1)

        A = self.__calculate_a1_ai(bbox)

        # shape: (B,)
        score_ac_layout_gan = self._compute_ac_layout_gan(S=S, batch_mask=mask, **A)
        # shape: (B,)
        score_layout_gan_pp = self._compute_layout_gan_pp(
            score_ac_layout_gan=score_ac_layout_gan, batch_mask=mask
        )
        # shape: (B,)
        score_layout_gan = self._compute_layout_gan(B=B, S=S, ai=A["ai"])

        return {
            "overlap-ACLayoutGAN": score_ac_layout_gan,
            "overlap-LayoutGAN++": score_layout_gan_pp,
            "overlap-LayoutGAN": score_layout_gan,
        }
