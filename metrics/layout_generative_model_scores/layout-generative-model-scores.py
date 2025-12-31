from typing import Dict, List, Union

import datasets as ds
import evaluate
import numpy as np
import numpy.typing as npt
from prdc import compute_prdc
from pytorch_fid.fid_score import calculate_frechet_distance

_DESCRIPTION = """\
Compute some generative model-based scores.
"""

_KWARGS_DESCRIPTION = """\
Args:
    feats_real (`list` of `list` of `float`): A list of lists of floats representing real features.
    feats_fake (`list` of `list` of `float`): A list of lists of floats representing fake features.

Returns:
    dictionaly: A set of generative model-based scores.

Examples:

    Example 1: Single processing
        >>> metric = evaluate.load("creative-graphic-design/layout-generative-model-scores")
        >> feat_size = 256
        >>> feats_real = np.random.rand(feat_size)
        >>> feats_fake = np.random.rand(feat_size)
        >>> metric.add(feats_real=feats_real, feats_fake=feats_fake)
        >>> print(metric.compute())
    
    Example 2: Batch processing
        >>> metric = evaluate.load("creative-graphic-design/layout-generative-model-scores")
        >>> batch_size, feat_size = 512, 256
        >>> feats_real = np.random.rand(batch_size, feat_size)
        >>> feats_fake = np.random.rand(batch_size, feat_size)
        >>> metric.add_batch(feats_real=feats_real, feats_fake=feats_fake)
        >>> print(metric.compute())
"""

_CITATION = """\
@article{heusel2017gans,
  title={Gans trained by a two time-scale update rule converge to a local nash equilibrium},
  author={Heusel, Martin and Ramsauer, Hubert and Unterthiner, Thomas and Nessler, Bernhard and Hochreiter, Sepp},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}

@inproceedings{naeem2020reliable,
  title={Reliable fidelity and diversity metrics for generative models},
  author={Naeem, Muhammad Ferjad and Oh, Seong Joon and Uh, Youngjung and Choi, Yunjey and Yoo, Jaejun},
  booktitle={International Conference on Machine Learning},
  pages={7176--7185},
  year={2020},
  organization={PMLR}
}
"""


class LayoutGenerativeModelScores(evaluate.Metric):
    def __init__(self, nearest_k: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.nearest_k = nearest_k

    def _info(self) -> evaluate.EvaluationModuleInfo:
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=ds.Features(
                {
                    "feats_real": ds.Sequence(ds.Value("float64")),
                    "feats_fake": ds.Sequence(ds.Value("float64")),
                }
            ),
            codebase_urls=[
                "https://github.com/CyberAgentAILab/layout-dm/blob/main/src/trainer/trainer/helpers/metric.py#L37-L59",
                "https://github.com/clovaai/generative-evaluation-prdc",
                "https://github.com/mseitzer/pytorch-fid",
            ],
        )

    def _compute(
        self,
        *,
        feats_real: Union[List[List[float]], npt.NDArray[np.float64]],
        feats_fake: Union[List[List[float]], npt.NDArray[np.float64]],
        nearest_k: int | None = None,
    ) -> Dict[str, float]:
        # パラメータの優先順位処理
        nearest_k = nearest_k if nearest_k is not None else self.nearest_k

        # shape: (N, 256)
        feats_real = np.asarray(feats_real)
        feats_fake = np.asarray(feats_fake)

        # shape: (256,)
        mu_real = np.mean(feats_real, axis=0)
        mu_fake = np.mean(feats_fake, axis=0)

        # shape: (256,)
        sigma_real = np.cov(feats_real, rowvar=False)
        sigma_fake = np.cov(feats_fake, rowvar=False)

        results_prdc = compute_prdc(
            real_features=feats_real, fake_features=feats_fake, nearest_k=nearest_k
        )
        result_fid = calculate_frechet_distance(
            mu1=mu_real, sigma1=sigma_real, mu2=mu_fake, sigma2=sigma_fake
        )
        return {**results_prdc, "fid": result_fid}
