# 🤗 Layout Evaluation Metrics by Huggingface Evaluate
[![CI](https://github.com/creative-graphic-design/huggingface-evaluate_layout-metrics/actions/workflows/ci.yaml/badge.svg)](https://github.com/creative-graphic-design/huggingface-evaluate_layout-metrics/actions/workflows/ci.yaml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A collection of metrics to evaluate layout generation that can be easily used in 🤗 huggingface [evaluate](https://huggingface.co/docs/evaluate/index).

| 📊 Metric | 🤗 Space | 📝 Paper |
|:---------:|:---------|:----------|
| [![FID](https://github.com/creative-graphic-design/huggingface-evaluate_layout-metrics/actions/workflows/layout_generative_model_scores.yaml/badge.svg)](https://github.com/creative-graphic-design/huggingface-evaluate_layout-metrics/actions/workflows/layout_generative_model_scores.yaml) | [`creative-graphic-design/layout-generative-model-scores`](https://huggingface.co/spaces/creative-graphic-design/layout-generative-model-scores) | [[Heusel+ NeurIPS'17](https://arxiv.org/abs/1706.08500)], [[Naeem+ ICML'20](https://arxiv.org/abs/2002.09797)] |
| [![Max. IoU](https://github.com/creative-graphic-design/huggingface-evaluate_layout-metrics/actions/workflows/layout_maximum_iou.yaml/badge.svg)](https://github.com/creative-graphic-design/huggingface-evaluate_layout-metrics/actions/workflows/layout_maximum_iou.yaml) | [`creative-graphic-design/layout-maximum-iou`](https://huggingface.co/spaces/creative-graphic-design/layout-maximum-iou) | [[Kikuchi+ ACMMM'21](https://arxiv.org/abs/2108.00871)] |
| [![Avg. IoU](https://github.com/creative-graphic-design/huggingface-evaluate_layout-metrics/actions/workflows/layout_average_iou.yaml/badge.svg)](https://github.com/creative-graphic-design/huggingface-evaluate_layout-metrics/actions/workflows/layout_average_iou.yaml) | [`creative-graphic-design/layout-average-iou`](https://huggingface.co/spaces/creative-graphic-design/layout-average-iou) | [[Arroyo+ CVPR'21](https://arxiv.org/abs/2104.02416)], [[Kong+ ECCV'22](https://arxiv.org/abs/2112.05112)] |
| [![Alignment](https://github.com/creative-graphic-design/huggingface-evaluate_layout-metrics/actions/workflows/layout_alignment.yaml/badge.svg)](https://github.com/creative-graphic-design/huggingface-evaluate_layout-metrics/actions/workflows/layout_alignment.yaml) | [`creative-graphic-design/layout-alignment`](https://huggingface.co/spaces/creative-graphic-design/layout-alignment) | [[Lee+ ECCV'20](https://arxiv.org/abs/1912.09421)], [[Li+ TVCG'21](https://arxiv.org/abs/2009.05284)], [[Kikuchi+ ACMMM'21](https://arxiv.org/abs/2108.00871)] |
| [![Overlap](https://github.com/creative-graphic-design/huggingface-evaluate_layout-metrics/actions/workflows/layout_overlap.yaml/badge.svg)](https://github.com/creative-graphic-design/huggingface-evaluate_layout-metrics/actions/workflows/layout_overlap.yaml) | [`creative-graphic-design/layout-overlap`](https://huggingface.co/spaces/creative-graphic-design/layout-overlap) | [[Li+ ICLR'19](https://arxiv.org/abs/1901.06767)], [[Li+ TVCG'21](https://arxiv.org/abs/2009.05284)], [[Kikuchi+ ACMMM'21](https://arxiv.org/abs/2108.00871)] |
| [![Validity](https://github.com/creative-graphic-design/huggingface-evaluate_layout-metrics/actions/workflows/layout_validity.yaml/badge.svg)](https://github.com/creative-graphic-design/huggingface-evaluate_layout-metrics/actions/workflows/layout_validity.yaml) | | [[Hsu+ CVPR'23](https://arxiv.org/abs/2303.15937)] |
| [![Occlusion](https://github.com/creative-graphic-design/huggingface-evaluate_layout-metrics/actions/workflows/layout_occlusion.yaml/badge.svg)](https://github.com/creative-graphic-design/huggingface-evaluate_layout-metrics/actions/workflows/layout_occlusion.yaml) | | |
| [![Overlap](https://github.com/creative-graphic-design/huggingface-evaluate_layout-metrics/actions/workflows/layout_overlap.yaml/badge.svg)](https://github.com/creative-graphic-design/huggingface-evaluate_layout-metrics/actions/workflows/layout_overlap.yaml) | | |
| [![Overlay](https://github.com/creative-graphic-design/huggingface-evaluate_layout-metrics/actions/workflows/layout_overlay.yaml/badge.svg)](https://github.com/creative-graphic-design/huggingface-evaluate_layout-metrics/actions/workflows/layout_overlay.yaml) | | |
| [![Underlay Effectiveness](https://github.com/creative-graphic-design/huggingface-evaluate_layout-metrics/actions/workflows/layout_underlay_effectivness.yaml/badge.svg)](https://github.com/creative-graphic-design/huggingface-evaluate_layout-metrics/actions/workflows/layout_underlay_effectivness.yaml) | | |
| [![Unreadability](https://github.com/creative-graphic-design/huggingface-evaluate_layout-metrics/actions/workflows/layout_unreadability.yaml/badge.svg)](https://github.com/creative-graphic-design/huggingface-evaluate_layout-metrics/actions/workflows/layout_unreadability.yaml) | | |
| [![Non-Alignment](https://github.com/creative-graphic-design/huggingface-evaluate_layout-metrics/actions/workflows/layout_non_alignment.yaml/badge.svg)](https://github.com/creative-graphic-design/huggingface-evaluate_layout-metrics/actions/workflows/layout_non_alignment.yaml) | | |

# Usage

- Install [`evaluate`](https://huggingface.co/docs/evaluate/index) library

```shell
pip install evaluate
```

- Load the layout metric and then compute the score

```python
import evaluate
import numpy as np

# Load the evaluation metric named "creative-graphic-design/layout-alignment"
alignment_score = evaluate.load("creative-graphic-design/layout-alignment")

# `batch_bbox` is a tensor representing (batch_size, max_num_elements, coordinates) 
# and `batch_mask` is a boolean tensor representing (batch_size, max_num_elements).
batch_bbox = np.random.rand(512, 25, 4)
# Note that padded fields will be set to `False`
batch_mask = np.full((512, 25), fill_value=True)

# Add the batch of bboxes and masks to the metric
alignment_score.add_batch(batch_bbox=batch_bbox, batch_mask=batch_mask)
# Perform the computation of the evaluation metric
alignment_score.compute()
```

## Reference

- Heusel, Martin, et al. "[Gans trained by a two time-scale update rule converge to a local nash equilibrium.](https://arxiv.org/abs/1706.08500)" Advances in neural information processing systems 30 (2017).
- Li, Jianan, et al. "[LayoutGAN: Generating Graphic Layouts with Wireframe Discriminators.](https://arxiv.org/abs/1901.06767)" International Conference on Learning Representations. 2019.
- Lee, Hsin-Ying, et al. "[Neural design network: Graphic layout generation with constraints.](https://arxiv.org/abs/1912.09421)" Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part III 16. Springer International Publishing, 2020.
- Naeem, Muhammad Ferjad, et al. "[Reliable fidelity and diversity metrics for generative models.](https://arxiv.org/abs/2002.09797)" International Conference on Machine Learning. PMLR, 2020.
- Li, Jianan, et al. "[Attribute-conditioned layout gan for automatic graphic design.](https://arxiv.org/abs/2009.05284)" IEEE Transactions on Visualization and Computer Graphics 27.10 (2020): 4039-4048.
- Kikuchi, Kotaro, et al. "[Constrained graphic layout generation via latent optimization.](https://arxiv.org/abs/2108.00871)" Proceedings of the 29th ACM International Conference on Multimedia. 2021.
- Arroyo, Diego Martin, Janis Postels, and Federico Tombari. "[Variational transformer networks for layout generation.](https://arxiv.org/abs/2104.02416)" Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.
- Kong, Xiang, et al. "[BLT: bidirectional layout transformer for controllable layout generation.](https://arxiv.org/abs/2112.05112)" European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2022.
- Hsu, Hsiao Yuan, et al. "[Posterlayout: A new benchmark and approach for content-aware visual-textual presentation layout.](https://arxiv.org/abs/2303.15937)" Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.
