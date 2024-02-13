# 🤗 Layout Evaluation Metrics by Huggingface Evaluate
[![CI](https://github.com/shunk031/huggingface-evaluate_layout-metrics/actions/workflows/ci.yaml/badge.svg)](https://github.com/shunk031/huggingface-evaluate_layout-metrics/actions/workflows/ci.yaml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A collection of metrics to evaluate layout generation that can be easily used in 🤗 huggingface [evaluate](https://huggingface.co/docs/evaluate/index).

| 📊 Metric | 🤗 Space |
|:---------:|:---------:|
| [![FID](https://github.com/shunk031/huggingface-evaluate_layout-metrics/actions/workflows/layout_generative_model_scores.yaml/badge.svg)](https://github.com/shunk031/huggingface-evaluate_layout-metrics/actions/workflows/layout_generative_model_scores.yaml) | [`pytorch-layout-generation/layout-generative-model-scores`](https://huggingface.co/spaces/pytorch-layout-generation/layout-generative-model-scores) |
| [![Max. IoU](https://github.com/shunk031/huggingface-evaluate_layout-metrics/actions/workflows/layout_maximum_iou.yaml/badge.svg)](https://github.com/shunk031/huggingface-evaluate_layout-metrics/actions/workflows/layout_maximum_iou.yaml) | [`pytorch-layout-generation/layout-maximum-iou`](https://huggingface.co/spaces/pytorch-layout-generation/layout-maximum-iou) |
| [![Avg. IoU](https://github.com/shunk031/huggingface-evaluate_layout-metrics/actions/workflows/layout_average_iou.yaml/badge.svg)](https://github.com/shunk031/huggingface-evaluate_layout-metrics/actions/workflows/layout_average_iou.yaml) | [`pytorch-layout-generation/layout-average-iou`](https://huggingface.co/spaces/pytorch-layout-generation/layout-average-iou) |
| [![Alignment](https://github.com/shunk031/huggingface-evaluate_layout-metrics/actions/workflows/layout_alignment.yaml/badge.svg)](https://github.com/shunk031/huggingface-evaluate_layout-metrics/actions/workflows/layout_alignment.yaml) | [`pytorch-layout-generation/layout-alignment`](https://huggingface.co/spaces/pytorch-layout-generation/layout-alignment) |
| [![Overlap](https://github.com/shunk031/huggingface-evaluate_layout-metrics/actions/workflows/layout_overlap.yaml/badge.svg)](https://github.com/shunk031/huggingface-evaluate_layout-metrics/actions/workflows/layout_overlap.yaml) | [`pytorch-layout-generation/layout-overlap`](https://huggingface.co/spaces/pytorch-layout-generation/layout-overlap) |

# How to use

- Install [`evaluate`](https://huggingface.co/docs/evaluate/index) library

```shell
pip install evaluate
```

- Load the layout metric and then compute the score

```python
import evaluate
import numpy as np

# Load the evaluation metric named "pytorch-layout-generation/layout-alignment"
alignment_score = evaluate.load("pytorch-layout-generation/layout-alignment")

# `batch_bbox` is a tensor representing (batch_size, max_num_elements, coordinates) 
# and `batch_mask` is a tensor representing (batch_size, max_num_elements).
batch_bbox = np.random.rand(512, 25, 4)
batch_mask = np.random.rand(512, 25)

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
