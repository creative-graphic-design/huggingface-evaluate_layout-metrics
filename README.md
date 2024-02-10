# 🤗 Layout Evaluation Metrics by Huggingface Evaluate
[![CI](https://github.com/shunk031/huggingface-evaluate_layout-metrics/actions/workflows/ci.yaml/badge.svg)](https://github.com/shunk031/huggingface-evaluate_layout-metrics/actions/workflows/ci.yaml)



| 📊 Metric | 🤗 Space |
|:---------:|:---------:|
| [![Alignment](https://github.com/shunk031/huggingface-evaluate_layout-metrics/actions/workflows/layout_alignment.yaml/badge.svg)](https://github.com/shunk031/huggingface-evaluate_layout-metrics/actions/workflows/layout_alignment.yaml) | [`shunk031/layout_alignment`](https://huggingface.co/spaces/shunk031/layout_alignment) |
| [![Overlap](https://github.com/shunk031/huggingface-evaluate_layout-metrics/actions/workflows/layout_overlap.yaml/badge.svg)](https://github.com/shunk031/huggingface-evaluate_layout-metrics/actions/workflows/layout_overlap.yaml) | [`shunk031/layout_overlap`](https://huggingface.co/spaces/shunk031/layout_overlap) |

# How to use

- Install [`evaluate`](https://huggingface.co/docs/evaluate/index) library

```shell
pip install evaluate
```

- Load the layout metric and then compute the score

```python
import evaluate
import numpy as np

# Load the evaluation metric named "shunk031/layout_alignment"
alignment_score = evaluate.load("shunk031/layout_alignment")

# `batch_bbox` is a tensor representing (batch_size, max_num_elements, coordinates) 
# and `batch_mask` is a tensor representing (batch_size, max_num_elements).
batch_bbox = np.random.rand(512, 25, 4)
batch_mask = np.random.rand(512, 25)

# Add the batch of bboxes and masks to the metric
alignment_score.add_batch(batch_bbox=batch_bbox, batch_mask=batch_mask)
# Perform the computation of the evaluation metric
alignment_score.compute()
```

## References

- Li, Jianan, et al. "LayoutGAN: Generating Graphic Layouts with Wireframe Discriminators." International Conference on Learning Representations. 2019.
- Lee, Hsin-Ying, et al. "Neural design network: Graphic layout generation with constraints." Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part III 16. Springer International Publishing, 2020.
- Li, Jianan, et al. "Attribute-conditioned layout gan for automatic graphic design." IEEE Transactions on Visualization and Computer Graphics 27.10 (2020): 4039-4048.
- Kikuchi, Kotaro, et al. "Constrained graphic layout generation via latent optimization." Proceedings of the 29th ACM International Conference on Multimedia. 2021.
