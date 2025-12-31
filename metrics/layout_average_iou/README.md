---
title: Layout Average IoU
emoji: 📊
colorFrom: pink
colorTo: purple
sdk: gradio
sdk_version: 4.36.1
app_file: app.py
pinned: false
---

# Layout Average IoU

## Description

The Layout Average IoU metric computes the average Intersection over Union (IoU) for all pairs of overlapping elements in a layout. This metric evaluates how efficiently elements are arranged by measuring the degree of overlap between layout components.

## What It Measures

This metric implements two variants of average IoU that have been used in different layout generation research:

1. **VTN (Variational Transformer Networks)**: Standard geometric IoU calculation
2. **BLT (Bidirectional Layout Transformer)**: Perceptual IoU that considers the global union area on a discrete grid

Lower values generally indicate better layouts with less overlap between elements.

## Metric Details

- Computes IoU for all pairs of elements in a layout (excluding diagonal comparisons where elements would overlap with themselves)
- Only considers pairs with IoU > 0 (actual overlap)
- Returns the mean IoU across all overlapping pairs
- Returns 0.0 for layouts with 0 or 1 elements (no overlap possible)

## Usage

### Installation

```bash
pip install evaluate
```

### Basic Example

```python
import evaluate
import numpy as np

# Load the metric
metric = evaluate.load("creative-graphic-design/layout-average-iou")

# Single layout processing
num_samples, num_coordinates = 24, 4
layout = {
    "bboxes": np.random.rand(num_samples, num_coordinates),
    "categories": np.random.randint(0, num_coordinates, size=(num_samples,)),
}
metric.add(layouts=layout)
print(metric.compute())
```

### Batch Processing Example

```python
import evaluate
import numpy as np

# Load the metric
metric = evaluate.load("creative-graphic-design/layout-average-iou")

# Batch processing
batch_size, num_samples, num_coordinates = 512, 24, 4
layouts = [
    {
        "bboxes": np.random.rand(num_samples, num_coordinates),
        "categories": np.random.randint(0, num_coordinates, size=(num_samples,)),
    }
    for _ in range(batch_size)
]
metric.add_batch(layouts=layouts)
print(metric.compute())
```

## Parameters

### Initialization Parameters

This metric does not require any initialization parameters.

### Computation Parameters

- **layouts** (`list` of `dict`): A list of dictionaries representing layouts, where each dictionary contains:
  - **bboxes** (`list` of `float`): Bounding boxes in center-x, center-y, width, height (xywh) format
  - **categories** (`list` of `int`): Category labels for each element

## Returns

Returns a dictionary containing:

- **average-iou_BLT** (`float`): Average IoU using the perceptual IoU method from BLT
- **average-iou_VTN** (`float`): Average IoU using standard geometric IoU from VTN

## Interpretation

- **Lower values** indicate better layouts with less element overlap
- **Value of 0.0**: No overlapping elements (ideal for most layout types)
- **Higher values**: More overlap between elements, potentially indicating layout quality issues
- **Typical range**: 0.0 to 1.0

The two variants (BLT and VTN) may produce slightly different values due to their different calculation methods:

- **VTN** uses standard geometric IoU
- **BLT** uses perceptual IoU with discrete grid quantization (32x32)

## Citations

```bibtex
@inproceedings{arroyo2021variational,
  title={Variational transformer networks for layout generation},
  author={Arroyo, Diego Martin and Postels, Janis and Tombari, Federico},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={13642--13652},
  year={2021}
}

@inproceedings{kong2022blt,
  title={BLT: bidirectional layout transformer for controllable layout generation},
  author={Kong, Xiang and Jiang, Lu and Chang, Huiwen and Zhang, Han and Hao, Yuan and Gong, Haifeng and Essa, Irfan},
  booktitle={European Conference on Computer Vision},
  pages={474--490},
  year={2022},
  organization={Springer}
}
```

## References

- **Paper**: [Variational Transformer Networks for Layout Generation (Arroyo et al., CVPR 2021)](https://arxiv.org/abs/2104.02416)
- **Paper**: [BLT: Bidirectional Layout Transformer for Controllable Layout Generation (Kong et al., ECCV 2022)](https://arxiv.org/abs/2112.05112)
- **Reference Implementation**: [layout-dm metric implementation](https://github.com/CyberAgentAILab/layout-dm/blob/main/src/trainer/trainer/helpers/metric.py#L399-L431)
- **Hugging Face Space**: [creative-graphic-design/layout-average-iou](https://huggingface.co/spaces/creative-graphic-design/layout-average-iou)

## Related Metrics

- [Layout Maximum IoU](../layout_maximum_iou/): Measures maximum IoU between two layout sets
- [Layout Overlap](../layout_overlap/): Alternative overlap metrics from various research works
- [Layout Alignment](../layout_alignment/): Measures spatial alignment of layout elements
