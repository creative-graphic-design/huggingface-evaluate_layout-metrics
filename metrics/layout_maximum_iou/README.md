---
title: Layout Maximum IoU
emoji: 📊
colorFrom: pink
colorTo: purple
sdk: gradio
sdk_version: 4.36.1
app_file: app.py
pinned: false
---

# Layout Maximum IoU

## Description

The Layout Maximum IoU metric computes the optimal matching between two sets of layouts and measures their similarity using Intersection over Union (IoU). This metric is particularly useful for evaluating conditional layout generation models by comparing generated layouts against reference layouts with the same element composition.

## What It Measures

This metric:

1. Groups layouts by their element category composition (e.g., layouts with {text, image, logo})
2. Finds the optimal one-to-one matching between layouts in each group using the Hungarian algorithm
3. Computes IoU for matched element pairs within each layout
4. Returns the average IoU across all optimally matched layout pairs

Higher values indicate better similarity between the two layout sets.

## Metric Details

- **Category-conditional matching**: Only compares layouts with identical element type compositions
- **Optimal assignment**: Uses linear sum assignment (Hungarian algorithm) to find the best matching
- **Element-wise IoU**: Computes IoU for each element pair after optimal matching
- **Normalized by layout size**: Final score is divided by the number of elements per layout
- Returns 0.0 if no layouts share the same category composition

## Usage

### Installation

```bash
pip install evaluate scipy
```

### Basic Example

```python
import evaluate
import numpy as np

# Load the metric
metric = evaluate.load("creative-graphic-design/layout-maximum-iou")

# Single pair of layouts
num_samples, num_coordinates = 24, 4
layout1 = {
    "bboxes": np.random.rand(num_samples, num_coordinates),
    "categories": np.random.randint(0, num_coordinates, size=(num_samples,)),
}
layout2 = {
    "bboxes": np.random.rand(num_samples, num_coordinates),
    "categories": np.random.randint(0, num_coordinates, size=(num_samples,)),
}
metric.add(layouts1=layout1, layouts2=layout2)
print(metric.compute())
```

### Batch Processing Example

```python
import evaluate
import numpy as np

# Load the metric
metric = evaluate.load("creative-graphic-design/layout-maximum-iou")

# Batch processing (recommended)
batch_size, num_samples, num_coordinates = 512, 24, 4
layouts1 = [
    {
        "bboxes": np.random.rand(num_samples, num_coordinates),
        "categories": np.random.randint(0, num_coordinates, size=(num_samples,)),
    }
    for _ in range(batch_size)
]
layouts2 = [
    {
        "bboxes": np.random.rand(num_samples, num_coordinates),
        "categories": np.random.randint(0, num_coordinates, size=(num_samples,)),
    }
    for _ in range(batch_size)
]
metric.add_batch(layouts1=layouts1, layouts2=layouts2)
print(metric.compute())
```

## Parameters

### Initialization Parameters

This metric does not require any initialization parameters.

### Computation Parameters

- **layouts1** (`list` of `dict`): First set of layouts, where each dictionary contains:

  - **bboxes** (`list` of `float`): Bounding boxes in center-x, center-y, width, height (xywh) format
  - **categories** (`list` of `int`): Category labels for each element

- **layouts2** (`list` of `dict`): Second set of layouts (same structure as layouts1)

**Note**: Layouts are automatically grouped by their category composition. Only layouts with matching categories are compared.

## Returns

Returns a single `float` value representing:

- The mean maximum IoU across all optimally matched layout pairs
- Range: 0.0 to 1.0

## Interpretation

- **Higher is better** (range: 0.0 to 1.0)
- **Value of 1.0**: Perfect match - all elements in matched layouts have identical positions and sizes
- **Value of 0.8-1.0**: Very high similarity - generated layouts closely match references
- **Value of 0.5-0.8**: Moderate similarity - layouts capture general structure but differ in details
- **Value of 0.0-0.5**: Low similarity - significant differences in element placement
- **Value of 0.0**: Either no overlap in matched elements, or no layouts share the same category composition

### Use Cases

- **Conditional generation evaluation**: Assess how well a model generates layouts matching specific element compositions
- **Layout retrieval**: Find the best matching layouts between two collections
- **Dataset comparison**: Compare layout distributions between different sources

### Key Features

- **Optimal matching**: Ensures fair comparison by finding the best possible pairing
- **Category-aware**: Only compares layouts that should be comparable (same elements)
- **Robust to ordering**: Element order within layouts doesn't affect the score

## Citations

```bibtex
@inproceedings{kikuchi2021constrained,
  title={Constrained graphic layout generation via latent optimization},
  author={Kikuchi, Kotaro and Simo-Serra, Edgar and Otani, Mayu and Yamaguchi, Kota},
  booktitle={Proceedings of the 29th ACM International Conference on Multimedia},
  pages={88--96},
  year={2021}
}
```

## References

- **Paper**: [Constrained Graphic Layout Generation via Latent Optimization (Kikuchi et al., ACM MM 2021)](https://arxiv.org/abs/2108.00871)
- **Reference Implementation**: [layout-dm metric implementation](https://github.com/CyberAgentAILab/layout-dm/blob/main/src/trainer/trainer/helpers/metric.py#L206-L371)
- **Hugging Face Space**: [creative-graphic-design/layout-maximum-iou](https://huggingface.co/spaces/creative-graphic-design/layout-maximum-iou)

## Related Metrics

- [Layout Average IoU](../layout_average_iou/): Measures average overlap within single layouts
- [Layout Overlap](../layout_overlap/): Alternative overlap metrics
- [Layout Validity](../layout_validity/): Validates layout constraints
