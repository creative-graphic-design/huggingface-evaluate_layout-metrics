---
title: Layout Validity
emoji: 🏢
colorFrom: pink
colorTo: pink
sdk: gradio
sdk_version: 4.36.1
app_file: app.py
pinned: false
---

# Layout Validity

## Description

The Layout Validity metric evaluates the basic validity of layout elements by checking whether they meet minimum size requirements. A layout element is considered valid if its area within the canvas boundaries is greater than 0.1% of the total canvas area. This metric helps identify layouts with degenerate or nearly-invisible elements that don't contribute meaningfully to the design.

## What It Measures

This metric computes:

- **Valid element ratio**: Proportion of elements that meet minimum size requirements
- **Layout integrity**: Whether elements are large enough to be functional
- **Degenerate element detection**: Identifies layouts with overly small or nearly-invisible elements

Higher scores indicate better layout validity with fewer problematic elements.

## Metric Details

- Validity threshold: Element area > 0.1% of canvas area (after clamping to canvas bounds)
- Clamps element bounding boxes to canvas boundaries before computing area
- Filters out padding elements (label == 0)
- From PosterLayout (Hsu et al., CVPR 2023) for poster design evaluation
- Returns the ratio of valid elements to total non-padding elements

## Usage

### Installation

```bash
pip install evaluate
```

### Basic Example

```python
import evaluate
import numpy as np

# Load the metric with canvas dimensions
metric = evaluate.load(
    "creative-graphic-design/layout-validity",
    canvas_width=360,
    canvas_height=504
)

# Prepare data (normalized ltrb coordinates)
predictions = np.random.rand(1, 25, 4)
gold_labels = np.random.randint(0, 4, size=(1, 25))
score = metric.compute(predictions=predictions, gold_labels=gold_labels)
print(score)
```

### Batch Processing Example

```python
import evaluate
import numpy as np

# Load the metric
metric = evaluate.load(
    "creative-graphic-design/layout-validity",
    canvas_width=360,
    canvas_height=504
)

# Batch processing
batch_size = 128
predictions = np.random.rand(batch_size, 25, 4)
gold_labels = np.random.randint(0, 4, size=(batch_size, 25))
score = metric.compute(predictions=predictions, gold_labels=gold_labels)
print(score)
```

## Parameters

### Initialization Parameters

- **canvas_width** (`int`, required): Width of the canvas in pixels
- **canvas_height** (`int`, required): Height of the canvas in pixels

### Computation Parameters

- **predictions** (`list` of `lists` of `float`): Normalized bounding boxes in ltrb format (0.0 to 1.0)
- **gold_labels** (`list` of `lists` of `int`): Class labels for each element (0 = padding)

**Note**:

- Elements with `gold_labels == 0` are treated as padding and excluded from computation
- Element coordinates are clamped to [0, canvas_width] × [0, canvas_height] before area calculation
- Validity threshold: area > (canvas_width × canvas_height) / 1000

## Returns

Returns a `float` value representing the ratio of valid elements to total elements (range: 0.0 to 1.0).

## Interpretation

- **Higher is better** (range: 0.0 to 1.0)
- **Value of 1.0**: All elements are valid (ideal)
- **Value of 0.9-1.0**: Mostly valid layout with few problematic elements
- **Value of 0.7-0.9**: Some invalid elements, may need review
- **Value of 0.5-0.7**: Many invalid elements, layout quality concerns
- **Value < 0.5**: Significant validity issues, many degenerate elements
- **Value of 0.0**: All elements invalid (critical failure)

### Use Cases

- **Basic layout quality check**: Ensure generated layouts don't have degenerate elements
- **Generative model evaluation**: Detect models producing invalid element sizes
- **Layout post-processing**: Filter out layouts with validity issues
- **Sanity check**: Validate layout data before further evaluation

### Key Insights

- **Minimum requirement**: This is a basic quality check, not a comprehensive quality metric
- **Threshold choice**: 0.1% of canvas is quite permissive - very small but still visible elements pass
- **Canvas boundaries matter**: Elements extending beyond canvas are clamped, which can reduce their area
- **Foundation metric**: Pass validity check before evaluating other metrics
- **Model debugging**: Low validity scores indicate fundamental generation issues

### Common Causes of Invalidity

- **Collapsed bounding boxes**: Width or height near zero
- **Out-of-bounds elements**: Elements mostly outside canvas boundaries
- **Numerical errors**: Floating-point precision issues in generation
- **Model failure modes**: Generation model producing degenerate outputs

## Citations

```bibtex
@inproceedings{hsu2023posterlayout,
  title={Posterlayout: A new benchmark and approach for content-aware visual-textual presentation layout},
  author={Hsu, Hsiao Yuan and He, Xiangteng and Peng, Yuxin and Kong, Hao and Zhang, Qing},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={6018--6026},
  year={2023}
}
```

## References

- **Paper**: [PosterLayout (Hsu et al., CVPR 2023)](https://arxiv.org/abs/2303.15937)
- **Reference Implementation**: [PosterLayout eval.py](https://github.com/PKU-ICST-MIPL/PosterLayout-CVPR2023/blob/main/eval.py#L105-L127)

## Related Metrics

- [Layout Overlap](../layout_overlap/): Evaluates element spacing and collisions
- [Layout Alignment](../layout_alignment/): Measures spatial organization
- [Layout Non-Alignment](../layout_non_alignment/): Detects alignment violations
- [Layout Utility](../layout_utility/): Evaluates space utilization
