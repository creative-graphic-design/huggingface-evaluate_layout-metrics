---
title: Layout Overlay
emoji: 📊
colorFrom: gray
colorTo: green
sdk: gradio
sdk_version: 4.36.1
app_file: app.py
pinned: false
---

# Layout Overlay

## Description

The Layout Overlay metric measures the average IoU (Intersection over Union) of all pairs of layout elements, specifically excluding "underlay" or decoration elements. This metric is designed for poster and presentation layouts where underlay elements serve as backgrounds and should not be counted in overlap calculations.

## What It Measures

This metric computes:

- **Non-underlay overlap**: IoU between all pairs of foreground elements (text, images, logos)
- **Element collision**: How much non-decoration elements interfere with each other
- **Foreground placement quality**: Whether foreground elements are properly spaced

Underlay/decoration elements (like background shapes) are excluded from the calculation since they're intended to sit behind other elements.

## Metric Details

- Filters out decoration/underlay elements (typically class index 3 in PosterLayout)
- Removes invalid elements (< 0.1% of canvas area)
- Computes pairwise IoU for all remaining element pairs
- Returns average IoU across all overlapping pairs
- From PosterLayout (Hsu et al., CVPR 2023) for poster design evaluation

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
    "creative-graphic-design/layout-overlay",
    canvas_width=360,
    canvas_height=504,
    decoration_label_index=3  # underlay/decoration class
)

# Prepare data
predictions = np.random.rand(1, 25, 4)  # normalized ltrb coordinates
gold_labels = np.random.randint(0, 4, size=(1, 25))  # class labels
score = metric.compute(predictions=predictions, gold_labels=gold_labels)
print(score)
```

### Batch Processing Example

```python
import evaluate
import numpy as np

# Load the metric
metric = evaluate.load(
    "creative-graphic-design/layout-overlay",
    canvas_width=360,
    canvas_height=504,
    decoration_label_index=3
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
- **decoration_label_index** (`int`, optional, default=3): Class index for underlay/decoration elements to exclude

### Computation Parameters

- **predictions** (`list` of `lists` of `float`): Normalized bounding boxes in ltrb format (0.0 to 1.0)
- **gold_labels** (`list` of `lists` of `int`): Class labels for each element (0 = padding)

**Note**:

- Elements with label == 0 are treated as padding
- Elements with label == decoration_label_index are excluded (underlay)
- Very small elements (< 0.1% of canvas) are filtered out

## Returns

Returns a `float` value representing the average IoU of overlapping element pairs (excluding underlay).

## Interpretation

- **Lower is better** (range: 0.0 to 1.0)
- **Value of 0.0**: No overlap between foreground elements (ideal)
- **Value of 0.1-0.3**: Minor overlap, possibly acceptable in dense layouts
- **Value of 0.3-0.5**: Moderate overlap, may indicate placement issues
- **Value > 0.5**: Significant overlap, likely problematic

### Use Cases

- **Poster/presentation layout evaluation**: Ensure foreground elements don't overlap excessively
- **Content-aware design**: Evaluate layouts with distinct foreground and background layers
- **Layered designs**: Assess foreground element placement independent of decoration layers
- **Multi-layer layouts**: Focus on collision detection for primary content

### Key Insights

- **Underlay exclusion is important**: Decoration elements are meant to be behind others
- **Context-specific**: Appropriate for designs with clear foreground/background separation
- **Different from general overlap**: Focuses only on foreground element interactions
- **Use with related metrics**: Combine with underlay effectiveness for full picture

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
- **Reference Implementation**: [PosterLayout eval.py](https://github.com/PKU-ICST-MIPL/PosterLayout-CVPR2023/blob/main/eval.py#L205-L222)

## Related Metrics

- [Layout Overlap](../layout_overlap/): General overlap metric for all elements
- [Layout Underlay Effectiveness](../layout_underlay_effectiveness/): Evaluates underlay element placement
- [Layout Average IoU](../layout_average_iou/): IoU-based overlap for all elements
- [Layout Validity](../layout_validity/): Checks basic validity constraints
