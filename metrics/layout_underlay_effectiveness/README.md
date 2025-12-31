---
title: Layout Underlay Effectiveness
emoji: 👀
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.36.1
app_file: app.py
pinned: false
---

# Layout Underlay Effectiveness

## Description

The Layout Underlay Effectiveness metric evaluates how well underlay (decoration/background) elements are placed in a layout. Underlay elements should be positioned underneath other layout elements to serve their intended purpose as background or decorative layers. This metric measures what proportion of underlay elements correctly have other elements placed on top of them.

## What It Measures

This metric computes:

- **Valid underlay ratio**: Proportion of underlay elements that correctly have foreground elements on top
- **Underlay functionality**: Whether decoration elements serve their background purpose
- **Layer correctness**: How well the layout respects foreground/background hierarchy

Higher scores indicate better underlay placement where decoration elements properly support foreground content.

## Metric Details

Two calculation modes:

1. **Strict mode**: Underlay scores 1 if there's a non-underlay element completely inside it, 0 otherwise
2. **Loose mode**: Calculates ratio of intersection area to underlay area (ai/a2)

- Filters out text and other underlay elements when evaluating each underlay
- From PosterLayout (Hsu et al., CVPR 2023) for poster design evaluation
- Typical underlay class index: 3 (decoration/background)

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
    "creative-graphic-design/layout-underlay-effectiveness",
    canvas_width=360,
    canvas_height=504,
    text_label_index=1,
    decoration_label_index=3
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
    "creative-graphic-design/layout-underlay-effectiveness",
    canvas_width=360,
    canvas_height=504,
    text_label_index=1,
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
- **text_label_index** (`int`, optional, default=1): Class index for text elements
- **decoration_label_index** (`int`, optional, default=3): Class index for underlay/decoration elements

### Computation Parameters

- **predictions** (`list` of `lists` of `float`): Normalized bounding boxes in ltrb format (0.0 to 1.0)
- **gold_labels** (`list` of `lists` of `int`): Class labels for each element (0 = padding)

**Note**:

- Elements with label == 0 are treated as padding
- Text elements (text_label_index) and other underlays are excluded when checking each underlay
- Very small elements (< 0.1% of canvas) are filtered out

## Returns

Returns a dictionary containing:

- **underlay-effectiveness-strict** (`float`): Strict mode score (0.0 to 1.0)
- **underlay-effectiveness-loose** (`float`): Loose mode score (0.0 to 1.0)

## Interpretation

- **Higher is better** (range: 0.0 to 1.0)
- **Value of 1.0**: All underlay elements have foreground content on top (perfect)
- **Value of 0.7-1.0**: Most underlays are effective
- **Value of 0.5-0.7**: Some underlays are not effectively used
- **Value < 0.5**: Many underlays lack foreground content (poor design)
- **Value of 0.0**: No underlays have elements on top (problematic)

### Mode Differences

**Strict Mode**:

- Binary scoring (0 or 1 per underlay)
- Requires complete containment of a foreground element
- More conservative evaluation

**Loose Mode**:

- Continuous scoring based on overlap ratio
- Partial overlaps count proportionally
- More lenient evaluation

### Use Cases

- **Poster/presentation layout evaluation**: Ensure decoration layers function properly
- **Multi-layer designs**: Validate foreground/background hierarchy
- **Content-aware generation**: Assess whether models understand layer relationships
- **Design quality**: Identify layouts with ineffective underlay usage

### Key Insights

- **Purpose of underlay**: Should support foreground content, not exist independently
- **Layout hierarchy matters**: Underlay effectiveness indicates proper layer understanding
- **Design intent**: Some layouts intentionally have standalone decorative elements
- **Context-specific**: Most relevant for poster, flyer, and presentation layouts

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
- **Reference Implementation**: [PosterLayout eval.py](https://github.com/PKU-ICST-MIPL/PosterLayout-CVPR2023/blob/main/eval.py)

## Related Metrics

- [Layout Overlay](../layout_overlay/): Measures overlap excluding underlay
- [Layout Occlusion](../layout_occlusion/): Evaluates coverage of salient regions
- [Layout Validity](../layout_validity/): Checks basic validity constraints
