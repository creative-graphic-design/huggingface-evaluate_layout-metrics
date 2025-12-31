---
title: Layout Non-Alignment
emoji: 🌍
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.36.1
app_file: app.py
pinned: false
---

# Layout Non-Alignment

## Description

The Layout Non-Alignment metric quantifies the extent of spatial non-alignment between layout elements. This metric evaluates layouts by detecting elements that break alignment patterns, providing insights into layout organization quality and visual coherence.

## What It Measures

This metric computes non-alignment scores that measure:

- **Alignment violations**: Elements that don't align with others along edges or centers
- **Spatial organization**: How consistently elements follow grid or alignment patterns
- **Visual disorder**: Degree of positional inconsistency between elements

The metric comes from PosterLayout (Hsu et al., CVPR 2023) and AC-GAN (Li et al., TVCG 2021) research, specifically designed for evaluating poster and graphic design layouts.

## Metric Details

- Analyzes element edge positions (left, right, top, bottom) to detect alignment patterns
- Computes delta (minimum distance) between element edges
- Applies logarithmic transformation to penalize near-misses more than obvious non-alignments
- Lower scores indicate better overall alignment (less non-alignment)

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
    "creative-graphic-design/layout-non-alignment",
    canvas_width=360,
    canvas_height=504
)

# Single layout processing
predictions = np.random.rand(1, 25, 4)  # (batch, max_elements, coordinates)
gold_labels = np.random.randint(0, 4, size=(1, 25))  # (batch, max_elements)
score = metric.compute(predictions=predictions, gold_labels=gold_labels)
print(score)
```

### Batch Processing Example

```python
import evaluate
import numpy as np

# Load the metric
metric = evaluate.load(
    "creative-graphic-design/layout-non-alignment",
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

- **predictions** (`list` of `lists` of `float`): Normalized bounding boxes in ltrb (left-top-right-bottom) format
- **gold_labels** (`list` of `lists` of `int`): Class labels for each element (0 = padding/invalid)

**Note**: Elements with `gold_labels == 0` are treated as padding and excluded from computation. Very small elements (< 0.1% of canvas area) are also filtered out.

## Returns

Returns a `float` value representing the non-alignment score.

## Interpretation

- **Lower is better**: Less non-alignment indicates better spatial organization
- **Value of 0**: Perfect alignment across all elements (rare in practice)
- **Typical range**: Varies based on layout complexity and density
- **Higher values**: More alignment violations, less organized layout

### Use Cases

- **Layout quality assessment**: Evaluate how well elements follow alignment principles
- **Generative model evaluation**: Compare alignment quality between different generation methods
- **Design feedback**: Identify layouts with poor spatial organization

### Key Insights

- **Professional designs** typically have lower non-alignment scores
- **Grid-based layouts** naturally achieve better alignment
- **Dense layouts** may have higher scores due to increased element interactions
- **Small violations** (near-alignments) are penalized more than obvious non-alignments

## Citations

```bibtex
@inproceedings{hsu2023posterlayout,
  title{Posterlayout: A new benchmark and approach for content-aware visual-textual presentation layout},
  author={Hsu, Hsiao Yuan and He, Xiangteng and Peng, Yuxin and Kong, Hao and Zhang, Qing},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={6018--6026},
  year={2023}
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
```

## References

- **Paper**: [PosterLayout (Hsu et al., CVPR 2023)](https://arxiv.org/abs/2303.15937)
- **Paper**: [Attribute-Conditioned Layout GAN (Li et al., TVCG 2021)](https://arxiv.org/abs/2009.05284)
- **Reference Implementation**: [PosterLayout eval.py](https://github.com/PKU-ICST-MIPL/PosterLayout-CVPR2023/blob/main/eval.py#L306-L339)

## Related Metrics

- [Layout Alignment](../layout_alignment/): Measures positive alignment between elements
- [Layout Validity](../layout_validity/): Checks basic validity constraints
- [Layout Overlay](../layout_overlay/): Measures element overlap (excluding underlay)
